"""
04_cross_validate.py  —  full-fold cross-validation with per-fold independent tuning
=====================================================================================
Rigorous full-fold CV: each fold is tuned INDEPENDENTLY on its training set (its own
measured CVs + its own alpha_obs / alpha_nsd) and then validated on the other three
batches it never saw. What is shared across folds is a single *selection RULE* for alpha,
NOT a single alpha value — so no value leaks across folds; only the method generalizes.

Two stages:

  --stage sweep    (per fold, on the training set)
     * calibrate the measured CVs (fixed-point -> NIV=1); save the full per-iteration NIV
       log, the final CVs, and the all-state trajectory + bands.
     * sweep alpha_obs (Asn/Glu) and alpha_nsd (7 NSDs) over their grids; for EVERY alpha
       save the full 17-state mean/std trajectory + uncertainty bands + metrics as pkl,
       plus per-alpha figures and an overlay-vs-alpha figure.
     During each sweep the other tier sits at a reference value; the exact picked pair is
     only applied at validation. Nothing auto-selected — you inspect and hand-pick.

  --stage validate --picks results_v1/picks.json
     * for each fold: load ITS calibrated CVs and ITS hand-picked (alpha_obs, alpha_nsd),
       apply to the held-out datasets, save all-state bands + metrics + grids.

Everything is saved as pkl for full offline recovery (re-plot / re-derive any statistic
without re-running). Output goes under --run (default: results_v1/).

picks.json format (fill in after inspecting the sweeps):
  { "P1": {"alpha_obs": 0.002, "alpha_nsd": 0.03},
    "P2": {"alpha_obs": 0.002, "alpha_nsd": 0.02}, ... }

Layout:
  results_v1/
    fold_<X>/cv/            NIV log, cv_final.json, trajectory pkl, convergence figure
    fold_<X>/alpha_obs/     per-alpha pkl (all 17 states) + figures + overlay
    fold_<X>/alpha_nsd/     per-alpha pkl (all 17 states) + figures + overlay
    fold_<X>/validation/    held-out all-state bands + grids + metrics  (validate stage)
    summary/                cross-fold tables + comparison figures       (validate stage)

Crash-safe / resumable at fold granularity (--resume). Runtime for the sweep stage is
large (~12 h for 4 folds at N=100); run one fold at a time with --train <ds>.

Usage (macOS venv):
  # Stage 1 — sweep + save everything, one fold at a time (accumulates):
  caffeinate -i ./.venv/bin/python scripts/04_cross_validate.py --stage sweep --train P4
  #   then --train P1 / P2 / P3   (or drop --train to do all four in one ~12h run)
  # ... inspect results_v1/fold_*/{alpha_obs,alpha_nsd}/figures, write results_v1/picks.json ...
  # Stage 2 — validate each fold at its picked alpha on its held-out sets:
  caffeinate -i ./.venv/bin/python scripts/04_cross_validate.py --stage validate --picks results_v1/picks.json
"""

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import warnings
import numpy as np
import pandas as pd
from scipy.linalg import LinAlgWarning
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

warnings.filterwarnings("ignore", category=LinAlgWarning)

import nsd_enkf.config as cfg
from nsd_enkf.data_loader import (
    select_datasets, load_dataset, get_initial_condition, build_schedule,
)
from nsd_enkf.model import compute_volume_results, model_step, simulate_dataset
from nsd_enkf.analysis import generate_measurement_ensembles
from nsd_enkf.enkf import EnsembleKalmanFilter, run_enkf_single_with_ensemble_diagnostics

# ── CLI ──────────────────────────────────────────────────────────────────────
p = argparse.ArgumentParser(description="Full-fold CV with per-fold independent tuning (sweep/validate)")
p.add_argument("--stage", required=True, choices=["sweep", "validate"])
p.add_argument("--folds", default="P1,P2,P3,P4")
p.add_argument("--scheme", default="rotate", choices=["rotate", "loo"])
p.add_argument("--train", default=None, help="restrict to the single fold training on this dataset")
p.add_argument("--run", default="results_v1")
p.add_argument("--picks", default=None, help="validate stage: json {fold: {alpha_obs, alpha_nsd}}")
p.add_argument("--ensemble-size", default=cfg.ENSEMBLE_SIZE, type=int)
p.add_argument("--cv-iters", default=8, type=int)
p.add_argument("--cv-max", default=0.006, type=float)
p.add_argument("--cv-min", default=1e-4, type=float)
p.add_argument("--cv-tol", default=0.15, type=float)
p.add_argument("--obs-alphas", default="0.001,0.002,0.004,0.006,0.008,0.01")
p.add_argument("--nsd-alphas", default="0.005,0.0075,0.01,0.02,0.03,0.04")
p.add_argument("--ref-alpha-obs", default=0.002, type=float,
               help="fixed alpha_obs during CV calibration and as the other tier while "
                    "sweeping alpha_nsd (matters: Asn/Glu are upstream)")
p.add_argument("--ref-alpha-nsd", default=0.02, type=float,
               help="fixed alpha_nsd during CV calibration and as the other tier while "
                    "sweeping alpha_obs (NSDs are downstream, so essentially inert here)")
p.add_argument("--seed", default=42, type=int)
p.add_argument("--traj-down", default=20, type=int)
p.add_argument("--no-plots", action="store_true")
p.add_argument("--resume", action="store_true")
args = p.parse_args()

DATASETS = [d for d in args.folds.split(",") if d]
ENS = args.ensemble_size
OBS_GRID = [float(a) for a in args.obs_alphas.split(",")]
NSD_GRID = [float(a) for a in args.nsd_alphas.split(",")]
RUN = cfg.PROJECT_ROOT / args.run if not Path(args.run).is_absolute() else Path(args.run)
DOWN = max(int(args.traj_down), 1)

meas_names = cfg.MEASURED_STATES
MEAS = cfg.MEAS_NUM
n_nsd = 7
nsd_state_idx = list(range(cfg.STATE_NUM - n_nsd, cfg.STATE_NUM))
nsd_names = [cfg.STATE_NAMES[i] for i in nsd_state_idx]
ASN = cfg.STATE_NAMES.index("Asn")
REF_OBS = args.ref_alpha_obs
REF_NSD = args.ref_alpha_nsd

# ── Fixed grids / matrices ────────────────────────────────────────────────────
time_grid = np.arange(cfg.DT, cfg.T_END + cfg.DT, cfg.DT)
step_len = np.full(len(time_grid), cfg.DT)
N_model = int(cfg.T_END / cfg.DT)
T_model = np.linspace(0, cfg.T_END, N_model + 1)
dt_kf = cfg.DT
N_kf = len(T_model) - 1
var_meas = np.array(list(cfg.MEASUREMENT_NOISE_VAR.values()))
R = np.diag(var_meas[:MEAS])
H = np.hstack((np.eye(MEAS), np.zeros((MEAS, cfg.STATE_NUM - MEAS))))
no_update_indices = {cfg.STATE_NAMES.index(s) for s in cfg.NO_UPDATE_STATES}
clip_indices = {cfg.STATE_NAMES.index(s) for s in cfg.CLIP_STATES}
scale_vec = np.zeros(cfg.STATE_NUM)
for s, sc in cfg.PROCESS_NOISE_SCALE.items():
    scale_vec[cfg.STATE_NAMES.index(s)] = sc
obs_idx = [cfg.STATE_NAMES.index(s) for s in getattr(cfg, "ALPHA_OBS_STATES", ["Asn", "Glu"])]
P0_meas = np.array([cfg.MEASUREMENT_NOISE_VAR.get(s, 0.0) for s in cfg.STATE_NAMES])
T_meas = np.array(cfg.T_MEAS_FIXED)
meas_grid_idx = [min(int(round(t / dt_kf)) + 1, N_kf) for t in T_meas]
time_steps_A = [round(i * dt_kf, 2) for i in range(N_kf)]
meas_time_to_index = {round(t, 2): i for i, t in enumerate(T_meas.tolist())}
volume_results = compute_volume_results(select_datasets(*DATASETS), cfg.INITIAL_VOLUMES,
                                        build_schedule, step_len)


def build_Q(alpha_obs, alpha_nsd):
    a = np.full(cfg.STATE_NUM, float(alpha_nsd))
    for i in obs_idx:
        a[i] = float(alpha_obs)
    return np.diag((a * scale_vec) ** 2)


def P0_from(Q):
    d = np.diag(Q).copy(); d[:MEAS] = P0_meas[:MEAS]
    return np.diag(d)


_cache = {}
def ds_data(name):
    if name not in _cache:
        d = load_dataset(name)
        _, x0 = get_initial_condition(d["met_df"], d["nsd_df"])
        sm = d["set_meas"].astype(float); se = d["set_meas_errorbar"].astype(float)
        asn_col = sm.shape[1] - 1
        np.random.seed(args.seed)
        mens = generate_measurement_ensembles(select_datasets(name), load_dataset,
                                              MEAS, ENS, var_meas)[name]
        Fin, Fout, Gf, Uf = build_schedule(name)
        model = np.vstack([x0, simulate_dataset(x0, Fin, Fout, Gf, Uf,
                                                volume_results[name][1:], time_grid, step_len,
                                                name=name)])
        _cache[name] = dict(
            x0=x0, set_meas=sm[:, :MEAS], asn_meas=sm[:, asn_col], asn_err=se[:, asn_col],
            set_err=se, mens=mens, controls=(Fin, Fout, Gf, Uf, volume_results[name][1:]), model=model,
            nsd_meas=pd.DataFrame(d["NSD_meas"]).apply(pd.to_numeric, errors="coerce").to_numpy(),
            nsd_err=pd.DataFrame(d["NSD_meas_errorbar"]).apply(pd.to_numeric, errors="coerce").to_numpy())
    return _cache[name]


def enkf_pass(name, cv_idx, Q, P0):
    np.random.seed(args.seed)
    _, std_traj, mean_traj = run_enkf_single_with_ensemble_diagnostics(
        dataset_name=name, load_dataset_fn=load_dataset, build_schedule_fn=build_schedule,
        state_init=ds_data(name)["x0"], volume_results=volume_results,
        set_meas_ens=ds_data(name)["mens"], T_meas=T_meas,
        state_num=cfg.STATE_NUM, meas_num=MEAS, ensemble_size=ENS,
        Q=Q, R=R, H=H, dt_kf=dt_kf, N_kf=N_kf, P0=P0,
        process_noise_cv=dict(cv_idx), no_update_indices=set(no_update_indices),
        clip_indices=set(clip_indices))
    return mean_traj, std_traj


def niv_sqnorm(name, cv_idx, Q, P0):
    dd = ds_data(name); Fin, Fout, Gf, Uf, V_traj = dd["controls"]
    np.random.seed(args.seed)
    enkf = EnsembleKalmanFilter(cfg.STATE_NUM, MEAS)
    enkf.x = dd["x0"].copy(); enkf.Q = Q.copy(); enkf.R = R.copy(); enkf.H = H.copy()
    enkf.fx = model_step; enkf.dt = dt_kf; enkf.process_noise_cv = dict(cv_idx)
    enkf.no_update_indices = set(no_update_indices); enkf.clip_indices = set(clip_indices)
    enkf.create_ensemble(ENS, P0)
    sq = [[] for _ in range(MEAS)]
    for idx_A, step_A in enumerate(time_steps_A):
        enkf.predict({"Fin": Fin[idx_A], "Fout": Fout[idx_A], "V": V_traj[idx_A],
                      "Gal_feed": Gf[idx_A], "Urd_feed": Uf[idx_A]})
        if step_A in meas_time_to_index:
            b = meas_time_to_index[step_A]
            Z = enkf.X @ enkf.H.T; zmean = Z.mean(axis=0); Ez = Z - zmean
            S = (Ez.T @ Ez) / (ENS - 1) + R; dvec = dd["set_meas"][b] - zmean
            for j in range(MEAS):
                if not np.isnan(dd["set_meas"][b, j]) and S[j, j] > 0:
                    sq[j].append(dvec[j] ** 2 / S[j, j])
            enkf.update(dd["mens"][b])
    return sq


def calibrate_cv(train_list):
    """Fixed-point CV -> NIV=1 (alphas fixed at reference); returns cv, niv_history, flags."""
    cv = {s: cfg.PROCESS_NOISE_CV[s] for s in meas_names}
    capped = {s: False for s in meas_names}; floored = {s: False for s in meas_names}
    Q = build_Q(REF_OBS, REF_NSD); P0 = P0_from(Q)
    hist = []
    for it in range(args.cv_iters):
        cv_idx = {cfg.STATE_NAMES.index(s): cv[s] for s in meas_names}
        pooled = [[] for _ in range(MEAS)]
        for name in train_list:
            sq = niv_sqnorm(name, cv_idx, Q, P0)
            for j in range(MEAS):
                pooled[j] += sq[j]
        niv = np.array([np.mean(v) if v else np.nan for v in pooled])
        hist.append([float(v) for v in niv])
        done = True
        for j, s in enumerate(meas_names):
            if np.isnan(niv[j]):
                continue
            new = cv[s] * np.sqrt(niv[j]); new_c = float(np.clip(new, args.cv_min, args.cv_max))
            capped[s] = bool(new > args.cv_max); floored[s] = bool(new < args.cv_min)
            if abs(niv[j] - 1.0) > args.cv_tol:
                done = False
            cv[s] = new_c
        print(f"      CV iter {it}: NIV=[" + ", ".join(f"{v:.2f}" for v in niv) + "]", flush=True)
        if done:
            break
    niv_final = {s: hist[-1][j] for j, s in enumerate(meas_names)}
    return cv, hist, niv_final, capped, floored


def nsd_asn_metrics(name, mean_traj, std_traj):
    dd = ds_data(name); out = {}
    for col, si in enumerate(nsd_state_idx):
        meas = dd["nsd_meas"][:, col]; m = mean_traj[meas_grid_idx, si]; s = std_traj[meas_grid_idx, si]
        valid = ~np.isnan(meas) & (s > 0)
        if valid.sum() == 0:
            out[nsd_names[col]] = dict(rmse=np.nan, nrmse=np.nan, cov=np.nan, ss=np.nan); continue
        err = meas[valid] - m[valid]; rmse = float(np.sqrt(np.mean(err ** 2)))
        norm = float(np.mean(np.abs(meas[valid]))) or 1.0
        out[nsd_names[col]] = dict(rmse=rmse, nrmse=rmse / norm,
                                   cov=100.0 * float(np.mean(np.abs(err) <= 2 * s[valid])),
                                   ss=(float(np.mean(s[valid]) / rmse) if rmse > 0 else np.nan))
    meas = dd["asn_meas"]; m = mean_traj[meas_grid_idx, ASN]; s = std_traj[meas_grid_idx, ASN]
    valid = ~np.isnan(meas) & (s > 0); err = meas[valid] - m[valid]
    rmse = float(np.sqrt(np.mean(err ** 2))); norm = float(np.mean(np.abs(meas[valid]))) or 1.0
    out["Asn"] = dict(rmse=rmse, nrmse=rmse / norm,
                      cov=100.0 * float(np.mean(np.abs(err) <= 2 * s[valid])),
                      ss=(float(np.mean(s[valid]) / rmse) if rmse > 0 else np.nan))
    return out


def folds():
    if args.train:
        if args.train not in DATASETS:
            raise SystemExit(f"--train {args.train} must be one of --folds {DATASETS}")
        base = [args.train]
    else:
        base = DATASETS
    if args.scheme == "rotate":
        return [(d, [d], [x for x in DATASETS if x != d]) for d in base]
    return [(d, [x for x in DATASETS if x != d], [d]) for d in base]  # loo


def save_pkl(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(obj, f)
    os.replace(tmp, path)


def meas_by_state(name):
    dd = ds_data(name)
    mbs = {i: (dd["set_meas"][:, i], dd["set_err"][:, i]) for i in range(MEAS)}
    mbs[ASN] = (dd["asn_meas"], dd["asn_err"])
    for j in range(n_nsd):
        mbs[nsd_state_idx[j]] = (dd["nsd_meas"][:, j], dd["nsd_err"][:, j])
    return mbs


LEGEND = [Line2D([0], [0], color="red", lw=1.6, label="Open-loop model"),
          Line2D([0], [0], color="steelblue", lw=2, label="EnKF mean"),
          Patch(facecolor="steelblue", alpha=0.30, label=r"$\pm1\sigma$"),
          Patch(facecolor="steelblue", alpha=0.15, label=r"$\pm2\sigma$"),
          Line2D([0], [0], color="darkorange", marker="o", lw=0, ms=6, label="Measurements")]


def allstate_grid(name, mt, st, model, title, out):
    tds = T_model[::DOWN]; mbs = meas_by_state(name)
    fig, axes = plt.subplots(5, 4, figsize=(20, 15)); axes = axes.flatten()
    for si in range(cfg.STATE_NUM):
        ax = axes[si]; m = mt[::DOWN, si]; s = st[::DOWN, si]
        ax.fill_between(tds, np.maximum(m - 2 * s, 0), m + 2 * s, color="steelblue", alpha=0.15)
        ax.fill_between(tds, np.maximum(m - s, 0), m + s, color="steelblue", alpha=0.30)
        ax.plot(tds, m, color="steelblue", lw=2)
        ax.plot(tds, model[::DOWN, si], color="red", lw=1.6)
        if si in mbs:
            v, e = mbs[si]
            ax.errorbar(T_meas, v, yerr=e, fmt="o", color="darkorange", ms=4, capsize=2, elinewidth=1, zorder=5)
        ax.set_title(cfg.STATE_NAMES[si] + ("" if si in mbs else "  (no meas)"), fontsize=11, fontweight="bold")
        ax.set_xlabel("Time (h)", fontsize=9); ax.grid(alpha=0.15)
    for k in range(cfg.STATE_NUM, len(axes)):
        axes[k].set_visible(False)
    fig.legend(handles=LEGEND, loc="lower center", ncol=5, fontsize=12, frameon=False, bbox_to_anchor=(0.5, -0.005))
    fig.suptitle(title, fontsize=15, fontweight="bold", y=1.005)
    plt.tight_layout(rect=[0, 0.03, 1, 1]); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)


def overlay(train, kind, per_alpha, out):
    """Overlay the +/-2 sigma envelopes across alpha for the scored states (NSDs or Asn)."""
    states = nsd_names if kind == "nsd" else ["Asn"]
    idxs = nsd_state_idx if kind == "nsd" else [ASN]
    alphas = sorted(per_alpha)
    colors = plt.cm.viridis(np.linspace(0, 1, len(alphas)))
    tds = T_model[::DOWN]; mbs = meas_by_state(train)
    ncol = 4 if kind == "nsd" else 1
    nrow = int(np.ceil(len(states) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.5 * ncol, 3.6 * nrow)); axes = np.atleast_1d(axes).flatten()
    for j, (nm, si) in enumerate(zip(states, idxs)):
        ax = axes[j]
        for i, a in enumerate(alphas):
            mt, st = per_alpha[a]
            ax.plot(tds, (mt[:, si] + 2 * st[:, si])[::DOWN], color=colors[i], lw=1.1)
            ax.plot(tds, np.maximum(mt[:, si] - 2 * st[:, si], 0)[::DOWN], color=colors[i], lw=1.1)
        if si in mbs:
            v, e = mbs[si]
            ax.errorbar(T_meas, v, yerr=e, fmt="o", color="darkorange", ms=4, capsize=2, zorder=6)
        ax.set_title(nm, fontsize=10, fontweight="bold"); ax.set_xlabel("Time (h)"); ax.set_ylabel("mM"); ax.grid(alpha=0.15)
    for k in range(len(states), len(axes)):
        axes[k].set_visible(False)
    handles = [Line2D([0], [0], color=colors[i], lw=4, label=f"α={alphas[i]:g}") for i in range(len(alphas))]
    handles += [Line2D([0], [0], color="darkorange", marker="o", lw=0, ms=6, label="measurements")]
    fig.legend(handles=handles, loc="lower center", ncol=len(alphas) + 1, fontsize=10, frameon=False, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle(f"{kind.upper()} ±2σ bands vs alpha — train {train}", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0.05, 1, 0.97]); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)


# ── Stage: sweep ──────────────────────────────────────────────────────────────
def sweep_alpha(train, cv_idx, kind, out_dir):
    grid = OBS_GRID if kind == "obs" else NSD_GRID
    per_alpha = {}
    for a in grid:
        Q = build_Q(a, REF_NSD) if kind == "obs" else build_Q(REF_OBS, a)
        mt, st = enkf_pass(train, cv_idx, Q, P0_from(Q))
        met = nsd_asn_metrics(train, mt, st)
        per_alpha[a] = (mt, st)
        save_pkl({"train": train, "kind": kind, "alpha": a, "T": T_model,
                  "state_names": list(cfg.STATE_NAMES),
                  "mean_trajectory": mt, "std_trajectory": st,
                  "band_2sigma_lo": np.maximum(mt - 2 * st, 0.0), "band_2sigma_hi": mt + 2 * st,
                  "model_trajectory": ds_data(train)["model"], "metrics": met},
                 out_dir / "pkl" / f"alpha_{a:g}.pkl")
        sc = met["Asn"] if kind == "obs" else {}
        tag = f"Asn cov={met['Asn']['cov']:.0f}%" if kind == "obs" else \
              f"meanNSD NRMSE={np.nanmean([met[n]['nrmse'] for n in nsd_names]):.2f} " \
              f"cov={np.nanmean([met[n]['cov'] for n in nsd_names]):.0f}%"
        print(f"      {kind} alpha={a:g}: {tag}", flush=True)
    if not args.no_plots:
        overlay(train, kind, per_alpha, out_dir / "figures" / f"{kind}_bands_vs_alpha.png")


def stage_sweep():
    for fid, train_list, _ in folds():
        fdir = RUN / f"fold_{fid}"
        done_marker = fdir / "cv" / "cv_final.json"
        if args.resume and (fdir / "alpha_nsd" / "pkl" / f"alpha_{NSD_GRID[-1]:g}.pkl").exists():
            print(f"\n[sweep] fold {fid}: complete, skip (resume)."); continue
        print(f"\n[sweep] fold {fid}: train={train_list}")
        # 1) CV calibration
        cv, niv_hist, niv_final, capped, floored = calibrate_cv(train_list)
        cv_idx = {cfg.STATE_NAMES.index(s): cv[s] for s in meas_names}
        save_pkl_json = {"train": train_list, "cv": cv, "niv_final": niv_final,
                         "niv_history": niv_hist, "capped": capped, "floored": floored,
                         "cv_max": args.cv_max, "cv_min": args.cv_min}
        (fdir / "cv").mkdir(parents=True, exist_ok=True)
        json.dump(save_pkl_json, open(fdir / "cv" / "cv_final.json", "w"), indent=2)
        mt, st = enkf_pass(train_list[0], cv_idx, build_Q(REF_OBS, REF_NSD), P0_from(build_Q(REF_OBS, REF_NSD)))
        save_pkl({"train": train_list, "cv": cv, "mean_trajectory": mt, "std_trajectory": st,
                  "model_trajectory": ds_data(train_list[0])["model"], "T": T_model,
                  "state_names": list(cfg.STATE_NAMES)}, fdir / "cv" / "cv_trajectory.pkl")
        if not args.no_plots:
            hist = np.array(niv_hist)
            fig, ax = plt.subplots(figsize=(9, 5.5))
            for j, s in enumerate(meas_names):
                ax.plot(range(hist.shape[0]), hist[:, j], marker="o", ms=4, label=s)
            ax.axhline(1.0, color="k", ls="--", lw=1); ax.set_xlabel("iteration"); ax.set_ylabel("NIV")
            ax.set_title(f"CV calibration (train {fid})"); ax.legend(ncol=4, fontsize=8); ax.grid(alpha=0.2)
            (fdir / "cv" / "figures").mkdir(parents=True, exist_ok=True)
            fig.tight_layout(); fig.savefig(fdir / "cv" / "figures" / "niv_convergence.png", dpi=150); plt.close(fig)
        # 2) alpha_obs sweep, 3) alpha_nsd sweep (both on the training set, other tier at reference)
        sweep_alpha(train_list[0], cv_idx, "obs", fdir / "alpha_obs")
        sweep_alpha(train_list[0], cv_idx, "nsd", fdir / "alpha_nsd")
        print(f"[sweep] fold {fid}: done -> {fdir}")
    # write a picks.json template if none exists
    tmpl = RUN / "picks.json"
    if not tmpl.exists():
        json.dump({fid: {"alpha_obs": REF_OBS, "alpha_nsd": REF_NSD} for fid, _, _ in folds()},
                  open(tmpl, "w"), indent=2)
        print(f"\nTemplate written: {tmpl}  (edit with your hand-picked alphas, then --stage validate)")


# ── Stage: validate ───────────────────────────────────────────────────────────
def stage_validate():
    if not args.picks:
        raise SystemExit("--stage validate needs --picks results_v1/picks.json")
    picks = json.load(open(args.picks))
    summary = {"scheme": args.scheme, "folds": {}}
    (RUN / "summary").mkdir(parents=True, exist_ok=True)
    for fid, train_list, val_list in folds():
        fdir = RUN / f"fold_{fid}"
        cvj = json.load(open(fdir / "cv" / "cv_final.json"))
        cv = cvj["cv"]; cv_idx = {cfg.STATE_NAMES.index(s): cv[s] for s in meas_names}
        a_obs = float(picks[fid]["alpha_obs"]); a_nsd = float(picks[fid]["alpha_nsd"])
        Q = build_Q(a_obs, a_nsd); P0 = P0_from(Q)
        print(f"\n[validate] fold {fid}: train={train_list} α_obs={a_obs:g} α_nsd={a_nsd:g} -> {val_list}")
        heldout = {}
        for name in val_list:
            mt, st = enkf_pass(name, cv_idx, Q, P0)
            met = nsd_asn_metrics(name, mt, st); heldout[name] = met
            save_pkl({"fold": fid, "train": train_list, "held_out": name,
                      "cv": cv, "alpha_obs": a_obs, "alpha_nsd": a_nsd, "T": T_model,
                      "state_names": list(cfg.STATE_NAMES),
                      "mean_trajectory": mt, "std_trajectory": st,
                      "band_2sigma_lo": np.maximum(mt - 2 * st, 0.0), "band_2sigma_hi": mt + 2 * st,
                      "model_trajectory": ds_data(name)["model"], "metrics": met},
                     fdir / "validation" / f"heldout_{name}.pkl")
            if not args.no_plots:
                allstate_grid(name, mt, st, ds_data(name)["model"],
                              f"Validate — trained {fid} (α_nsd={a_nsd:g}, α_obs={a_obs:g}), held-out {name}",
                              fdir / "validation" / "figures" / f"heldout_{name}.png")
            print(f"      held-out {name}: mean NSD NRMSE="
                  f"{np.nanmean([met[n]['nrmse'] for n in nsd_names]):.3f} cov="
                  f"{np.nanmean([met[n]['cov'] for n in nsd_names]):.0f}%  Asn NRMSE={met['Asn']['nrmse']:.3f}")
        summary["folds"][fid] = {"train": train_list, "validate": val_list,
                                 "alpha_obs": a_obs, "alpha_nsd": a_nsd, "cv": cv, "heldout": heldout}
        json.dump(summary, open(RUN / "summary" / "validation_summary.json", "w"), indent=2)
    print(f"\n[validate] done. Summary: {RUN / 'summary' / 'validation_summary.json'}")


# ── Dispatch ──────────────────────────────────────────────────────────────────
print("=" * 78)
print(f"Full-fold CV  stage={args.stage}  scheme={args.scheme}  N={ENS}  run={RUN.name}"
      + (f"  train={args.train}" if args.train else ""))
print("=" * 78)
if args.stage == "sweep":
    stage_sweep()
else:
    stage_validate()
