"""
04_cross_validate.py  —  Stage 4/5: generalized full-fold cross-validation
===========================================================================
Cross-validate the tuning *procedure* across the four batches to show it generalizes
(rebuts the "lack of methodological validation" review point). For each fold the filter
is (re-)tuned on the training data and then evaluated on the held-out data it never saw;
we report parameter stability across folds and held-out soft-sensor accuracy/coverage.

Fold schemes (`--scheme`):
  rotate : each dataset takes a turn as the SOLE training set; validate on the other 3.
           (Harder test — tune on one batch, must generalize to three.)
  loo    : leave-one-out — train on the other 3 (pooled), validate on the held-out 1.
           (Conventional k-fold; more training data per fold.)

Re-tune modes (`--retune`, run one or `both` to compare):
  cv  (A): per fold, re-calibrate ONLY the measured-state CVs (automated NIV=1, cap
           CV_MAX) on the training data; hold the design constants fixed (alpha_obs,
           alpha_nsd from config). This is the honest "does our automated calibration
           generalize" test — the alphas were physical/bioprocessing choices, part of the
           method, not per-dataset fits.
  all (B): additionally auto-select alpha_nsd (min mean NSD NRMSE) and alpha_obs (min Asn
           NRMSE) on the training data. Fully data-driven per fold; heavier, and its alpha
           selection differs from the judgment actually used — included for comparison.

R stays pooled across P1-P4 (assay replicate precision, a property of the measurement
protocol, not of a batch — see docs/tuning_strategy.md Stage 0).

Held-out scoring targets the soft-sensor deliverables — the unmeasured states (7 NSDs +
Asn); measured states are assimilated on the held-out batch so their fit is not the test.
Per state: RMSE, NRMSE (RMSE/mean meas), 2-sigma coverage, spread-skill (mean std/RMSE).

Crash-safe: each (mode, fold) is checkpointed to summary.json as it finishes; --resume
skips completed folds. Runtime is large (many EnKF passes) — see below.

Outputs (results/<run>/<mode>/):
  summary.json                      per-fold tuned params + held-out metrics
  fold_<id>.pkl                     tuned params + all-state mean/std/bands per held-out ds
  figures/cv_param_stability.png    tuned CVs (and alphas for B) across folds
  figures/cv_heldout_metrics.png    held-out NSD/Asn NRMSE + coverage across folds
And when both modes run:  figures/compare_A_vs_B.png,  comparison.json

Runtime (N=100, ~9 min/EnKF pass):
  rotate, retune=cv  : 4 folds x (~8 calib + 3 eval)      ~= 44 passes ~= 6.6 h
  rotate, retune=all : + alpha grids per fold             ~= 92 passes ~= 14 h
  both               : ~= 20 h  -> run with caffeinate, --resume friendly

Usage (macOS venv) — run A and B on SEPARATE days to keep each run short; the A-vs-B
comparison is assembled from whatever mode summaries are already on disk:
  mkdir -p results/cross_validation
  # day 1 — mode A (~6.6 h):
  caffeinate -i ./.venv/bin/python scripts/04_cross_validate.py --scheme rotate --retune cv \
      2>&1 | tee results/cross_validation/run_cv.log
  # day 2 — mode B (~14 h); also emits the A-vs-B comparison since A is on disk:
  caffeinate -i ./.venv/bin/python scripts/04_cross_validate.py --scheme rotate --retune all \
      2>&1 | tee results/cross_validation/run_all.log
  # Finer still — ONE fold per run (mode A single fold ~1.6 h), any order, accumulating:
  caffeinate -i ./.venv/bin/python scripts/04_cross_validate.py --retune cv --train P4
  #   then --train P1, --train P2, --train P3 (separate runs -> same summary.json)
  # (--retune both runs A+B in one ~20 h shot; --resume continues an interrupted run)
  # quick structural preview (small N, few iters) — NOT for real results:
  ./.venv/bin/python scripts/04_cross_validate.py --scheme rotate --retune cv \
      --folds P3,P4 --ensemble-size 15 --cv-iters 2 --no-plots
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

warnings.filterwarnings("ignore", category=LinAlgWarning)

import nsd_enkf.config as cfg
from nsd_enkf.data_loader import (
    select_datasets, load_dataset, get_initial_condition, build_schedule,
)
from nsd_enkf.model import compute_volume_results, model_step, simulate_dataset
from nsd_enkf.analysis import generate_measurement_ensembles
from nsd_enkf.enkf import EnsembleKalmanFilter, run_enkf_single_with_ensemble_diagnostics

# ── CLI ──────────────────────────────────────────────────────────────────────
p = argparse.ArgumentParser(description="Generalized full-fold cross-validation of the tuning procedure")
p.add_argument("--folds", default="P1,P2,P3,P4", help="datasets participating in the CV")
p.add_argument("--scheme", default="rotate", choices=["rotate", "loo"])
p.add_argument("--train", default=None,
               help="run ONLY the single fold that trains on THIS dataset (validate on all "
                    "others). Lets you break the CV into one short ~1.6h run at a time, any "
                    "order; folds accumulate into the same summary.json across runs.")
p.add_argument("--retune", default="cv", choices=["cv", "all", "both"],
               help="cv=A (~6.6h), all=B (~14h), both=one shot (~20h). Run 'cv' and 'all' "
                    "on separate days — the A-vs-B comparison is built from whatever is on disk.")
p.add_argument("--ensemble-size", default=cfg.ENSEMBLE_SIZE, type=int)
p.add_argument("--cv-iters", default=8, type=int, help="max CV fixed-point iterations per fold")
p.add_argument("--cv-max", default=0.006, type=float)
p.add_argument("--cv-min", default=1e-4, type=float)
p.add_argument("--cv-tol", default=0.15, type=float)
p.add_argument("--nsd-alphas", default="0.005,0.0075,0.01,0.02,0.03,0.04", help="alpha grid for retune=all")
p.add_argument("--obs-alphas", default="0.001,0.002,0.004,0.006,0.008,0.01", help="alpha grid for retune=all")
p.add_argument("--seed", default=42, type=int)
p.add_argument("--run", default="cross_validation")
p.add_argument("--traj-down", default=20, type=int)
p.add_argument("--no-plots", action="store_true")
p.add_argument("--resume", action="store_true", help="skip (mode, fold) already in summary.json")
args = p.parse_args()

DATASETS = [d for d in args.folds.split(",") if d]
MODES = ["cv", "all"] if args.retune == "both" else [args.retune]
ENS = args.ensemble_size
NSD_GRID = [float(a) for a in args.nsd_alphas.split(",")]
OBS_GRID = [float(a) for a in args.obs_alphas.split(",")]
RESULTS_DIR = cfg.PROJECT_ROOT / "results" / args.run

meas_names = cfg.MEASURED_STATES
MEAS = cfg.MEAS_NUM
n_nsd = 7
nsd_state_idx = list(range(cfg.STATE_NUM - n_nsd, cfg.STATE_NUM))
nsd_names = [cfg.STATE_NAMES[i] for i in nsd_state_idx]
ASN = cfg.STATE_NAMES.index("Asn")

print("=" * 78)
print(f"Full-fold cross-validation  scheme={args.scheme}  retune={args.retune}  N={ENS}")
print(f"  datasets={DATASETS}  (R pooled)")
print("=" * 78)

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
    d = np.diag(Q).copy()
    d[:MEAS] = P0_meas[:MEAS]
    return np.diag(d)


# ── Per-dataset static inputs (cached) ────────────────────────────────────────
_cache = {}
def ds_data(name):
    if name not in _cache:
        d = load_dataset(name)
        _, x0 = get_initial_condition(d["met_df"], d["nsd_df"])
        set_meas_full = d["set_meas"].astype(float)
        set_err = d["set_meas_errorbar"].astype(float)
        asn_col = set_meas_full.shape[1] - 1
        np.random.seed(args.seed)
        mens = generate_measurement_ensembles(select_datasets(name), load_dataset,
                                              MEAS, ENS, var_meas)[name]
        Fin, Fout, Gf, Uf = build_schedule(name)
        model = np.vstack([x0, simulate_dataset(x0, Fin, Fout, Gf, Uf,
                                                volume_results[name][1:], time_grid, step_len,
                                                name=name)])
        _cache[name] = dict(
            x0=x0, set_meas=set_meas_full[:, :MEAS], set_err=set_err,
            asn_meas=set_meas_full[:, asn_col], asn_err=set_err[:, asn_col],
            nsd_meas=pd.DataFrame(d["NSD_meas"]).apply(pd.to_numeric, errors="coerce").to_numpy(),
            nsd_err=pd.DataFrame(d["NSD_meas_errorbar"]).apply(pd.to_numeric, errors="coerce").to_numpy(),
            mens=mens, controls=(Fin, Fout, Gf, Uf, volume_results[name][1:]), model=model,
        )
    return _cache[name]


# ── EnKF passes ────────────────────────────────────────────────────────────
def enkf_pass(name, cv_idx, Q, P0):
    """Diagnostic pass -> (mean_traj, std_traj) for all states."""
    np.random.seed(args.seed)
    _, std_traj, mean_traj = run_enkf_single_with_ensemble_diagnostics(
        dataset_name=name, load_dataset_fn=load_dataset, build_schedule_fn=build_schedule,
        state_init=ds_data(name)["x0"], volume_results=volume_results,
        set_meas_ens=ds_data(name)["mens"], T_meas=T_meas,
        state_num=cfg.STATE_NUM, meas_num=MEAS, ensemble_size=ENS,
        Q=Q, R=R, H=H, dt_kf=dt_kf, N_kf=N_kf, P0=P0,
        process_noise_cv=dict(cv_idx), no_update_indices=set(no_update_indices),
        clip_indices=set(clip_indices),
    )
    return mean_traj, std_traj


def niv_sqnorm(name, cv_idx, Q, P0):
    """One EnKF pass; return per-measured-state list of squared normalised innovations
    (raw, so they can be pooled across several training datasets)."""
    dd = ds_data(name)
    Fin, Fout, Gf, Uf, V_traj = dd["controls"]
    np.random.seed(args.seed)
    enkf = EnsembleKalmanFilter(cfg.STATE_NUM, MEAS)
    enkf.x = dd["x0"].copy(); enkf.Q = Q.copy(); enkf.R = R.copy(); enkf.H = H.copy()
    enkf.fx = model_step; enkf.dt = dt_kf
    enkf.process_noise_cv = dict(cv_idx)
    enkf.no_update_indices = set(no_update_indices); enkf.clip_indices = set(clip_indices)
    enkf.create_ensemble(ENS, P0)
    sq = [[] for _ in range(MEAS)]
    for idx_A, step_A in enumerate(time_steps_A):
        enkf.predict({"Fin": Fin[idx_A], "Fout": Fout[idx_A], "V": V_traj[idx_A],
                      "Gal_feed": Gf[idx_A], "Urd_feed": Uf[idx_A]})
        if step_A in meas_time_to_index:
            b = meas_time_to_index[step_A]
            Z = enkf.X @ enkf.H.T
            zmean = Z.mean(axis=0); Ez = Z - zmean
            S = (Ez.T @ Ez) / (ENS - 1) + R
            dvec = dd["set_meas"][b] - zmean
            for j in range(MEAS):
                if not np.isnan(dd["set_meas"][b, j]) and S[j, j] > 0:
                    sq[j].append(dvec[j] ** 2 / S[j, j])
            enkf.update(dd["mens"][b])
    return sq


# ── Tuning on a training set ─────────────────────────────────────────────────
def calibrate_cv(train_list):
    """Fixed-point CV -> NIV=1, pooling innovations over all training datasets."""
    cv = {s: cfg.PROCESS_NOISE_CV[s] for s in meas_names}
    capped = {s: False for s in meas_names}; floored = {s: False for s in meas_names}
    Q = build_Q(cfg.PROCESS_NOISE_ALPHA_OBS, cfg.PROCESS_NOISE_ALPHA)  # alphas fixed here
    P0 = P0_from(Q)
    niv = np.full(MEAS, np.nan)
    for it in range(args.cv_iters):
        cv_idx = {cfg.STATE_NAMES.index(s): cv[s] for s in meas_names}
        pooled = [[] for _ in range(MEAS)]
        for name in train_list:
            sq = niv_sqnorm(name, cv_idx, Q, P0)
            for j in range(MEAS):
                pooled[j] += sq[j]
        niv = np.array([np.mean(v) if v else np.nan for v in pooled])
        done = True
        for j, s in enumerate(meas_names):
            if np.isnan(niv[j]):
                continue
            new = cv[s] * np.sqrt(niv[j])
            new_c = float(np.clip(new, args.cv_min, args.cv_max))
            capped[s] = bool(new > args.cv_max); floored[s] = bool(new < args.cv_min)
            if abs(niv[j] - 1.0) > args.cv_tol:
                done = False
            cv[s] = new_c
        print(f"      CV iter {it}: NIV=[" + ", ".join(f"{v:.2f}" for v in niv) + "]", flush=True)
        if done:
            break
    return cv, {s: float(niv[j]) for j, s in enumerate(meas_names)}, capped, floored


def nsd_asn_metrics(name, mean_traj, std_traj):
    dd = ds_data(name)
    out = {}
    for col, si in enumerate(nsd_state_idx):
        meas = dd["nsd_meas"][:, col]
        m = mean_traj[meas_grid_idx, si]; s = std_traj[meas_grid_idx, si]
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


def mean_nsd_nrmse(train_list, cv_idx, alpha_obs, alpha_nsd):
    Q = build_Q(alpha_obs, alpha_nsd); P0 = P0_from(Q)
    vals = []
    for name in train_list:
        mt, st = enkf_pass(name, cv_idx, Q, P0)
        met = nsd_asn_metrics(name, mt, st)
        vals.append(np.nanmean([met[n]["nrmse"] for n in nsd_names]))
    return float(np.mean(vals))


def asn_nrmse(train_list, cv_idx, alpha_obs, alpha_nsd):
    Q = build_Q(alpha_obs, alpha_nsd); P0 = P0_from(Q)
    vals = []
    for name in train_list:
        mt, st = enkf_pass(name, cv_idx, Q, P0)
        vals.append(nsd_asn_metrics(name, mt, st)["Asn"]["nrmse"])
    return float(np.mean(vals))


def select_alphas(train_list, cv_idx):
    """Mode B: auto-select alpha_nsd (min mean NSD NRMSE) then alpha_obs (min Asn NRMSE)."""
    a_nsd = min(NSD_GRID, key=lambda a: mean_nsd_nrmse(train_list, cv_idx, cfg.PROCESS_NOISE_ALPHA_OBS, a))
    a_obs = min(OBS_GRID, key=lambda a: asn_nrmse(train_list, cv_idx, a, a_nsd))
    return a_obs, a_nsd


# ── Fold construction ─────────────────────────────────────────────────────────
def folds():
    if args.train:
        if args.train not in DATASETS:
            raise SystemExit(f"--train {args.train} must be one of --folds {DATASETS}")
        return [(args.train, [args.train], [x for x in DATASETS if x != args.train])]
    if args.scheme == "rotate":
        return [(d, [d], [x for x in DATASETS if x != d]) for d in DATASETS]
    return [(d, [x for x in DATASETS if x != d], [d]) for d in DATASETS]  # loo


def run_mode(mode):
    out_dir = RESULTS_DIR / mode
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    summ_path = out_dir / "summary.json"
    # Always load an existing summary so single-fold (--train) runs ACCUMULATE across days;
    # --resume additionally skips folds already present (below) instead of recomputing them.
    summary = json.load(open(summ_path)) if summ_path.exists() else {"scheme": args.scheme, "mode": mode, "folds": {}}

    for fid, train_list, val_list in folds():
        if args.resume and fid in summary["folds"]:
            print(f"  [{mode}] fold {fid}: cached, skip."); continue
        print(f"\n  [{mode}] fold {fid}: train={train_list} validate={val_list}")
        cv, niv, capped, floored = calibrate_cv(train_list)
        cv_idx = {cfg.STATE_NAMES.index(s): cv[s] for s in meas_names}
        if mode == "cv":
            a_obs, a_nsd = cfg.PROCESS_NOISE_ALPHA_OBS, cfg.PROCESS_NOISE_ALPHA
        else:
            print("    selecting alphas on training set ...", flush=True)
            a_obs, a_nsd = select_alphas(train_list, cv_idx)
        print(f"    tuned: alpha_obs={a_obs:g} alpha_nsd={a_nsd:g}")

        Q = build_Q(a_obs, a_nsd); P0 = P0_from(Q)
        heldout = {}
        payload = {"fold": fid, "train": train_list, "validate": val_list,
                   "cv": cv, "niv": niv, "capped": capped, "floored": floored,
                   "alpha_obs": a_obs, "alpha_nsd": a_nsd, "T": T_model,
                   "state_names": list(cfg.STATE_NAMES), "val": {}}
        for name in val_list:
            mt, st = enkf_pass(name, cv_idx, Q, P0)
            met = nsd_asn_metrics(name, mt, st)
            heldout[name] = met
            payload["val"][name] = {
                "mean_trajectory": mt, "std_trajectory": st,
                "band_2sigma_lo": np.maximum(mt - 2 * st, 0.0), "band_2sigma_hi": mt + 2 * st,
                "model_trajectory": ds_data(name)["model"], "metrics": met,
            }
            print(f"      held-out {name}: mean NSD NRMSE="
                  f"{np.nanmean([met[n]['nrmse'] for n in nsd_names]):.3f}  "
                  f"Asn NRMSE={met['Asn']['nrmse']:.3f}")
        with open(out_dir / f"fold_{fid}.pkl", "wb") as f:
            pickle.dump(payload, f)
        summary["folds"][fid] = {"train": train_list, "validate": val_list,
                                 "cv": cv, "niv": niv, "capped": capped, "floored": floored,
                                 "alpha_obs": a_obs, "alpha_nsd": a_nsd, "heldout": heldout}
        tmp = summ_path.with_suffix(".json.tmp")
        json.dump(summary, open(tmp, "w"), indent=2); os.replace(tmp, summ_path)  # atomic checkpoint
    return summary


# ── Run the requested mode(s) ────────────────────────────────────────────────
for m in MODES:
    run_mode(m)

# Load EVERY mode summary present on disk (so A and B run on separate days still compare).
all_summ = {}
for m in ("cv", "all"):
    sp = RESULTS_DIR / m / "summary.json"
    if sp.exists():
        all_summ[m] = json.load(open(sp))
present = [m for m in ("cv", "all") if m in all_summ]

# ── Report ─────────────────────────────────────────────────────────────────
def held_agg(summary):
    """mean held-out NSD NRMSE / coverage / Asn NRMSE across all folds' held-out sets."""
    nrmse, cov, asn = [], [], []
    for fd in summary["folds"].values():
        for m in fd["heldout"].values():
            nrmse.append(np.nanmean([m[n]["nrmse"] for n in nsd_names]))
            cov.append(np.nanmean([m[n]["cov"] for n in nsd_names]))
            asn.append(m["Asn"]["nrmse"])
    return (np.nanmean(nrmse), np.nanmean(cov), np.nanmean(asn))

print("\n" + "=" * 78)
for m in present:
    nr, cv_, asn = held_agg(all_summ[m])
    lbl = "A (CVs only)" if m == "cv" else "B (CVs + alpha)"
    print(f"  retune={m:3s} [{lbl}]  held-out mean NSD NRMSE={nr:.3f}  cov={cv_:.0f}%  Asn NRMSE={asn:.3f}")
print("=" * 78)
if len(present) == 2:
    comp = {m: dict(zip(("nsd_nrmse", "nsd_cov", "asn_nrmse"), held_agg(all_summ[m]))) for m in present}
    json.dump(comp, open(RESULTS_DIR / "comparison.json", "w"), indent=2)
    print(f"A-vs-B comparison written: {RESULTS_DIR / 'comparison.json'}")
else:
    print(f"(Run the other mode later to get the A-vs-B comparison; have: {present})")

# ── Plots ────────────────────────────────────────────────────────────────
if not args.no_plots:
    for m in present:
        S = all_summ[m]; fids = list(S["folds"].keys())
        # parameter stability: CV per measured state across folds
        fig, ax = plt.subplots(figsize=(10, 5))
        for s in meas_names:
            ax.plot(fids, [S["folds"][f]["cv"][s] for f in fids], "o-", label=s)
        ax.set_ylabel("calibrated CV"); ax.set_xlabel("fold (training set)")
        ax.set_title(f"CV parameter stability across folds — retune={m}, scheme={args.scheme}")
        ax.legend(ncol=4, fontsize=8); ax.grid(alpha=0.2)
        fig.tight_layout(); fig.savefig(RESULTS_DIR / m / "figures" / "cv_param_stability.png", dpi=150)
        plt.close(fig)
        # held-out NSD NRMSE + coverage per fold
        fig, axs = plt.subplots(1, 2, figsize=(13, 4.5))
        for f in fids:
            for name, met in S["folds"][f]["heldout"].items():
                axs[0].scatter(f, np.nanmean([met[n]["nrmse"] for n in nsd_names]), color="tab:red")
                axs[1].scatter(f, np.nanmean([met[n]["cov"] for n in nsd_names]), color="tab:blue")
        axs[0].set_title("held-out mean NSD NRMSE"); axs[1].set_title("held-out mean 2σ coverage %")
        axs[1].axhline(95, ls=":", color="gray")
        for ax in axs:
            ax.set_xlabel("fold (training set)"); ax.grid(alpha=0.2)
        fig.suptitle(f"Held-out generalization — retune={m}, scheme={args.scheme}", fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(RESULTS_DIR / m / "figures" / "cv_heldout_metrics.png", dpi=150); plt.close(fig)

    if len(present) == 2:
        fig, ax = plt.subplots(figsize=(7, 5))
        labels = ["mean NSD NRMSE", "mean NSD cov %/100", "Asn NRMSE"]
        for m in present:
            nr, cv_, asn = held_agg(all_summ[m])
            ax.plot(labels, [nr, cv_ / 100, asn], "o-", label=f"retune={m}")
        ax.set_title(f"A vs B held-out (scheme={args.scheme})"); ax.legend(); ax.grid(alpha=0.2)
        fig.tight_layout(); fig.savefig(RESULTS_DIR / "figures_compare_A_vs_B.png", dpi=150)
        plt.close(fig)

print(f"\nDone. Results in {RESULTS_DIR}")
