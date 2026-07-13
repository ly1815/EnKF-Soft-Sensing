"""
validate_multiseed.py  —  multi-seed held-out cross-validation (seed-averaged, divergent-rejected)
==================================================================================================
The validation analogue of sweep_alpha_nsd_multiseed.py. For each cross-validation fold
(training set P_k), the filter uses THAT fold's calibrated measured CVs
(results_single_sweep/fold_<k>/cv/cv_final.json) and ITS picked alphas
(results_single_sweep/picks.json), and is applied to the THREE held-out batches it never saw
(rotate scheme). Each held-out run is repeated over N seeds; every run's mean+std trajectory is
archived; divergent replicates are rejected (pool-relative peak-sigma outlier rule, identical to
the tuning sweep) and resampled to N clean runs; calibration is reported on the seed-averaged
held-out posterior.

Nothing is tuned here — CVs and alphas are fixed inputs. This is the honest generalisation test:
no held-out batch influences the filter applied to it (R stays a pooled instrument constant).

Output under --out (default results_multirun_validation/), per training fold:
    fold_<k>/pkl/heldout_<name>_seed_<s>.pkl   per-run: mean_traj + std_traj + metrics (float32)
    fold_<k>/agg/heldout_<name>.pkl            all seeds stacked + seed-averaged mean/std +
                                               between-seed spread + per-seed metrics + rejected
    fold_<k>/figures/heldout_<name>.png        all-17-state grid on the seed-averaged posterior
    fold_<k>/seed_selection.json               used/rejected seeds per held-out set
    summary.json                               cross-fold seed-averaged held-out metrics

Usage (macOS venv):
  # all four training folds x their 3 held-out sets, 10 seeds each, reject divergent (long):
  caffeinate -i ./.venv/bin/python scripts/validate_multiseed.py
  # one training fold at a time (recommended), resumable:
  caffeinate -i ./.venv/bin/python scripts/validate_multiseed.py --folds P4
"""

import argparse
import json
import os
import pickle
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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
from nsd_enkf.model import compute_volume_results, simulate_dataset
from nsd_enkf.analysis import generate_measurement_ensembles
from nsd_enkf.enkf import run_enkf_single_with_ensemble_diagnostics

# ── CLI ────────────────────────────────────────────────────────────────────────
p = argparse.ArgumentParser(description="Multi-seed held-out cross-validation (seed-averaged, divergent-rejected)")
p.add_argument("--folds", default="P1,P2,P3,P4", help="training folds to run (subset of --universe)")
p.add_argument("--universe", default="P1,P2,P3,P4", help="full dataset set; held-out = universe minus training")
p.add_argument("--scheme", default="rotate", choices=["rotate", "loo"])
p.add_argument("--picks", default="results_single_sweep/picks.json")
p.add_argument("--cv-run", default="results_single_sweep", help="run dir holding fold_*/cv/cv_final.json")
p.add_argument("--out", default="results_multirun_validation")
p.add_argument("--ensemble-size", default=cfg.ENSEMBLE_SIZE, type=int)
p.add_argument("--n-runs", default=10, type=int)
p.add_argument("--seed-base", default=42, type=int)
p.add_argument("--no-reject", dest="auto_reject", action="store_false", default=True,
               help="disable divergent-replicate rejection (default: on)")
p.add_argument("--reject-mult", default=3.0, type=float,
               help="reject a run if any unmeasured state's peak sigma exceeds this multiple of "
                    "the across-run median peak (default 3.0)")
p.add_argument("--target-good", default=None, type=int, help="clean replicates required (default --n-runs)")
p.add_argument("--max-seeds", default=40, type=int)
p.add_argument("--archive-down", default=1, type=int)
p.add_argument("--traj-down", default=10, type=int)
p.add_argument("--dpi", default=200, type=int)
p.add_argument("--no-plots", action="store_true")
p.add_argument("--resume", action="store_true", default=True)
p.add_argument("--no-resume", dest="resume", action="store_false")
args = p.parse_args()

UNIVERSE = [d for d in args.universe.split(",") if d]
RUN_FOLDS = [d for d in args.folds.split(",") if d]
ENS = args.ensemble_size
TARGET_GOOD = args.target_good if args.target_good is not None else args.n_runs
CV_RUN = cfg.PROJECT_ROOT / args.cv_run if not Path(args.cv_run).is_absolute() else Path(args.cv_run)
OUT = cfg.PROJECT_ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out)
PICKS = cfg.PROJECT_ROOT / args.picks if not Path(args.picks).is_absolute() else Path(args.picks)
ADOWN = max(int(args.archive_down), 1)
FDOWN = max(int(args.traj_down), 1)

# ── Fixed grids / matrices (identical to 04_cross_validate.py) ──────────────────
meas_names = cfg.MEASURED_STATES
MEAS = cfg.MEAS_NUM
n_nsd = 7
nsd_state_idx = list(range(cfg.STATE_NUM - n_nsd, cfg.STATE_NUM))
nsd_names = [cfg.STATE_NAMES[i] for i in nsd_state_idx]
REPORTED = ["UDPGal", "UDPGlc", "UDPGlcNAc"]
ASN = cfg.STATE_NAMES.index("Asn")
AX = cfg.AXIS_NAMES

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
unmeas_idx = obs_idx + nsd_state_idx           # divergence gate operates on these
P0_meas = np.array([cfg.MEASUREMENT_NOISE_VAR.get(s, 0.0) for s in cfg.STATE_NAMES])
T_meas = np.array(cfg.T_MEAS_FIXED)
meas_grid_idx = [min(int(round(t / dt_kf)) + 1, N_kf) for t in T_meas]
volume_results = compute_volume_results(select_datasets(*UNIVERSE), cfg.INITIAL_VOLUMES,
                                        build_schedule, step_len)


def build_Q(alpha_obs, alpha_nsd):
    a = np.full(cfg.STATE_NUM, float(alpha_nsd))
    for i in obs_idx:
        a[i] = float(alpha_obs)
    return np.diag((a * scale_vec) ** 2)


def P0_from(Q):
    d = np.diag(Q).copy(); d[:MEAS] = P0_meas[:MEAS]
    return np.diag(d)


_static = {}
def ds_static(name):
    if name not in _static:
        d = load_dataset(name)
        _, x0 = get_initial_condition(d["met_df"], d["nsd_df"])
        sm = d["set_meas"].astype(float); se = d["set_meas_errorbar"].astype(float)
        asn_col = sm.shape[1] - 1
        Fin, Fout, Gf, Uf = build_schedule(name)
        model = np.vstack([x0, simulate_dataset(x0, Fin, Fout, Gf, Uf,
                                                volume_results[name][1:], time_grid, step_len,
                                                name=name)])
        _static[name] = dict(
            x0=x0, set_meas=sm[:, :MEAS], set_err=se, asn_meas=sm[:, asn_col], asn_err=se[:, asn_col],
            model=model,
            nsd_meas=pd.DataFrame(d["NSD_meas"]).apply(pd.to_numeric, errors="coerce").to_numpy(),
            nsd_err=pd.DataFrame(d["NSD_meas_errorbar"]).apply(pd.to_numeric, errors="coerce").to_numpy())
    return _static[name]


def enkf_pass_seeded(name, seed, cv_idx, Q, P0):
    s = ds_static(name)
    np.random.seed(seed)
    mens = generate_measurement_ensembles(select_datasets(name), load_dataset, MEAS, ENS, var_meas)[name]
    np.random.seed(seed)
    _, std_traj, mean_traj = run_enkf_single_with_ensemble_diagnostics(
        dataset_name=name, load_dataset_fn=load_dataset, build_schedule_fn=build_schedule,
        state_init=s["x0"], volume_results=volume_results, set_meas_ens=mens, T_meas=T_meas,
        state_num=cfg.STATE_NUM, meas_num=MEAS, ensemble_size=ENS,
        Q=Q, R=R, H=H, dt_kf=dt_kf, N_kf=N_kf, P0=P0,
        process_noise_cv=dict(cv_idx), no_update_indices=set(no_update_indices),
        clip_indices=set(clip_indices))
    return mean_traj, std_traj


def nsd_asn_metrics(name, mean_traj, std_traj):
    s = ds_static(name); out = {}
    for col, si in enumerate(nsd_state_idx):
        meas = s["nsd_meas"][:, col]; m = mean_traj[meas_grid_idx, si]; sd = std_traj[meas_grid_idx, si]
        valid = ~np.isnan(meas) & (sd > 0)
        if valid.sum() == 0:
            out[nsd_names[col]] = dict(rmse=np.nan, nrmse=np.nan, cov=np.nan, ss=np.nan); continue
        err = meas[valid] - m[valid]; rmse = float(np.sqrt(np.mean(err ** 2)))
        norm = float(np.mean(np.abs(meas[valid]))) or 1.0
        out[nsd_names[col]] = dict(rmse=rmse, nrmse=rmse / norm,
                                   cov=100.0 * float(np.mean(np.abs(err) <= 2 * sd[valid])),
                                   ss=(float(np.mean(sd[valid]) / rmse) if rmse > 0 else np.nan))
    meas = s["asn_meas"]; m = mean_traj[meas_grid_idx, ASN]; sd = std_traj[meas_grid_idx, ASN]
    valid = ~np.isnan(meas) & (sd > 0); err = meas[valid] - m[valid]
    rmse = float(np.sqrt(np.mean(err ** 2))); norm = float(np.mean(np.abs(meas[valid]))) or 1.0
    out["Asn"] = dict(rmse=rmse, nrmse=rmse / norm,
                      cov=100.0 * float(np.mean(np.abs(err) <= 2 * sd[valid])),
                      ss=(float(np.mean(sd[valid]) / rmse) if rmse > 0 else np.nan))
    return out


def save_pkl(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(obj, f)
    os.replace(tmp, path)


def meas_by_state(name):
    s = ds_static(name)
    mbs = {i: (s["set_meas"][:, i], s["set_err"][:, i]) for i in range(MEAS)}
    mbs[ASN] = (s["asn_meas"], s["asn_err"])
    for j in range(n_nsd):
        mbs[nsd_state_idx[j]] = (s["nsd_meas"][:, j], s["nsd_err"][:, j])
    return mbs


# ── all-state grid figure on the seed-averaged held-out posterior ───────────────
LEGEND = [Line2D([0], [0], color="red", lw=1.8, label="Mechanistic model"),
          Line2D([0], [0], color="steelblue", lw=2.0, label="EnKF mean (seed-avg, held-out)"),
          Patch(facecolor="steelblue", alpha=0.30, label=r"EnKF $\pm1\sigma$"),
          Patch(facecolor="steelblue", alpha=0.15, label=r"EnKF $\pm2\sigma$"),
          Line2D([0], [0], color="darkorange", marker="o", lw=0, markersize=6, label="Measurements")]


def allstate_grid(name, mt, st, model, n_seeds, title, out):
    t = T_model[::FDOWN]; mbs = meas_by_state(name)
    fig, axes = plt.subplots(5, 4, figsize=(20, 15)); axes = axes.flatten()
    for si in range(cfg.STATE_NUM):
        ax = axes[si]; m = mt[::FDOWN, si]; sd = st[::FDOWN, si]
        ax.fill_between(t, np.maximum(m - 2 * sd, 0), m + 2 * sd, color="steelblue", alpha=0.15)
        ax.fill_between(t, np.maximum(m - sd, 0), m + sd, color="steelblue", alpha=0.30)
        ax.plot(t, m, color="steelblue", lw=2.0)
        ax.plot(t, model[::FDOWN, si], color="red", lw=1.6)
        if si in mbs:
            v, e = mbs[si]
            ax.errorbar(T_meas, v, yerr=e, fmt="o", color="darkorange", markersize=4,
                        capsize=2, elinewidth=1, alpha=0.9, zorder=5)
        tag = "" if si in mbs else "  (no meas)"
        star = " *" if cfg.STATE_NAMES[si] in REPORTED else ""
        ax.set_title(cfg.STATE_NAMES[si] + star + tag, fontsize=11, fontweight="bold")
        ax.set_xlabel("Time (hours)", fontsize=9); ax.grid(alpha=0.15)
    for k in range(cfg.STATE_NUM, len(axes)):
        axes[k].set_visible(False)
    fig.legend(handles=LEGEND, loc="lower center", ncol=5, fontsize=12, frameon=False, bbox_to_anchor=(0.5, -0.005))
    fig.suptitle(title, fontsize=15, fontweight="bold", y=1.005)
    plt.tight_layout(rect=[0, 0.03, 1, 1]); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight"); plt.close(fig)


# ── per-seed compute/cache + divergence gate + resample ─────────────────────────
def get_run(name, seed, cv_idx, Q, P0, fdir):
    seed_pkl = fdir / "pkl" / f"heldout_{name}_seed_{seed}.pkl"
    if args.resume and seed_pkl.exists():
        dd = pickle.load(open(seed_pkl, "rb"))
        return (np.asarray(dd["mean_trajectory"], dtype=np.float64),
                np.asarray(dd["std_trajectory"], dtype=np.float64), dd["metrics"], "cached")
    t0 = time.time()
    mt, st = enkf_pass_seeded(name, seed, cv_idx, Q, P0)
    met = nsd_asn_metrics(name, mt, st)
    save_pkl({"held_out": name, "seed": seed, "T": T_model[::ADOWN],
              "state_names": list(cfg.STATE_NAMES),
              "mean_trajectory": mt[::ADOWN].astype(np.float32),
              "std_trajectory": st[::ADOWN].astype(np.float32),
              "metrics": met}, seed_pkl)
    return mt, st, met, f"{time.time()-t0:.0f}s"


def collect_seeds(name, cv_idx, Q, P0, fdir):
    """Return (used, rejected, means, stds, per_seed_met) for one held-out set."""
    pool, peak = {}, {}
    cand, cap = args.seed_base, args.seed_base + args.max_seeds
    def log(seed, met, tag):
        print(f"    held-out {name} seed={seed}: reported cov="
              f"{np.nanmean([met[r]['cov'] for r in REPORTED]):.0f}% "
              f"ss={np.nanmean([met[r]['ss'] for r in REPORTED]):.2f}  [{tag}]", flush=True)
    def add(seed):
        mt, st, met, tag = get_run(name, seed, cv_idx, Q, P0, fdir)
        pool[seed] = (mt, st, met, tag)
        peak[seed] = {i: float(st[:, i].max()) for i in unmeas_idx}
        log(seed, met, tag)
    if not args.auto_reject:
        for s in range(args.seed_base, args.seed_base + args.n_runs):
            add(s)
        used = sorted(pool)
        return (used, [], [pool[s][0] for s in used], [pool[s][1] for s in used],
                [pool[s][2] for s in used])
    def gate():
        med = {i: float(np.median([peak[s][i] for s in pool])) for i in unmeas_idx}
        good, rej = [], []
        for s in sorted(pool):
            bad = any(peak[s][i] > args.reject_mult * med[i] and med[i] > 0 for i in unmeas_idx)
            (rej if bad else good).append(s)
        return good, rej
    while True:
        while len(pool) < TARGET_GOOD and cand < cap:
            add(cand); cand += 1
        good, rej = gate()
        if len(good) >= TARGET_GOOD or cand >= cap:
            break
        add(cand); cand += 1
    used = good[:TARGET_GOOD]
    if rej:
        print(f"    held-out {name}: REJECTED divergent seeds {rej}; using {used}", flush=True)
    if len(used) < TARGET_GOOD:
        print(f"    held-out {name}: WARNING only {len(used)}/{TARGET_GOOD} clean seeds "
              f"within --max-seeds={args.max_seeds}", flush=True)
    return (used, rej, [pool[s][0] for s in used], [pool[s][1] for s in used],
            [pool[s][2] for s in used])


def folds():
    if args.scheme == "rotate":
        return [(d, [d], [x for x in UNIVERSE if x != d]) for d in RUN_FOLDS]
    return [(d, [x for x in UNIVERSE if x != d], [d]) for d in RUN_FOLDS]  # loo


# ── run ─────────────────────────────────────────────────────────────────────────
print("=" * 82)
print(f"MULTI-SEED VALIDATION  |  scheme={args.scheme}  N_ens={ENS}  "
      f"{'AUTO-REJECT C=%g' % args.reject_mult if args.auto_reject else 'no-reject'}  "
      f"target={TARGET_GOOD}  out={OUT.name}/")
print("=" * 82)

picks = json.load(open(PICKS))
summary = {"scheme": args.scheme, "seed_base": args.seed_base, "target_good": TARGET_GOOD,
           "reject_mult": args.reject_mult if args.auto_reject else None, "folds": {}}
OUT.mkdir(parents=True, exist_ok=True)

for fid, train_list, val_list in folds():
    cvj = json.load(open(CV_RUN / f"fold_{fid}" / "cv" / "cv_final.json"))
    cv = cvj["cv"]; cv_idx = {cfg.STATE_NAMES.index(s): cv[s] for s in meas_names}
    a_obs = float(picks[fid]["alpha_obs"]); a_nsd = float(picks[fid]["alpha_nsd"])
    Q = build_Q(a_obs, a_nsd); P0 = P0_from(Q)
    fdir = OUT / f"fold_{fid}"
    print(f"\n[{fid}] train={train_list}  alpha_obs={a_obs:g} alpha_nsd={a_nsd:g}  held-out={val_list}")
    manifest = {}; fold_sum = {"train": train_list, "alpha_obs": a_obs, "alpha_nsd": a_nsd,
                               "cv": cv, "heldout": {}}
    for name in val_list:
        used, rej, means, stds, per_seed_met = collect_seeds(name, cv_idx, Q, P0, fdir)
        manifest[name] = {"used": used, "rejected": rej}
        M = np.stack(means); S = np.stack(stds)
        avg_mean = M.mean(axis=0); avg_std = S.mean(axis=0)
        between_seed_std = M.std(axis=0)
        met_avg = nsd_asn_metrics(name, avg_mean, avg_std)
        def stack_metric(key):
            return {st_: dict(mean=float(np.nanmean([m[st_][key] for m in per_seed_met])),
                              std=float(np.nanstd([m[st_][key] for m in per_seed_met])))
                    for st_ in list(nsd_names) + ["Asn"]}
        met_seed = {k: stack_metric(k) for k in ("cov", "ss", "nrmse", "rmse")}
        save_pkl({"fold": fid, "train": train_list, "held_out": name,
                  "alpha_obs": a_obs, "alpha_nsd": a_nsd, "cv": cv,
                  "seeds": used, "rejected_seeds": rej, "T": T_model[::ADOWN],
                  "state_names": list(cfg.STATE_NAMES),
                  "all_mean_trajectories": M[:, ::ADOWN].astype(np.float32),
                  "all_std_trajectories": S[:, ::ADOWN].astype(np.float32),
                  "avg_mean_trajectory": avg_mean, "avg_std_trajectory": avg_std,
                  "between_seed_std": between_seed_std[::ADOWN].astype(np.float32),
                  "band_2sigma_lo": np.maximum(avg_mean - 2 * avg_std, 0.0),
                  "band_2sigma_hi": avg_mean + 2 * avg_std,
                  "model_trajectory": ds_static(name)["model"],
                  "metrics_on_average": met_avg, "metrics_per_seed": per_seed_met,
                  "metrics_seed_summary": met_seed},
                 fdir / "agg" / f"heldout_{name}.pkl")
        if not args.no_plots:
            nsd_nrmse = np.nanmean([met_avg[n]["nrmse"] for n in REPORTED])
            allstate_grid(name, avg_mean, avg_std, ds_static(name)["model"], len(used),
                          f"Held-out validation — trained {fid} (alpha_obs={a_obs:g}, alpha_nsd={a_nsd:g}), "
                          f"held-out {name}  |  {len(used)} clean seeds"
                          + (f", rejected {rej}" if rej else "")
                          + f"  |  reported-NSD NRMSE={nsd_nrmse:.2f}",
                          fdir / "figures" / f"heldout_{name}.png")
        fold_sum["heldout"][name] = met_avg
        print(f"  => held-out {name} ({len(used)} runs): reported-NSD cov="
              f"{np.nanmean([met_avg[n]['cov'] for n in REPORTED]):.1f}% "
              f"ss={np.nanmean([met_avg[n]['ss'] for n in REPORTED]):.2f} "
              f"nrmse={np.nanmean([met_avg[n]['nrmse'] for n in REPORTED]):.2f} "
              f"| Asn NRMSE={met_avg['Asn']['nrmse']:.2f}", flush=True)
    fdir.mkdir(parents=True, exist_ok=True)
    json.dump({"reject_mult": args.reject_mult if args.auto_reject else None,
               "target_good": TARGET_GOOD, "per_heldout": manifest},
              open(fdir / "seed_selection.json", "w"), indent=2)
    summary["folds"][fid] = fold_sum
    json.dump(summary, open(OUT / "summary.json", "w"), indent=2)

print(f"\n[done] summary -> {OUT / 'summary.json'}")
