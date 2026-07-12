"""
sweep_alpha_nsd_multiseed.py  —  multi-seed alpha_nsd sweep with seed-averaged calibration
==========================================================================================
The EnKF is stochastic (random measurement-perturbation ensemble + random initial ensemble +
random process-noise draws). A single realization can under/over-state a band by chance, so
selecting alpha_nsd from ONE run is fragile. This script runs the filter N times with distinct
seeds per (fold, alpha), records EVERY run's full trajectory, and reports metrics on the
SEED-AVERAGED posterior — the legitimate stochastic estimate.

For each dataset (fold) it loads that fold's CALIBRATED measured CVs from
results_single_sweep/fold_<X>/cv/cv_final.json and holds alpha_obs = 0.002 (the adopted value), then
sweeps alpha_nsd over the grid (0.05 added) at N seeds each.

Seeds: seed 42 reproduces the existing single-seed results_single_sweep nsd pkls bit-for-bit (sanity
check); the default set is 42..42+N-1. This reuses the EXACT machinery of 04_cross_validate.py
(same matrices, same enkf call) so nothing about the filter changes — only the averaging.

Saved under --out (default results_multirun_nsd/), per fold:
    fold_<X>/pkl/alpha_<a>_seed_<s>.pkl  per-run: mean_traj + std_traj + metrics (float32, resumable)
    fold_<X>/agg/alpha_<a>.pkl           all seeds stacked + seed-averaged mean/std + between-seed
                                         spread + per-seed metrics + metrics-on-average
    fold_<X>/figures/nsd_alpha_<a>.png   paper-style 7-NSD grid on the seed-averaged posterior
    summary.pkl                          seed-averaged calibration for all folds/alphas

Usage (macOS venv):
  # all four folds, 10 seeds, full grid incl 0.05 (long; run per fold / overnight):
  caffeinate -i ./.venv/bin/python scripts/sweep_alpha_nsd_multiseed.py --n-runs 10
  # one fold at a time (recommended), resumable:
  caffeinate -i ./.venv/bin/python scripts/sweep_alpha_nsd_multiseed.py --n-runs 10 --datasets P4
  # just add the new alpha to an existing multi-seed run:
  ./.venv/bin/python scripts/sweep_alpha_nsd_multiseed.py --n-runs 10 --nsd-alphas 0.05
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
p = argparse.ArgumentParser(description="Multi-seed alpha_nsd sweep with seed-averaged calibration")
p.add_argument("--datasets", default="P1,P2,P3,P4")
p.add_argument("--nsd-alphas", default="0.005,0.0075,0.01,0.02,0.03,0.04,0.05")
p.add_argument("--alpha-obs", default=0.002, type=float)
p.add_argument("--n-runs", default=10, type=int, help="number of stochastic runs (distinct seeds)")
p.add_argument("--seed-base", default=42, type=int, help="seeds = seed_base .. seed_base+n_runs-1")
p.add_argument("--seeds", default=None, help="explicit CSV of seeds (overrides n-runs/seed-base)")
p.add_argument("--cv-run", default="results_single_sweep", help="run dir holding fold_*/cv/cv_final.json")
p.add_argument("--out", default="results_multirun_nsd", help="output run dir (writes fold_*/)")
p.add_argument("--ensemble-size", default=cfg.ENSEMBLE_SIZE, type=int)
p.add_argument("--archive-down", default=1, type=int, help="downsample factor for per-seed archive (1=full)")
p.add_argument("--traj-down", default=10, type=int, help="downsample factor for figures")
p.add_argument("--dpi", default=200, type=int)
p.add_argument("--no-plots", action="store_true")
p.add_argument("--resume", action="store_true", default=True)
p.add_argument("--no-resume", dest="resume", action="store_false")
# ── auto-reject + resample (option B) ──────────────────────────────────────────
p.add_argument("--auto-reject", action="store_true",
               help="reject divergent replicates (pool-relative peak-sigma outliers) and "
                    "resample fresh seeds until --target-good clean runs are collected")
p.add_argument("--reject-mult", default=3.0, type=float,
               help="a run is divergent if any unmeasured state's peak posterior sigma exceeds "
                    "this multiple of the across-run median peak sigma (default 3.0)")
p.add_argument("--target-good", default=None, type=int,
               help="number of clean (non-divergent) replicates required (default: --n-runs)")
p.add_argument("--max-seeds", default=40, type=int,
               help="safety cap on how many candidate seeds to try when resampling")
args = p.parse_args()

DATASETS = [d for d in args.datasets.split(",") if d]
NSD_GRID = [float(a) for a in args.nsd_alphas.split(",")]
A_OBS = args.alpha_obs
ENS = args.ensemble_size
if args.seeds:
    SEEDS = [int(s) for s in args.seeds.split(",") if s]
else:
    SEEDS = [args.seed_base + i for i in range(args.n_runs)]
CV_RUN = cfg.PROJECT_ROOT / args.cv_run if not Path(args.cv_run).is_absolute() else Path(args.cv_run)
OUT = cfg.PROJECT_ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out)
ADOWN = max(int(args.archive_down), 1)
FDOWN = max(int(args.traj_down), 1)

# ── Fixed grids / matrices (identical to 04_cross_validate.py) ──────────────────
meas_names = cfg.MEASURED_STATES
MEAS = cfg.MEAS_NUM
n_nsd = 7
nsd_state_idx = list(range(cfg.STATE_NUM - n_nsd, cfg.STATE_NUM))
nsd_names = [cfg.STATE_NAMES[i] for i in nsd_state_idx]
REPORTED = ["UDPGal", "UDPGlc", "UDPGlcNAc"]  # the reliably-measured NSDs used for selection
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
unmeas_idx = obs_idx + nsd_state_idx          # Asn, Glu + 7 NSDs — states used for the divergence gate
TARGET_GOOD = args.target_good if args.target_good is not None else args.n_runs
P0_meas = np.array([cfg.MEASUREMENT_NOISE_VAR.get(s, 0.0) for s in cfg.STATE_NAMES])
T_meas = np.array(cfg.T_MEAS_FIXED)
meas_grid_idx = [min(int(round(t / dt_kf)) + 1, N_kf) for t in T_meas]
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


# Deterministic per-dataset data (seed-independent); mens is generated per-seed below.
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
            x0=x0, set_meas=sm[:, :MEAS], asn_meas=sm[:, asn_col], asn_err=se[:, asn_col],
            set_err=se, model=model,
            nsd_meas=pd.DataFrame(d["NSD_meas"]).apply(pd.to_numeric, errors="coerce").to_numpy(),
            nsd_err=pd.DataFrame(d["NSD_meas_errorbar"]).apply(pd.to_numeric, errors="coerce").to_numpy())
    return _static[name]


def enkf_pass_seeded(name, seed, cv_idx, Q, P0):
    """One stochastic EnKF pass at a given seed. Seed 42 == 04_cross_validate.py baseline."""
    s = ds_static(name)
    np.random.seed(seed)
    mens = generate_measurement_ensembles(select_datasets(name), load_dataset,
                                          MEAS, ENS, var_meas)[name]
    np.random.seed(seed)
    _, std_traj, mean_traj = run_enkf_single_with_ensemble_diagnostics(
        dataset_name=name, load_dataset_fn=load_dataset, build_schedule_fn=build_schedule,
        state_init=s["x0"], volume_results=volume_results,
        set_meas_ens=mens, T_meas=T_meas,
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


# ── paper-style NSD figure (seed-averaged posterior) ────────────────────────────
def _style_ax(ax):
    ax.grid(alpha=0.15); ax.tick_params(axis="both", labelsize=10)
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_fontweight("bold")
    for sp in ax.spines.values():
        sp.set_linewidth(1.5)

LEGEND = [
    Line2D([0], [0], color="red", lw=1.8, label="Mechanistic model"),
    Line2D([0], [0], color="steelblue", lw=2.0, label="EnKF mean (seed-avg)"),
    Patch(facecolor="steelblue", alpha=0.30, label=r"EnKF $\pm 1\sigma$"),
    Patch(facecolor="steelblue", alpha=0.15, label=r"EnKF $\pm 2\sigma$"),
    Line2D([0], [0], color="darkorange", marker="o", lw=0, markersize=5, label="Measurements"),
]

def nsd_figure(name, a, mt, st, model, n_seeds, out):
    t = T_model[::FDOWN]; s = ds_static(name)
    fig, axes = plt.subplots(2, 4, figsize=(5.0 * 4, 3.4 * 2)); axes = axes.flatten()
    for k, (nm, si) in enumerate(zip(nsd_names, nsd_state_idx)):
        ax = axes[k]; m = mt[::FDOWN, si]; sd = st[::FDOWN, si]
        ax.fill_between(t, np.maximum(m - 2 * sd, 0), m + 2 * sd, color="steelblue", alpha=0.15)
        ax.fill_between(t, np.maximum(m - sd, 0), m + sd, color="steelblue", alpha=0.30)
        ax.plot(t, m, color="steelblue", lw=2.0)
        ax.plot(t, model[::FDOWN, si], color="red", lw=1.8)
        v = s["nsd_meas"][:, k]; e = s["nsd_err"][:, k]
        ax.errorbar(T_meas, v, yerr=e, fmt="o", color="darkorange", markersize=4.0,
                    capsize=2, elinewidth=1, alpha=0.9, zorder=5)
        star = " *" if nm in REPORTED else ""
        ax.set_ylabel(AX[si], fontsize=10, fontweight="bold")
        ax.set_xlabel("Time (hours)", fontsize=11, fontweight="bold")
        ax.set_title(nm + star, fontsize=13, fontweight="bold"); _style_ax(ax)
    axes[7].set_visible(False)
    fig.legend(handles=LEGEND, loc="lower center", ncol=5, fontsize=11, frameon=False,
               bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(rf"NSDs — {name}, $\alpha_{{nsd}}$={a:g}, $\alpha_{{obs}}$={A_OBS:g}  "
                 rf"(seed-averaged over {n_seeds} runs;  * = reported)",
                 fontsize=15, fontweight="bold")
    plt.tight_layout(rect=[0, 0.04, 1, 0.99])
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight"); plt.close(fig)


# ── per-seed compute/cache (resume: never re-run a seed already on disk) ─────────
def get_seed(name, a, seed, cv_idx, Q, P0, ms_dir):
    """Return (mean_traj, std_traj, metrics, tag) for one seed, from cache or a fresh run."""
    seed_pkl = ms_dir / "pkl" / f"alpha_{a:g}_seed_{seed}.pkl"
    if args.resume and seed_pkl.exists():
        dd = pickle.load(open(seed_pkl, "rb"))
        return (np.asarray(dd["mean_trajectory"], dtype=np.float64),
                np.asarray(dd["std_trajectory"], dtype=np.float64), dd["metrics"], "cached")
    t0 = time.time()
    mt, st = enkf_pass_seeded(name, seed, cv_idx, Q, P0)
    met = nsd_asn_metrics(name, mt, st)
    save_pkl({"dataset": name, "alpha_nsd": a, "alpha_obs": A_OBS, "seed": seed,
              "cv": {s: cv_idx[cfg.STATE_NAMES.index(s)] for s in meas_names},
              "T": T_model[::ADOWN], "state_names": list(cfg.STATE_NAMES),
              "mean_trajectory": mt[::ADOWN].astype(np.float32),
              "std_trajectory": st[::ADOWN].astype(np.float32),
              "metrics": met}, seed_pkl)
    return mt, st, met, f"{time.time()-t0:.0f}s"


def collect_seeds(name, a, cv_idx, Q, P0, ms_dir):
    """Return (used_seeds, rejected, means, stds, per_seed_met) for one (fold, alpha).

    Default: the fixed SEEDS list. With --auto-reject: draw candidate seeds in order,
    reject pool-relative peak-sigma outliers (divergent runs), and keep drawing fresh
    seeds until TARGET_GOOD clean replicates are collected. Cached seeds are never re-run.
    """
    if not args.auto_reject:
        pool = {s: get_seed(name, a, s, cv_idx, Q, P0, ms_dir) for s in SEEDS}
        for s in SEEDS:
            m = pool[s][2]
            print(f"    alpha={a:g} seed={s}: reported cov="
                  f"{np.nanmean([m[r]['cov'] for r in REPORTED]):.0f}% "
                  f"ss={np.nanmean([m[r]['ss'] for r in REPORTED]):.2f}  [{pool[s][3]}]", flush=True)
        return (list(SEEDS), [], [pool[s][0] for s in SEEDS],
                [pool[s][1] for s in SEEDS], [pool[s][2] for s in SEEDS])

    # --auto-reject: pool-relative divergence gate + resample
    pool = {}                      # seed -> (mt, st, met, tag)
    peak = {}                      # seed -> {state_idx: peak sigma}
    cand = args.seed_base
    cap = args.seed_base + args.max_seeds
    def gate():
        med = {i: float(np.median([peak[s][i] for s in pool])) for i in unmeas_idx}
        good, rej = [], []
        for s in sorted(pool):
            bad = any(peak[s][i] > args.reject_mult * med[i] and med[i] > 0 for i in unmeas_idx)
            (rej if bad else good).append(s)
        return good, rej
    while True:
        while len(pool) < TARGET_GOOD and cand < cap:
            mt, st, met, tag = get_seed(name, a, cand, cv_idx, Q, P0, ms_dir)
            pool[cand] = (mt, st, met, tag)
            peak[cand] = {i: float(st[:, i].max()) for i in unmeas_idx}
            print(f"    alpha={a:g} seed={cand}: reported cov="
                  f"{np.nanmean([met[r]['cov'] for r in REPORTED]):.0f}% "
                  f"ss={np.nanmean([met[r]['ss'] for r in REPORTED]):.2f}  [{tag}]", flush=True)
            cand += 1
        good, rej = gate()
        if len(good) >= TARGET_GOOD or cand >= cap:
            break
        # not enough clean runs — draw one more candidate and re-gate
        mt, st, met, tag = get_seed(name, a, cand, cv_idx, Q, P0, ms_dir)
        pool[cand] = (mt, st, met, tag)
        peak[cand] = {i: float(st[:, i].max()) for i in unmeas_idx}
        print(f"    alpha={a:g} seed={cand}: reported cov="
              f"{np.nanmean([met[r]['cov'] for r in REPORTED]):.0f}% "
              f"ss={np.nanmean([met[r]['ss'] for r in REPORTED]):.2f}  [{tag}]", flush=True)
        cand += 1
    used = good[:TARGET_GOOD]
    if rej:
        print(f"    alpha={a:g}: REJECTED divergent seeds {rej}; using {used}", flush=True)
    if len(used) < TARGET_GOOD:
        print(f"    alpha={a:g}: WARNING only {len(used)}/{TARGET_GOOD} clean seeds within "
              f"--max-seeds={args.max_seeds}", flush=True)
    return (used, rej, [pool[s][0] for s in used],
            [pool[s][1] for s in used], [pool[s][2] for s in used])


# ── run ─────────────────────────────────────────────────────────────────────────
print("=" * 82)
mode = f"AUTO-REJECT (C={args.reject_mult:g}, target={TARGET_GOOD})" if args.auto_reject else f"seeds={SEEDS}"
print(f"MULTI-SEED alpha_nsd sweep  |  {mode}  N_ens={ENS}  alpha_obs={A_OBS:g}")
print(f"grid={NSD_GRID}  datasets={DATASETS}  out={OUT.name}/fold_*/")
print("=" * 82)

selection = {}  # fold -> {alpha: metrics_on_average}
for name in DATASETS:
    cv_json = CV_RUN / f"fold_{name}" / "cv" / "cv_final.json"
    if not cv_json.exists():
        raise SystemExit(f"missing calibrated CVs: {cv_json} — run 04 --stage sweep --train {name} first")
    cv = json.load(open(cv_json))["cv"]
    cv_idx = {cfg.STATE_NAMES.index(s): cv[s] for s in meas_names}
    ms_dir = OUT / f"fold_{name}"
    selection[name] = {}
    print(f"\n[{name}] calibrated CVs loaded; {len(NSD_GRID)} alphas"
          + (f" x >={TARGET_GOOD} clean seeds (auto-reject)" if args.auto_reject
             else f" x {len(SEEDS)} seeds"))
    fold_manifest = {}
    for a in NSD_GRID:
        Q = build_Q(A_OBS, a); P0 = P0_from(Q)
        used_seeds, rejected, means, stds, per_seed_met = collect_seeds(name, a, cv_idx, Q, P0, ms_dir)
        fold_manifest[f"{a:g}"] = {"used": used_seeds, "rejected": rejected}
        M = np.stack(means); S = np.stack(stds)         # [n_good, T, 17]
        avg_mean = M.mean(axis=0); avg_std = S.mean(axis=0)
        between_seed_std = M.std(axis=0)                 # seed sensitivity of the posterior mean
        met_avg = nsd_asn_metrics(name, avg_mean, avg_std)
        # per-seed metric mean/std for the reported states + all
        def stack_metric(key):
            return {st_: dict(mean=float(np.nanmean([m[st_][key] for m in per_seed_met])),
                              std=float(np.nanstd([m[st_][key] for m in per_seed_met])))
                    for st_ in list(nsd_names) + ["Asn"]}
        met_seed = {k: stack_metric(k) for k in ("cov", "ss", "nrmse", "rmse")}
        save_pkl({"dataset": name, "alpha_nsd": a, "alpha_obs": A_OBS,
                  "seeds": used_seeds, "rejected_seeds": rejected,
                  "cv": cv, "T": T_model[::ADOWN], "state_names": list(cfg.STATE_NAMES),
                  "all_mean_trajectories": M[:, ::ADOWN].astype(np.float32),
                  "all_std_trajectories": S[:, ::ADOWN].astype(np.float32),
                  "avg_mean_trajectory": avg_mean, "avg_std_trajectory": avg_std,
                  "between_seed_std": between_seed_std[::ADOWN].astype(np.float32),
                  "band_2sigma_lo": np.maximum(avg_mean - 2 * avg_std, 0.0),
                  "band_2sigma_hi": avg_mean + 2 * avg_std,
                  "model_trajectory": ds_static(name)["model"],
                  "metrics_on_average": met_avg, "metrics_per_seed": per_seed_met,
                  "metrics_seed_summary": met_seed},
                 ms_dir / "agg" / f"alpha_{a:g}.pkl")
        selection[name][a] = met_avg
        if not args.no_plots:
            nsd_figure(name, a, avg_mean, avg_std, ds_static(name)["model"], len(used_seeds),
                       ms_dir / "figures" / f"nsd_alpha_{a:g}.png")
        print(f"  => alpha={a:g} SEED-AVG ({len(used_seeds)} runs) reported: cov="
              f"{np.nanmean([met_avg[s]['cov'] for s in REPORTED]):.1f}% "
              f"ss={np.nanmean([met_avg[s]['ss'] for s in REPORTED]):.2f} "
              f"nrmse={np.nanmean([met_avg[s]['nrmse'] for s in REPORTED]):.2f}", flush=True)
    if args.auto_reject:
        ms_dir.mkdir(parents=True, exist_ok=True)
        json.dump({"reject_mult": args.reject_mult, "target_good": TARGET_GOOD,
                   "per_alpha": fold_manifest}, open(ms_dir / "seed_selection.json", "w"), indent=2)
        print(f"  [{name}] seed-selection manifest -> {ms_dir / 'seed_selection.json'}")

# ── seed-averaged calibration summary (reported NSDs) ───────────────────────────
def summ(metric, label, tgt):
    print(f"\n=== SEED-AVG {label}  [target {tgt}]  (mean over {', '.join(REPORTED)}) ===")
    print("fold  " + "  ".join(f"a={a:<6g}" for a in NSD_GRID))
    for name in DATASETS:
        cells = [f"{np.nanmean([selection[name][a][s][metric] for s in REPORTED]):8.2f}" for a in NSD_GRID]
        print(f"{name:5} " + " ".join(cells))
    means = [np.nanmean([selection[n][a][s][metric] for n in DATASETS for s in REPORTED]) for a in NSD_GRID]
    print("mean  " + " ".join(f"{v:8.2f}" for v in means))

print("\n" + "=" * 82)
print("SEED-AVERAGED calibration (metrics computed on the seed-averaged posterior)")
print("=" * 82)
summ("cov", "2sigma coverage %", "~95")
summ("ss", "spread-skill std/RMSE", "~1.0")
summ("nrmse", "NRMSE", "guard rail")

save_pkl({"alpha_obs": A_OBS, "grid": NSD_GRID, "reported": REPORTED,
          "auto_reject": args.auto_reject, "reject_mult": args.reject_mult,
          "target_good": TARGET_GOOD, "candidate_seed_base": args.seed_base,
          "selection_metrics_on_average": selection},
         OUT / "summary.pkl")
print(f"\nSaved per-seed + aggregate pkls under {OUT}/fold_*/ ; "
      f"summary -> {OUT/'summary.pkl'}")
print("Done.")
