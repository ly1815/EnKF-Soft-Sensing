"""
ensemble_size_sensitivity.py
============================
Sweep ensemble sizes on P4 with multiple independent runs per size.

Three groups evaluated:
  1. Measured metabolites (8 states): normalised RMSE, NIV, 2σ coverage
  2. Validated NSDs (UDP-Gal, UDP-Glc, UDP-GlcNAc): normalised RMSE
  3. Asparagine (unmeasured extracellular): normalised RMSE

All RMSE normalised by median absolute measurement value per state.

Usage:
    poetry run python scripts/ensemble_size_sensitivity.py
    poetry run python scripts/ensemble_size_sensitivity.py --run tuned_v6 --n-runs 10
"""

import argparse
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy.linalg import solve, LinAlgWarning

warnings.filterwarnings("ignore", category=LinAlgWarning)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import gc
import nsd_enkf.config as cfg
from nsd_enkf.data_loader import load_dataset, build_schedule
from nsd_enkf.model import model_step
from nsd_enkf.io_utils import load_pkl, set_dirs, save_pkl
from nsd_enkf.enkf import EnsembleKalmanFilter

# ── CLI ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--run", default=cfg.RUN_NAME)
parser.add_argument("--n-runs", default=10, type=int, help="Independent runs per ensemble size")
parser.add_argument("--sizes", default=None, help="Comma-separated ensemble sizes (default: all)")
parser.add_argument("--seed-offset", default=42, type=int, help="Base seed (run i uses seed offset+i)")
args = parser.parse_args()

RUN_NAME = args.run
N_RUNS = args.n_runs
RESULTS_DIR = cfg.PROJECT_ROOT / "results" / RUN_NAME
PKL_DIR = RESULTS_DIR / "pkl"
if not PKL_DIR.exists():
    alt = RESULTS_DIR / "01_run_enkf" / "pkl"
    if alt.exists():
        PKL_DIR = alt

OUT_DIR = RESULTS_DIR / "ensemble_sensitivity"
OUT_DIR.mkdir(parents=True, exist_ok=True)
set_dirs(PKL_DIR, OUT_DIR)

DATASET = "P4"
if args.sizes:
    ENSEMBLE_SIZES = [int(x) for x in args.sizes.split(",")]
else:
    ENSEMBLE_SIZES = [25, 50, 75, 100, 150, 200, 300]
SEED_OFFSET = args.seed_offset

print("=" * 60)
print(f"Ensemble Size Sensitivity  [{RUN_NAME}]")
print(f"  Dataset: {DATASET}")
print(f"  Sizes:   {ENSEMBLE_SIZES}")
print(f"  Runs per size: {N_RUNS}")
print("=" * 60)

# ── Load shared data ─────────────────────────────────────────────────────────
volume_results = load_pkl("volume_results.pkl", subdir=PKL_DIR)
state_init_by_dataset = load_pkl("state_init_by_dataset.pkl", subdir=PKL_DIR)
T_meas_by_dataset = load_pkl("T_meas_by_dataset.pkl", subdir=PKL_DIR)

var_model = np.array(list(cfg.PROCESS_NOISE_VAR.values()))
var_meas = np.array(list(cfg.MEASUREMENT_NOISE_VAR.values()))
Q = cfg.KQ * np.diag(var_model)
R = np.diag(var_meas[: cfg.MEAS_NUM])
H = np.hstack((np.eye(cfg.MEAS_NUM), np.zeros((cfg.MEAS_NUM, cfg.STATE_NUM - cfg.MEAS_NUM))))

process_noise_cv = {}
if hasattr(cfg, "PROCESS_NOISE_CV"):
    for s, cv in cfg.PROCESS_NOISE_CV.items():
        process_noise_cv[cfg.STATE_NAMES.index(s)] = cv

no_update_indices = set()
if hasattr(cfg, "NO_UPDATE_STATES"):
    for s in cfg.NO_UPDATE_STATES:
        no_update_indices.add(cfg.STATE_NAMES.index(s))

P0_diag = var_model.copy()
if hasattr(cfg, "INITIAL_COV_OVERRIDE"):
    for i, s in enumerate(cfg.STATE_NAMES):
        if s in cfg.INITIAL_COV_OVERRIDE:
            P0_diag[i] = cfg.INITIAL_COV_OVERRIDE[s]
P0 = np.diag(P0_diag)

# ── Dataset setup ────────────────────────────────────────────────────────────
data = load_dataset(DATASET)
set_meas = data["set_meas"][:, : cfg.MEAS_NUM].astype(float)
T_meas = T_meas_by_dataset[DATASET]
state_init = state_init_by_dataset[DATASET].copy()
Fin, Fout, Gal_feed, Urd_feed = build_schedule(DATASET)
V_traj = volume_results[DATASET][1:]

dt_kf = cfg.DT
N_kf = int(cfg.T_END / dt_kf)
T_kf = np.linspace(0, cfg.T_END, N_kf + 1)
time_steps_A = [round(i * dt_kf, 2) for i in range(N_kf)]
time_steps_B = [round(t, 2) for t in T_meas.tolist()]
meas_time_to_index = {t: i for i, t in enumerate(time_steps_B)}

# NSD validation data
nsd_vals = pd.DataFrame(data["NSD_meas"]).apply(pd.to_numeric, errors="coerce").to_numpy()

# Full measurement array for Asn (column 8)
asn_full = data["set_meas"][:, 8].astype(float) if data["set_meas"].shape[1] > 8 else None

ASN_IDX = cfg.STATE_NAMES.index("Asn")
NSD_VALIDATED = {
    "UDPGal": (cfg.STATE_NAMES.index("UDPGal"), 0),
    "UDPGlc": (cfg.STATE_NAMES.index("UDPGlc"), 2),
    "UDPGlcNAc": (cfg.STATE_NAMES.index("UDPGlcNAc"), 3),
}

# ── Normalisation scales ─────────────────────────────────────────────────────
meas_scales = np.array([
    max(np.median(np.abs(set_meas[~np.isnan(set_meas[:, j]), j])), 1e-12)
    for j in range(cfg.MEAS_NUM)
])
asn_scale = max(np.median(np.abs(asn_full[~np.isnan(asn_full)])), 1e-12) if asn_full is not None else 1.0
nsd_scales = {
    name: max(np.median(np.abs(nsd_vals[~np.isnan(nsd_vals[:, col]), col])), 1e-12)
    for name, (_, col) in NSD_VALIDATED.items()
}


def run_single(ensemble_size, seed):
    """Run one EnKF pass and return diagnostics."""
    np.random.seed(seed)

    meas_std = np.sqrt(np.diag(R))
    N_meas_time = set_meas.shape[0]
    set_meas_ens = np.zeros((N_meas_time, ensemble_size, cfg.MEAS_NUM))
    for i in range(N_meas_time):
        noise = np.random.multivariate_normal(np.zeros(cfg.MEAS_NUM), R, size=ensemble_size)
        for j in range(cfg.MEAS_NUM):
            noise[:, j] = np.clip(noise[:, j], -3 * meas_std[j], 3 * meas_std[j])
        set_meas_ens[i] = np.clip(set_meas[i] + noise, a_min=1e-12, a_max=None)

    enkf = EnsembleKalmanFilter(cfg.STATE_NUM, cfg.MEAS_NUM)
    enkf.x = state_init.copy()
    enkf.Q = Q.copy(); enkf.R = R.copy(); enkf.H = H.copy()
    enkf.fx = model_step; enkf.dt = dt_kf
    enkf.process_noise_cv = dict(process_noise_cv)
    enkf.no_update_indices = set(no_update_indices)
    enkf.create_ensemble(ensemble_size, P0)

    mean_traj = [enkf.x.copy()]
    std_traj = [np.std(enkf.X, axis=0)]
    innovations = []
    innov_covs = []
    meas_at_updates = []

    t0 = time.time()
    for idx_A, step_A in enumerate(time_steps_A):
        enkf.predict({
            "Fin": Fin[idx_A], "Fout": Fout[idx_A], "V": V_traj[idx_A],
            "Gal_feed": Gal_feed[idx_A], "Urd_feed": Urd_feed[idx_A],
        })
        if step_A in meas_time_to_index:
            idx_B = meas_time_to_index[step_A]
            Z = np.array([H @ x for x in enkf.X])
            z_pred_mean = np.mean(Z, axis=0)
            d = set_meas[idx_B] - z_pred_mean
            E_z = Z - z_pred_mean
            P_zz = (E_z.T @ E_z) / (ensemble_size - 1) + R
            innovations.append(d)
            innov_covs.append(P_zz)
            meas_at_updates.append(set_meas[idx_B])
            enkf.update(set_meas_ens[idx_B])
        mean_traj.append(enkf.x.copy())
        std_traj.append(np.std(enkf.X, axis=0))

    wall_time = time.time() - t0
    mean_traj = np.array(mean_traj)
    std_traj = np.array(std_traj)
    innovations = np.array(innovations)
    innov_covs = np.array(innov_covs)
    n_updates = len(innovations)
    update_traj_indices = [time_steps_A.index(sA) + 1 for sA in time_steps_A if sA in meas_time_to_index]

    # Metabolite diagnostics
    meas_nrmse = np.zeros(cfg.MEAS_NUM)
    meas_niv = np.zeros(cfg.MEAS_NUM)
    meas_cov = np.zeros(cfg.MEAS_NUM)
    for j in range(cfg.MEAS_NUM):
        pred = np.interp(T_meas, T_kf, mean_traj[:, j])
        mask = ~np.isnan(set_meas[:, j])
        meas_nrmse[j] = np.sqrt(np.mean((set_meas[mask, j] - pred[mask]) ** 2)) / meas_scales[j]
        meas_niv[j] = np.var([innovations[k, j] / np.sqrt(innov_covs[k, j, j]) for k in range(n_updates)])
        in_band = sum(
            1 for k, tidx in enumerate(update_traj_indices)
            if not np.isnan(meas_at_updates[k][j]) and std_traj[tidx, j] > 0
            and abs(meas_at_updates[k][j] - mean_traj[tidx, j]) <= 2 * std_traj[tidx, j]
        )
        meas_cov[j] = in_band / n_updates * 100

    # NSD diagnostics
    nsd_nrmse = {}
    for name, (state_idx, nsd_col) in NSD_VALIDATED.items():
        measured = nsd_vals[:, nsd_col]
        pred = np.interp(T_meas, T_kf, mean_traj[:, state_idx])
        mask = ~np.isnan(measured)
        nsd_nrmse[name] = np.sqrt(np.mean((measured[mask] - pred[mask]) ** 2)) / nsd_scales[name] if mask.any() else np.nan

    # Asparagine
    if asn_full is not None:
        pred_asn = np.interp(T_meas, T_kf, mean_traj[:, ASN_IDX])
        mask_asn = ~np.isnan(asn_full)
        asn_nrmse = np.sqrt(np.mean((asn_full[mask_asn] - pred_asn[mask_asn]) ** 2)) / asn_scale
    else:
        asn_nrmse = np.nan

    nis_values = [d @ np.linalg.solve(S, d) / cfg.MEAS_NUM for d, S in zip(innovations, innov_covs)]

    return {
        "wall_time_s": wall_time,
        "meas_nrmse": meas_nrmse, "meas_nrmse_mean": np.mean(meas_nrmse),
        "meas_niv": meas_niv, "meas_niv_mean": np.mean(meas_niv),
        "meas_cov": meas_cov, "meas_cov_mean": np.mean(meas_cov),
        "nis_mean": np.mean(nis_values),
        "nsd_nrmse": dict(nsd_nrmse), "nsd_nrmse_mean": np.mean(list(nsd_nrmse.values())),
        "asn_nrmse": asn_nrmse,
    }


# ── Run sweep (save per-size to disk to avoid OOM) ───────────────────────────
all_results = {}

for N in ENSEMBLE_SIZES:
    print(f"\n  N = {N}:", flush=True)
    runs = []
    for run_i in range(N_RUNS):
        seed = SEED_OFFSET + run_i
        res = run_single(N, seed)
        runs.append(res)
        print(f"    run {run_i+1}/{N_RUNS}: {res['wall_time_s']:.0f}s", flush=True)
    all_results[N] = runs
    # Save per-seed if single run, or bundled if multiple
    if N_RUNS == 1:
        save_pkl(runs[0], f"ensemble_N{N}_seed{SEED_OFFSET}.pkl", subdir=OUT_DIR)
    else:
        save_pkl(runs, f"ensemble_N{N}_runs.pkl", subdir=OUT_DIR)
    gc.collect()

save_pkl(all_results, "ensemble_sensitivity_all_runs.pkl", subdir=OUT_DIR)

# ── Aggregate: mean and std across runs ──────────────────────────────────────
summary = []
for N in ENSEMBLE_SIZES:
    runs = all_results[N]
    s = {
        "N": N,
        "wall_time_mean": np.mean([r["wall_time_s"] for r in runs]),
        "wall_time_std": np.std([r["wall_time_s"] for r in runs]),
        "nis_mean": np.mean([r["nis_mean"] for r in runs]),
        "nis_std": np.std([r["nis_mean"] for r in runs]),
        "met_nrmse_mean": np.mean([r["meas_nrmse_mean"] for r in runs]),
        "met_nrmse_std": np.std([r["meas_nrmse_mean"] for r in runs]),
        "met_cov_mean": np.mean([r["meas_cov_mean"] for r in runs]),
        "met_cov_std": np.std([r["meas_cov_mean"] for r in runs]),
        "nsd_nrmse_mean": np.mean([r["nsd_nrmse_mean"] for r in runs]),
        "nsd_nrmse_std": np.std([r["nsd_nrmse_mean"] for r in runs]),
        "asn_nrmse_mean": np.mean([r["asn_nrmse"] for r in runs]),
        "asn_nrmse_std": np.std([r["asn_nrmse"] for r in runs]),
    }
    # Per-NSD breakdown
    for name in NSD_VALIDATED:
        vals = [r["nsd_nrmse"][name] for r in runs]
        s[f"nsd_{name}_mean"] = np.mean(vals)
        s[f"nsd_{name}_std"] = np.std(vals)
    summary.append(s)

save_pkl(summary, "ensemble_sensitivity_summary.pkl", subdir=OUT_DIR)

# ── Print summary ────────────────────────────────────────────────────────────
print("\n" + "=" * 85)
print("SUMMARY (mean ± std across runs)")
print("=" * 85)
print(f"{'N':>5s} | {'Time(s)':>10s} | {'NIS':>10s} | {'Met NRMSE':>12s} | {'Met Cov%':>10s} | {'NSD NRMSE':>12s} | {'Asn NRMSE':>12s}")
print("-" * 85)
for s in summary:
    print(f"{s['N']:>5d} | "
          f"{s['wall_time_mean']:>5.0f}±{s['wall_time_std']:>3.0f} | "
          f"{s['nis_mean']:>4.2f}±{s['nis_std']:>4.2f} | "
          f"{s['met_nrmse_mean']:>5.4f}±{s['met_nrmse_std']:>.4f} | "
          f"{s['met_cov_mean']:>4.1f}±{s['met_cov_std']:>4.1f}% | "
          f"{s['nsd_nrmse_mean']:>5.4f}±{s['nsd_nrmse_std']:>.4f} | "
          f"{s['asn_nrmse_mean']:>5.4f}±{s['asn_nrmse_std']:>.4f}")

# ── Figures ──────────────────────────────────────────────────────────────────
Ns = [s["N"] for s in summary]

fig, axes = plt.subplots(2, 3, figsize=(16, 9))

def plot_with_errorbars(ax, x, y, yerr, color, marker, ylabel, title):
    ax.errorbar(x, y, yerr=yerr, fmt=f"{marker}-", color=color, lw=2, ms=7, capsize=4)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Ensemble size N")
    ax.grid(alpha=0.2)

plot_with_errorbars(axes[0, 0], Ns,
    [s["met_nrmse_mean"] for s in summary],
    [s["met_nrmse_std"] for s in summary],
    "tab:red", "o", "Normalised RMSE",
    "(a) Measured metabolites\n(normalised RMSE)")

ax = axes[0, 1]
ax.errorbar(Ns, [s["nis_mean"] for s in summary], [s["nis_std"] for s in summary],
            fmt="s-", color="tab:blue", lw=2, ms=7, capsize=4)
ax.axhline(1.0, ls="--", color="gray", alpha=0.5, label="Ideal (1.0)")
ax.set_ylabel("Mean NIS"); ax.set_title("(b) Measured metabolites\n(normalised innovation squared)", fontsize=11)
ax.set_xlabel("Ensemble size N"); ax.legend(fontsize=9); ax.grid(alpha=0.2)

ax = axes[0, 2]
ax.errorbar(Ns, [s["met_cov_mean"] for s in summary], [s["met_cov_std"] for s in summary],
            fmt="^-", color="tab:green", lw=2, ms=7, capsize=4)
ax.axhline(95, ls="--", color="gray", alpha=0.5, label="Target (95%)")
ax.set_ylabel("Mean 2σ coverage (%)"); ax.set_title("(c) Measured metabolites\n(2σ coverage)", fontsize=11)
ax.set_xlabel("Ensemble size N"); ax.legend(fontsize=9); ax.grid(alpha=0.2)

# NSD per-state
ax = axes[1, 0]
for name in NSD_VALIDATED:
    ax.errorbar(Ns, [s[f"nsd_{name}_mean"] for s in summary],
                [s[f"nsd_{name}_std"] for s in summary],
                fmt="o-", lw=1.5, ms=5, capsize=3, label=name)
ax.errorbar(Ns, [s["nsd_nrmse_mean"] for s in summary],
            [s["nsd_nrmse_std"] for s in summary],
            fmt="k--", lw=2, ms=7, capsize=4, label="Mean")
ax.set_ylabel("Normalised RMSE"); ax.set_title("(d) Validated NSDs\n(normalised RMSE)", fontsize=11)
ax.set_xlabel("Ensemble size N"); ax.legend(fontsize=8); ax.grid(alpha=0.2)

plot_with_errorbars(axes[1, 1], Ns,
    [s["asn_nrmse_mean"] for s in summary],
    [s["asn_nrmse_std"] for s in summary],
    "tab:purple", "D", "Normalised RMSE",
    "(e) Asparagine\n(normalised RMSE)")

plot_with_errorbars(axes[1, 2], Ns,
    [s["wall_time_mean"] for s in summary],
    [s["wall_time_std"] for s in summary],
    "tab:gray", "o", "Wall time (s)",
    "(f) Computational cost\n(single run, P4)")

plt.tight_layout()
out_path = OUT_DIR / "ensemble_size_sensitivity.png"
plt.savefig(out_path, dpi=200, bbox_inches="tight")
print(f"\nSaved: {out_path}")
plt.close()

print("\nEnsemble sensitivity analysis complete.")
