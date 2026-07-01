"""
ensemble_size_sensitivity.py
============================
Ensemble-size sensitivity + calibration diagnostics on P4, using the CURRENT
production filter configuration (self-contained — no dependency on prior runs).

Directly targets Reviewer 2.1 (is N=100 justified? sensitivity, instability) and
Reviewer 3.4 (does the ensemble keep meaningful spread / stay calibrated, not just
accurate in the mean?).

For each ensemble size N, N_RUNS independent EnKF passes are run on P4 with the
exact production settings pulled from config:
  - measured states  -> multiplicative CV noise (PROCESS_NOISE_CV)
  - unmeasured states -> additive two-stage-alpha noise (PROCESS_NOISE_VAR)
  - IQR clipping on CLIP_STATES, localization on NO_UPDATE_STATES
  - P0: measured = measurement variance, unmeasured = process-noise variance

Metrics per size (mean +/- std across runs):
  measured : normalised RMSE, NIS = mean(d^2/S) [ideal 1], 2-sigma coverage %
  NSD (7)  : normalised RMSE, 2-sigma coverage %, spread-skill = std/RMSE [ideal 1]
  Asn      : normalised RMSE
  cost     : wall-clock seconds per pass

Crash-safe: each size is saved immediately; --resume skips sizes already on disk.

Usage:
    # Recommended overnight run (drops N=300; full 10 seeds) -- ~8.4 h wall time:
    ./.venv/Scripts/python.exe scripts/ensemble_size_sensitivity.py --sizes 25,50,75,100,150,200 --n-runs 10 --run ensemble_sens

    # Full default sweep incl. N=300 -- ~12.6 h:
    ./.venv/Scripts/python.exe scripts/ensemble_size_sensitivity.py --n-runs 10 --run ensemble_sens

    # Resume a killed/interrupted run (each size is saved as it finishes):
    ./.venv/Scripts/python.exe scripts/ensemble_size_sensitivity.py --sizes 25,50,75,100,150,200 --n-runs 10 --run ensemble_sens --resume

Runtime (measured, scales ~linearly with N): ~8.4 min per pass at N=100, so total
wall time ~= 5.05 s * (sum of sizes / 100) * n_runs. Uses the current config.py
(tuned_v6 measured CVs + Option B alpha=0.01/0.001), NOT the cv_tuning checkpoint.

Trajectories: each run also stores its mean/std trajectory (downsampled by --traj-down,
default x20) plus the raw innovations and S at the measurement-update times, so any
trajectory-level statistic can be recomputed later without re-running. Pass
--traj-down 0 for metrics only.
"""

import argparse
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

warnings.filterwarnings("ignore", category=LinAlgWarning)

import nsd_enkf.config as cfg
from nsd_enkf.data_loader import (
    select_datasets, load_dataset, get_initial_condition, build_schedule,
)
from nsd_enkf.model import compute_volume_results, model_step
from nsd_enkf.enkf import EnsembleKalmanFilter

# ── CLI ──────────────────────────────────────────────────────────────────────
p = argparse.ArgumentParser(description="Ensemble-size sensitivity + calibration on P4")
p.add_argument("--run", default="ensemble_sens")
p.add_argument("--dataset", default="P4")
p.add_argument("--sizes", default="25,50,75,100,150,200,300")
p.add_argument("--n-runs", default=10, type=int)
p.add_argument("--seed-offset", default=42, type=int)
p.add_argument("--resume", action="store_true", help="skip sizes whose pkl already exists")
p.add_argument("--traj-down", default=20, type=int,
               help="also save per-run mean/std trajectories downsampled by this factor "
                    "(0 = metrics only, no trajectories)")
args = p.parse_args()

DS = args.dataset
SIZES = [int(x) for x in args.sizes.split(",")]
N_RUNS = args.n_runs

OUT_DIR = cfg.PROJECT_ROOT / "results" / args.run / "ensemble_sensitivity"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def save_pkl(obj, name):
    tmp = OUT_DIR / (name + ".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(obj, f)
    tmp.replace(OUT_DIR / name)

print("=" * 64)
print(f"Ensemble-size sensitivity  [run={args.run}, dataset={DS}]")
print(f"  sizes = {SIZES} | runs/size = {N_RUNS}")
print(f"  trajectories: " + (f"saved, downsampled x{args.traj_down}"
                             if args.traj_down > 0 else "not saved (metrics only)"))
print("=" * 64)

# ── Fixed config (current production filter) ─────────────────────────────────
time_grid = np.arange(cfg.DT, cfg.T_END + cfg.DT, cfg.DT)
step_len = np.full(len(time_grid), cfg.DT)
N_kf = int(cfg.T_END / cfg.DT)
T_kf = np.linspace(0, cfg.T_END, N_kf + 1)
dt_kf = cfg.DT

var_model = np.array(list(cfg.PROCESS_NOISE_VAR.values()))   # two-stage alpha already baked in
var_meas = np.array(list(cfg.MEASUREMENT_NOISE_VAR.values()))
Q = np.diag(var_model)
R = np.diag(var_meas[:cfg.MEAS_NUM])
H = np.hstack((np.eye(cfg.MEAS_NUM), np.zeros((cfg.MEAS_NUM, cfg.STATE_NUM - cfg.MEAS_NUM))))
meas_std = np.sqrt(np.diag(R))

process_noise_cv = {cfg.STATE_NAMES.index(s): cv for s, cv in cfg.PROCESS_NOISE_CV.items()}
no_update_indices = {cfg.STATE_NAMES.index(s) for s in getattr(cfg, "NO_UPDATE_STATES", [])}
clip_indices = {cfg.STATE_NAMES.index(s) for s in getattr(cfg, "CLIP_STATES", [])}

# P0: measured -> measurement variance, unmeasured -> process-noise variance
P0_diag = var_model.copy()
P0_diag[:cfg.MEAS_NUM] = var_meas[:cfg.MEAS_NUM]
P0 = np.diag(P0_diag)

# ── Shared data for the dataset (computed inline; no prior-run dependency) ────
volume_results = compute_volume_results(select_datasets(DS), cfg.INITIAL_VOLUMES,
                                        build_schedule, step_len)
data = load_dataset(DS)
_, state_init = get_initial_condition(data["met_df"], data["nsd_df"])
Fin, Fout, Gal_feed, Urd_feed = build_schedule(DS)
V_traj = volume_results[DS][1:]

set_meas = data["set_meas"][:, :cfg.MEAS_NUM].astype(float)
asn_full = data["set_meas"][:, 8].astype(float) if data["set_meas"].shape[1] > 8 else None
nsd_vals = pd.DataFrame(data["NSD_meas"]).apply(pd.to_numeric, errors="coerce").to_numpy()

T_meas = np.array(cfg.T_MEAS_FIXED)
time_steps_A = [round(i * dt_kf, 2) for i in range(N_kf)]
meas_time_to_index = {round(t, 2): i for i, t in enumerate(T_meas.tolist())}
meas_grid_idx = [min(int(round(t / dt_kf)) + 1, N_kf) for t in T_meas]  # post-update index

n_nsd = nsd_vals.shape[1]
nsd_state_idx = list(range(cfg.STATE_NUM - n_nsd, cfg.STATE_NUM))
nsd_names = [cfg.STATE_NAMES[i] for i in nsd_state_idx]
ASN_IDX = cfg.STATE_NAMES.index("Asn")

# Normalisation scales = median |measurement| per state
def med_scale(v):
    v = v[~np.isnan(v)]
    return max(np.median(np.abs(v)), 1e-12) if v.size else 1.0
meas_scales = np.array([med_scale(set_meas[:, j]) for j in range(cfg.MEAS_NUM)])
nsd_scales = np.array([med_scale(nsd_vals[:, j]) for j in range(n_nsd)])
asn_scale = med_scale(asn_full) if asn_full is not None else 1.0


def run_pass(N, seed):
    """One production-config EnKF pass; return per-metric diagnostics."""
    np.random.seed(seed)
    # perturbed measurement ensemble
    N_meas_time = set_meas.shape[0]
    set_meas_ens = np.zeros((N_meas_time, N, cfg.MEAS_NUM))
    for i in range(N_meas_time):
        noise = np.random.multivariate_normal(np.zeros(cfg.MEAS_NUM), R, size=N)
        for j in range(cfg.MEAS_NUM):
            noise[:, j] = np.clip(noise[:, j], -3 * meas_std[j], 3 * meas_std[j])
        set_meas_ens[i] = np.clip(set_meas[i] + noise, a_min=1e-12, a_max=None)

    enkf = EnsembleKalmanFilter(cfg.STATE_NUM, cfg.MEAS_NUM)
    enkf.x = state_init.copy()
    enkf.Q = Q.copy(); enkf.R = R.copy(); enkf.H = H.copy()
    enkf.fx = model_step; enkf.dt = dt_kf
    enkf.process_noise_cv = dict(process_noise_cv)
    enkf.no_update_indices = set(no_update_indices)
    enkf.clip_indices = set(clip_indices)          # <-- production clipping (was missing)
    enkf.create_ensemble(N, P0)

    mean_traj = [enkf.x.copy()]
    std_traj = [np.std(enkf.X, axis=0)]
    innovations, innov_covs, meas_at_updates = [], [], []

    t0 = time.time()
    for idx_A, step_A in enumerate(time_steps_A):
        enkf.predict({"Fin": Fin[idx_A], "Fout": Fout[idx_A], "V": V_traj[idx_A],
                      "Gal_feed": Gal_feed[idx_A], "Urd_feed": Urd_feed[idx_A]})
        if step_A in meas_time_to_index:
            b = meas_time_to_index[step_A]
            Z = enkf.X @ enkf.H.T
            z_mean = Z.mean(axis=0)
            Ez = Z - z_mean
            S = (Ez.T @ Ez) / (N - 1) + R
            innovations.append(set_meas[b] - z_mean)
            innov_covs.append(np.diag(S).copy())
            meas_at_updates.append(set_meas[b])
            enkf.update(set_meas_ens[b])
        mean_traj.append(enkf.x.copy())
        std_traj.append(np.std(enkf.X, axis=0))
    wall = time.time() - t0

    mean_traj = np.array(mean_traj); std_traj = np.array(std_traj)
    innovations = np.array(innovations); innov_covs = np.array(innov_covs)
    n_up = len(innovations)

    # measured: NRMSE, NIS = mean(d^2/S), 2-sigma coverage
    m_nrmse = np.zeros(cfg.MEAS_NUM); m_nis = np.zeros(cfg.MEAS_NUM); m_cov = np.zeros(cfg.MEAS_NUM)
    for j in range(cfg.MEAS_NUM):
        pred = np.interp(T_meas, T_kf, mean_traj[:, j])
        mask = ~np.isnan(set_meas[:, j])
        m_nrmse[j] = np.sqrt(np.mean((set_meas[mask, j] - pred[mask]) ** 2)) / meas_scales[j]
        m_nis[j] = np.mean(innovations[:, j] ** 2 / innov_covs[:, j])
        m = mean_traj[meas_grid_idx, j]; s = std_traj[meas_grid_idx, j]
        mm = ~np.isnan(set_meas[:, j]) & (s > 0)
        m_cov[j] = 100.0 * np.mean(np.abs(set_meas[mm, j] - m[mm]) <= 2 * s[mm]) if mm.any() else np.nan

    # NSD (all 7): NRMSE, coverage, spread-skill
    nsd_nrmse = {}; nsd_cov = {}; nsd_ss = {}
    for col, si in enumerate(nsd_state_idx):
        meas = nsd_vals[:, col]
        m = mean_traj[meas_grid_idx, si]; s = std_traj[meas_grid_idx, si]
        valid = ~np.isnan(meas) & (s > 0)
        if valid.sum() == 0:
            nsd_nrmse[nsd_names[col]] = nsd_cov[nsd_names[col]] = nsd_ss[nsd_names[col]] = np.nan
            continue
        err = meas[valid] - m[valid]
        rmse = np.sqrt(np.mean(err ** 2))
        nsd_nrmse[nsd_names[col]] = rmse / nsd_scales[col]
        nsd_cov[nsd_names[col]] = 100.0 * np.mean(np.abs(err) <= 2 * s[valid])
        nsd_ss[nsd_names[col]] = np.mean(s[valid]) / rmse if rmse > 0 else np.nan

    # Asn
    if asn_full is not None:
        pred = np.interp(T_meas, T_kf, mean_traj[:, ASN_IDX])
        mask = ~np.isnan(asn_full)
        asn_nrmse = np.sqrt(np.mean((asn_full[mask] - pred[mask]) ** 2)) / asn_scale
    else:
        asn_nrmse = np.nan

    out = {
        "wall_time_s": wall,
        "meas_nrmse_mean": np.mean(m_nrmse), "meas_nis_mean": np.mean(m_nis),
        "meas_cov_mean": np.nanmean(m_cov),
        "nsd_nrmse": nsd_nrmse, "nsd_cov": nsd_cov, "nsd_ss": nsd_ss,
        "nsd_nrmse_mean": np.nanmean(list(nsd_nrmse.values())),
        "nsd_ss_median": np.nanmedian(list(nsd_ss.values())),
        "asn_nrmse": asn_nrmse,
    }

    # Also persist the (downsampled) mean/std trajectories + the raw innovations at
    # update times, so any trajectory-level statistic can be recomputed later without
    # re-running. Full ensemble is NOT stored (too large); mean+std+innovations
    # reconstruct every diagnostic this script reports.
    d = int(args.traj_down)
    if d > 0:
        out["traj"] = {
            "down": d,
            "state_names": list(cfg.STATE_NAMES),
            "T": T_kf[::d].copy(),              # (M,)      downsampled time grid
            "mean": mean_traj[::d].copy(),      # (M, 17)   ensemble mean
            "std": std_traj[::d].copy(),        # (M, 17)   ensemble std (uncertainty band)
        }
        out["innov"] = {                        # at the 17 measurement-update times
            "T_meas": T_meas.copy(),
            "meas_names": list(cfg.MEASURED_STATES),
            "d": innovations.copy(),            # (n_up, 8) innovation z - forecast mean
            "S_diag": innov_covs.copy(),        # (n_up, 8) diag(P_zz + R)
            "z": np.array(meas_at_updates),     # (n_up, 8) actual measurements
        }
    return out


# ── Sweep (save each size immediately; resumable) ────────────────────────────
all_results = {}
for N in SIZES:
    pkl_name = f"ensemble_N{N}.pkl"
    if args.resume and (OUT_DIR / pkl_name).exists():
        with open(OUT_DIR / pkl_name, "rb") as f:
            all_results[N] = pickle.load(f)
        print(f"\n  N={N}: loaded from disk (resume).")
        continue
    print(f"\n  N={N}:", flush=True)
    runs = []
    for run_i in range(N_RUNS):
        res = run_pass(N, args.seed_offset + run_i)
        runs.append(res)
        print(f"    run {run_i+1}/{N_RUNS}: {res['wall_time_s']:.0f}s  "
              f"NIS={res['meas_nis_mean']:.2f} NSD-ss={res['nsd_ss_median']:.2f}", flush=True)
    all_results[N] = runs
    save_pkl(runs, pkl_name)      # crash-safe: persist this size before moving on

# ── Aggregate ────────────────────────────────────────────────────────────────
def agg(N, key):
    return np.array([r[key] for r in all_results[N]])

summary = []
for N in SIZES:
    row = {"N": N}
    for key in ["wall_time_s", "meas_nrmse_mean", "meas_nis_mean", "meas_cov_mean",
                "nsd_nrmse_mean", "nsd_ss_median", "asn_nrmse"]:
        v = agg(N, key)
        row[key + "_mean"] = np.nanmean(v); row[key + "_std"] = np.nanstd(v)
    for name in nsd_names:
        v = np.array([r["nsd_nrmse"][name] for r in all_results[N]])
        row[f"nsd_{name}_mean"] = np.nanmean(v); row[f"nsd_{name}_std"] = np.nanstd(v)
    summary.append(row)
save_pkl(summary, "ensemble_sensitivity_summary.pkl")

print("\n" + "=" * 88)
print(f"{'N':>5s} | {'Time(s)':>9s} | {'NIS':>9s} | {'MetNRMSE':>10s} | {'MetCov%':>9s} | "
      f"{'NSD_NRMSE':>10s} | {'NSD_ss':>8s}")
print("-" * 88)
for s in summary:
    print(f"{s['N']:>5d} | {s['wall_time_s_mean']:>4.0f}±{s['wall_time_s_std']:>3.0f} | "
          f"{s['meas_nis_mean_mean']:>4.2f}±{s['meas_nis_mean_std']:>3.2f} | "
          f"{s['meas_nrmse_mean_mean']:>10.4f} | "
          f"{s['meas_cov_mean_mean']:>4.0f}±{s['meas_cov_mean_std']:>3.0f} | "
          f"{s['nsd_nrmse_mean_mean']:>10.4f} | "
          f"{s['nsd_ss_median_mean']:>4.2f}±{s['nsd_ss_median_std']:>3.2f}")

# ── Figure ───────────────────────────────────────────────────────────────────
Ns = [s["N"] for s in summary]
def eb(ax, key, color, marker, ylabel, title, ref=None, reflabel=None):
    ax.errorbar(Ns, [s[key + "_mean"] for s in summary], [s[key + "_std"] for s in summary],
                fmt=f"{marker}-", color=color, lw=2, ms=7, capsize=4)
    if ref is not None:
        ax.axhline(ref, ls="--", color="gray", alpha=0.6, label=reflabel); ax.legend(fontsize=9)
    ax.set_xlabel("Ensemble size N"); ax.set_ylabel(ylabel); ax.set_title(title, fontsize=11)
    ax.grid(alpha=0.2)

fig, ax = plt.subplots(2, 3, figsize=(16, 9))
eb(ax[0, 0], "meas_nrmse_mean", "tab:red", "o", "Normalised RMSE", "(a) Measured — NRMSE")
eb(ax[0, 1], "meas_nis_mean", "tab:blue", "s", "Mean NIS", "(b) Measured — consistency (NIS)", 1.0, "ideal 1.0")
eb(ax[0, 2], "meas_cov_mean", "tab:green", "^", "2σ coverage (%)", "(c) Measured — 2σ coverage", 95, "target 95%")
axd = ax[1, 0]
for name in nsd_names:
    axd.errorbar(Ns, [s[f"nsd_{name}_mean"] for s in summary],
                 [s[f"nsd_{name}_std"] for s in summary], fmt="o-", lw=1.2, ms=4, capsize=2, label=name)
axd.errorbar(Ns, [s["nsd_nrmse_mean_mean"] for s in summary],
             [s["nsd_nrmse_mean_std"] for s in summary], fmt="k--", lw=2, ms=6, capsize=4, label="mean")
axd.set_xlabel("Ensemble size N"); axd.set_ylabel("Normalised RMSE")
axd.set_title("(d) NSDs — NRMSE", fontsize=11); axd.legend(fontsize=7, ncol=2); axd.grid(alpha=0.2)
eb(ax[1, 1], "nsd_ss_median", "tab:orange", "D", "Spread-skill (std/RMSE)",
   "(e) NSDs — calibration (spread-skill)", 1.0, "ideal 1.0")
eb(ax[1, 2], "wall_time_s", "tab:gray", "o", "Wall time (s)", "(f) Computational cost / pass")
plt.tight_layout()
out = OUT_DIR / "ensemble_size_sensitivity.png"
plt.savefig(out, dpi=200, bbox_inches="tight"); plt.close()
print(f"\nSaved figure: {out}")
print("Done.")
