"""
05_systematic_tuning.py
========================
Systematic EnKF covariance tuning on P4.

Procedure:
  1. R set from experimental error bars (biological triplicate variance)
  2. Initial ensemble covariance set independently of per-step Q
  3. KQ sweep with innovation diagnostics to select optimal process noise

Usage:
    poetry run python scripts/05_systematic_tuning.py
    poetry run python scripts/05_systematic_tuning.py --ensemble-size 200
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import warnings
import numpy as np
from scipy.linalg import solve, LinAlgWarning

warnings.filterwarnings("ignore", category=LinAlgWarning)

import nsd_enkf.config as cfg
from nsd_enkf.data_loader import (
    select_datasets, load_dataset, get_initial_condition, build_schedule,
)
from nsd_enkf.model import compute_volume_results, model_step
from nsd_enkf.io_utils import set_dirs, save_pkl, load_pkl, fig_path

# ── CLI ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Systematic EnKF tuning on P4")
parser.add_argument("--ensemble-size", default=cfg.ENSEMBLE_SIZE, type=int)
parser.add_argument("--run", default="tuning_v1")
args = parser.parse_args()

ENSEMBLE_SIZE = args.ensemble_size
RUN_NAME = args.run
TUNING_DATASET = "P4"

RESULTS_DIR = cfg.PROJECT_ROOT / "results" / RUN_NAME
S05_PKL = RESULTS_DIR / "05_tuning" / "pkl"
S05_FIG = RESULTS_DIR / "05_tuning" / "figures"
set_dirs(S05_PKL, S05_FIG)

np.random.seed(42)

print("=" * 70)
print(f"Systematic EnKF Tuning on {TUNING_DATASET}  [{RUN_NAME}]")
print(f"  ENSEMBLE_SIZE={ENSEMBLE_SIZE}")
print("=" * 70)


# ── Step 1: Compute R from experimental error bars ──────────────────────────

print("\n── Step 1: Computing R from experimental error bars ──")

meas_state_names = cfg.MEASURED_STATES  # ['Xv','mAb','Gal','Urd','Glc','Amm','Gln','Lac']
n_meas = cfg.MEAS_NUM

# Collect error bar variances across all datasets
all_err_var = {s: [] for s in meas_state_names}

for ds_name in ["P1", "P2", "P3", "P4"]:
    data = load_dataset(ds_name)
    err = data["set_meas_errorbar"].astype(float)
    for i, s in enumerate(meas_state_names):
        if i < err.shape[1]:
            # error bars are +/- std from biological triplicates
            # variance = mean(err^2) across timepoints
            err_vals = err[:, i]
            mask = ~np.isnan(err_vals) & (err_vals > 0)
            if mask.any():
                all_err_var[s].append(np.mean(err_vals[mask] ** 2))

R_empirical = np.zeros(n_meas)
print(f"\n  {'State':>8s} | {'R_empirical':>12s} | {'R_current':>12s} | {'Ratio new/old':>14s}")
print("  " + "-" * 56)
for i, s in enumerate(meas_state_names):
    R_empirical[i] = np.mean(all_err_var[s])
    R_old = list(cfg.MEASUREMENT_NOISE_VAR.values())[i]
    print(f"  {s:>8s} | {R_empirical[i]:12.4g} | {R_old:12.4g} | {R_empirical[i]/R_old:14.2f}x")

R_matrix = np.diag(R_empirical)

save_pkl({"R_empirical": R_empirical, "R_matrix": R_matrix,
          "meas_state_names": meas_state_names}, "R_empirical.pkl")
print("\n  R_empirical saved.")


# ── Step 2: Build shared data ───────────────────────────────────────────────

print("\n── Step 2: Loading/building shared data ──")

ALL_DATASETS = select_datasets("P1", "P2", "P3", "P4")
time_grid = np.arange(cfg.DT, cfg.T_END + cfg.DT, cfg.DT)
step_len = np.full(len(time_grid), cfg.DT)
dt_model = cfg.DT
N_model = int(cfg.T_END / dt_model)
T_model = np.linspace(0, cfg.T_END, N_model + 1)
T_kf = T_model
dt_kf = dt_model
N_kf = len(T_kf) - 1

# H matrix
A = np.identity(n_meas)
B = np.zeros((n_meas, cfg.STATE_NUM - n_meas))
H = np.hstack((A, B))

# Load or compute shared data
shared_pkl = cfg.PROJECT_ROOT / "results" / "run_v3" / "01_run_enkf" / "pkl"
if shared_pkl.exists() and (shared_pkl / "volume_results.pkl").exists():
    print("  Loading shared data from run_v3 ...")
    volume_results = load_pkl("volume_results.pkl", subdir=shared_pkl)
    state_init_by_dataset = load_pkl("state_init_by_dataset.pkl", subdir=shared_pkl)
    set_model_by_dataset = load_pkl("set_model_by_dataset.pkl", subdir=shared_pkl)
    T_meas_by_dataset = load_pkl("T_meas_by_dataset.pkl", subdir=shared_pkl)
else:
    print("  Computing shared data from scratch ...")
    state_init_by_dataset = {}
    for name in ALL_DATASETS:
        data = load_dataset(name)
        _, state_init = get_initial_condition(data["met_df"], data["nsd_df"])
        state_init_by_dataset[name] = state_init

    volume_results = compute_volume_results(
        ALL_DATASETS, cfg.INITIAL_VOLUMES, build_schedule, step_len
    )

    set_model_by_dataset = {}
    from nsd_enkf.model import simulate_dataset
    for name in ALL_DATASETS:
        Fin, Fout, Gal_feed, Urd_feed = build_schedule(name)
        V_traj = volume_results[name][1:]
        traj = simulate_dataset(
            state_init_by_dataset[name], Fin, Fout, Gal_feed, Urd_feed,
            V_traj, time_grid, step_len, name=name,
        )
        set_model_by_dataset[name] = np.vstack([state_init_by_dataset[name], traj])

    T_meas_by_dataset = {}
    for name in ALL_DATASETS:
        data = load_dataset(name)
        N_meas_time = data["set_meas"].shape[0]
        if len(cfg.T_MEAS_FIXED) == N_meas_time:
            T_meas_by_dataset[name] = np.array(cfg.T_MEAS_FIXED)
        else:
            interval = int(24.0 / dt_model)
            T_index_meas = [i * interval for i in range(N_meas_time)]
            T_meas_by_dataset[name] = T_model[T_index_meas]

# P4-specific data
data_P4 = load_dataset(TUNING_DATASET)
set_meas_P4 = data_P4["set_meas"][:, :n_meas].astype(float)
T_meas_P4 = T_meas_by_dataset[TUNING_DATASET]
state_init_P4 = state_init_by_dataset[TUNING_DATASET].copy()

Fin, Fout, Gal_feed, Urd_feed = build_schedule(TUNING_DATASET)
V_traj_P4 = volume_results[TUNING_DATASET][1:]

# Time step mapping
time_steps_A = [round(i * dt_kf, 2) for i in range(N_kf)]
time_steps_B = [round(t, 2) for t in T_meas_P4.tolist()]
meas_time_to_index = {t: i for i, t in enumerate(time_steps_B)}


# ── Step 3: Initial ensemble covariance ─────────────────────────────────────

print("\n── Step 3: Setting initial ensemble covariance ──")

# For measured states: use biological error bar variance (how uncertain x0 is)
# For unmeasured states: use process noise variance (prior uncertainty)
var_model = np.array(list(cfg.PROCESS_NOISE_VAR.values()))
P0_diag = var_model.copy()

# Override measured states with empirical measurement variance
for i in range(n_meas):
    P0_diag[i] = R_empirical[i]

P0 = np.diag(P0_diag)

print(f"\n  {'State':>12s} | {'P0 (init cov)':>14s} | {'Source':>20s}")
print("  " + "-" * 54)
for i, s in enumerate(cfg.STATE_NAMES):
    source = "error bars" if i < n_meas else "process noise var"
    print(f"  {s:>12s} | {P0_diag[i]:14.4g} | {source:>20s}")

save_pkl({"P0_diag": P0_diag, "P0": P0}, "P0_initial_cov.pkl")


# ── Step 4: KQ sweep with innovation diagnostics ───────────────────────────

print("\n── Step 4: KQ sweep on P4 ──")

KQ_VALUES = [1e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3]

print(f"  Sweeping KQ = {[f'{k:.0e}' for k in KQ_VALUES]}")
print(f"  Ensemble size = {ENSEMBLE_SIZE}")


def run_enkf_with_diagnostics(kq, R, P0, H, state_init, ensemble_size,
                              set_meas, T_meas, n_meas, state_num,
                              Fin, Fout, Gal_feed, Urd_feed, V_traj,
                              dt_kf, N_kf, time_steps_A, meas_time_to_index):
    """
    Run single EnKF pass on P4 and collect innovation diagnostics.

    Returns dict with: RMSE, coverage, innovation stats, ensemble spread, timing.
    """
    var_model = np.array(list(cfg.PROCESS_NOISE_VAR.values()))
    Q = kq * np.diag(var_model)

    # Generate measurement ensemble (once)
    meas_std = np.sqrt(np.diag(R))
    N_meas_time = set_meas.shape[0]
    set_meas_ens = np.zeros((N_meas_time, ensemble_size, n_meas))
    for i in range(N_meas_time):
        noise = np.random.multivariate_normal(
            np.zeros(n_meas), R, size=ensemble_size
        )
        for j in range(n_meas):
            noise[:, j] = np.clip(noise[:, j], -3 * meas_std[j], 3 * meas_std[j])
        set_meas_ens[i] = np.clip(set_meas[i] + noise, a_min=1e-12, a_max=None)

    # Initialize ensemble with P0 (NOT Q)
    X = np.random.multivariate_normal(state_init, P0, ensemble_size)
    for i in range(state_num):
        sd = np.sqrt(P0[i, i])
        X[:, i] = np.clip(X[:, i], state_init[i] - 3 * sd, state_init[i] + 3 * sd)
    X = np.clip(X, a_min=1e-12, a_max=None)
    x_mean = np.mean(X, axis=0)

    # Storage
    mean_traj = [x_mean.copy()]
    std_traj = [np.std(X, axis=0)]
    innovations = []          # z - H*x_mean_forecast
    innov_covs = []           # P_zz at each update
    meas_at_updates = []      # actual measurements at update times

    t_start = time.time()

    for idx_A, step_A in enumerate(time_steps_A):
        # Predict
        X_new = []
        for x in X:
            X_new.append(model_step(x, 0.0, {
                "Fin": Fin[idx_A], "Fout": Fout[idx_A], "V": V_traj[idx_A],
                "Gal_feed": Gal_feed[idx_A], "Urd_feed": Urd_feed[idx_A],
            }, dt_kf))

        noise = np.random.multivariate_normal(
            np.zeros(state_num), Q, size=ensemble_size
        )
        for i in range(state_num):
            sd = np.sqrt(Q[i, i])
            noise[:, i] = np.clip(noise[:, i], -3 * sd, 3 * sd)

        X = np.array(X_new) + noise
        X = np.clip(X, a_min=1e-12, a_max=None)
        x_mean = np.mean(X, axis=0)

        # Update if measurement available
        if step_A in meas_time_to_index:
            idx_B = meas_time_to_index[step_A]
            z_ens = set_meas_ens[idx_B]
            z_actual = set_meas[idx_B]

            # Forecast ensemble in measurement space
            Z = np.array([H @ x for x in X])
            z_pred_mean = np.mean(Z, axis=0)

            # Innovation
            d = z_actual - z_pred_mean

            # Ensemble anomalies
            E_x = X - x_mean
            E_z = Z - z_pred_mean

            P_xz = (E_x.T @ E_z) / (ensemble_size - 1)
            P_zz = (E_z.T @ E_z) / (ensemble_size - 1) + R

            K = solve(P_zz.T, P_xz.T, assume_a='pos').T

            X += (K @ (z_ens - Z).T).T
            X = np.clip(X, a_min=1e-12, a_max=None)
            x_mean = np.mean(X, axis=0)

            innovations.append(d)
            innov_covs.append(P_zz)
            meas_at_updates.append(z_actual)

        mean_traj.append(x_mean.copy())
        std_traj.append(np.std(X, axis=0))

    wall_time = time.time() - t_start

    mean_traj = np.array(mean_traj)
    std_traj = np.array(std_traj)
    innovations = np.array(innovations)
    innov_covs = np.array(innov_covs)

    # ── Compute diagnostics ──

    # 1. Normalized Innovation Squared (NIS): d' S^{-1} d / n_z ~ 1
    nis_values = []
    for d, S in zip(innovations, innov_covs):
        nis = d @ np.linalg.solve(S, d) / n_meas
        nis_values.append(nis)
    nis_values = np.array(nis_values)

    # 2. Per-state normalized innovation: d_i / sqrt(S_ii) ~ N(0,1)
    norm_innov_per_state = np.zeros((len(innovations), n_meas))
    for k in range(len(innovations)):
        for j in range(n_meas):
            norm_innov_per_state[k, j] = innovations[k, j] / np.sqrt(innov_covs[k, j, j])

    # 3. Coverage: fraction of measurements within ±2σ of ensemble mean
    # At measurement times, check if measurement falls in mean ± 2*std
    coverage_per_state = np.zeros(n_meas)
    n_updates = len(meas_at_updates)
    meas_time_indices = []
    for step_A in time_steps_A:
        if step_A in meas_time_to_index:
            # Index in mean_traj: idx_A + 1 (since mean_traj[0] is initial)
            idx_A = time_steps_A.index(step_A)
            meas_time_indices.append(idx_A + 1)

    for j in range(n_meas):
        in_band = 0
        for k, traj_idx in enumerate(meas_time_indices):
            m = mean_traj[traj_idx, j]
            s = std_traj[traj_idx, j]
            z = meas_at_updates[k][j]
            if not np.isnan(z) and s > 0:
                if abs(z - m) <= 2 * s:
                    in_band += 1
        coverage_per_state[j] = in_band / n_updates * 100

    # 4. Mean ensemble CV for measured states over full trajectory
    mean_cv = np.zeros(n_meas)
    for j in range(n_meas):
        means = mean_traj[:, j]
        stds = std_traj[:, j]
        valid = means > 1e-12
        if valid.any():
            mean_cv[j] = np.mean(stds[valid] / means[valid]) * 100

    # 5. RMSE on P4 (measured states)
    rmse = np.zeros(n_meas)
    for j in range(n_meas):
        measured = set_meas[:, j]
        enkf_pred = np.interp(T_meas, T_kf, mean_traj[:, j])
        mask = ~np.isnan(measured)
        if mask.any():
            rmse[j] = np.sqrt(np.mean((measured[mask] - enkf_pred[mask]) ** 2))

    return {
        "kq": kq,
        "wall_time_s": wall_time,
        "mean_traj": mean_traj,
        "std_traj": std_traj,
        "innovations": innovations,
        "innov_covs": innov_covs,
        "nis_mean": np.mean(nis_values),
        "nis_values": nis_values,
        "norm_innov_per_state": norm_innov_per_state,
        "norm_innov_var": np.var(norm_innov_per_state, axis=0),
        "coverage_2sigma": coverage_per_state,
        "mean_cv_pct": mean_cv,
        "rmse": rmse,
    }


# ── Run the sweep ───────────────────────────────────────────────────────────

from tqdm import tqdm

results = []
for kq in KQ_VALUES:
    print(f"\n  KQ = {kq:.0e} ...", flush=True)
    res = run_enkf_with_diagnostics(
        kq=kq, R=R_matrix, P0=P0, H=H,
        state_init=state_init_P4, ensemble_size=ENSEMBLE_SIZE,
        set_meas=set_meas_P4, T_meas=T_meas_P4,
        n_meas=n_meas, state_num=cfg.STATE_NUM,
        Fin=Fin, Fout=Fout, Gal_feed=Gal_feed, Urd_feed=Urd_feed,
        V_traj=V_traj_P4, dt_kf=dt_kf, N_kf=N_kf,
        time_steps_A=time_steps_A, meas_time_to_index=meas_time_to_index,
    )
    results.append(res)
    print(f"    Wall time: {res['wall_time_s']:.1f}s | NIS mean: {res['nis_mean']:.3f} "
          f"(ideal ~1.0)", flush=True)

save_pkl(results, "kq_sweep_results.pkl")


# ── Step 5: Summary table ──────────────────────────────────────────────────

print("\n" + "=" * 70)
print("SUMMARY: KQ Sweep Results")
print("=" * 70)

# Header
header = (f"{'KQ':>8s} | {'NIS':>6s} | {'Time(s)':>7s} | "
          + " | ".join(f"CV_{s[:3]:>3s}" for s in meas_state_names)
          + " | "
          + " | ".join(f"Cov_{s[:3]:>3s}" for s in meas_state_names))
print(header)
print("-" * len(header))

for res in results:
    line = f"{res['kq']:>8.0e} | {res['nis_mean']:>6.3f} | {res['wall_time_s']:>7.1f} | "
    line += " | ".join(f"{v:>7.1f}" for v in res["mean_cv_pct"])
    line += " | "
    line += " | ".join(f"{v:>7.0f}" for v in res["coverage_2sigma"])
    print(line)

print("\n── Per-state normalized innovation variance (ideal ~1.0) ──")
header2 = f"{'KQ':>8s} | " + " | ".join(f"{s:>8s}" for s in meas_state_names)
print(header2)
print("-" * len(header2))
for res in results:
    line = f"{res['kq']:>8.0e} | "
    line += " | ".join(f"{v:>8.3f}" for v in res["norm_innov_var"])
    print(line)

print("\n── RMSE on P4 ──")
header3 = f"{'KQ':>8s} | " + " | ".join(f"{s:>10s}" for s in meas_state_names)
print(header3)
print("-" * len(header3))

# Also print open-loop RMSE for comparison
model_traj = set_model_by_dataset[TUNING_DATASET]
model_rmse = []
for j in range(n_meas):
    measured = set_meas_P4[:, j]
    model_pred = np.interp(T_meas_P4, T_model, model_traj[:, j])
    mask = ~np.isnan(measured)
    if mask.any():
        model_rmse.append(np.sqrt(np.mean((measured[mask] - model_pred[mask]) ** 2)))
    else:
        model_rmse.append(np.nan)

line = f"{'Model':>8s} | " + " | ".join(f"{v:>10.3f}" for v in model_rmse)
print(line)
for res in results:
    line = f"{res['kq']:>8.0e} | "
    line += " | ".join(f"{v:>10.3f}" for v in res["rmse"])
    print(line)

# ── Recommendation ──────────────────────────────────────────────────────────
print("\n── Recommendation ──")
print("Look for KQ where:")
print("  1. NIS ~ 1.0 (filter is statistically consistent)")
print("  2. Coverage ~ 95% for 2-sigma bands")
print("  3. Normalized innovation variance ~ 1.0 per state")
print("  4. RMSE is improved over open-loop model")
print("  5. CV shows meaningful (non-collapsed) ensemble spread")

# Find best KQ by NIS closest to 1.0
best_idx = np.argmin([abs(r["nis_mean"] - 1.0) for r in results])
best = results[best_idx]
print(f"\n  Best KQ by NIS criterion: {best['kq']:.0e} (NIS = {best['nis_mean']:.3f})")
print(f"  Mean 2σ coverage: {np.mean(best['coverage_2sigma']):.1f}%")

print("\nTuning complete. Results saved to:", S05_PKL)
