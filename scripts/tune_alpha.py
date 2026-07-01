"""
tune_alpha.py
=============
Re-derive the single universal process-noise scalar ALPHA (Option B) on P4.

Unmeasured states use additive noise Q_ii = (alpha * scale_i)^2, where scale_i is
the fixed climatological std from config (PROCESS_NOISE_SCALE) and alpha is the
one tunable knob shared by all unmeasured states. Measured states keep their
per-state multiplicative CV noise (PROCESS_NOISE_CV), which is unaffected by alpha.

Because the NSDs are structurally unobservable, alpha cannot be calibrated from
the measured-state innovations. It is instead calibrated so the ENSEMBLE SPREAD
of the unmeasured states is statistically consistent with the withheld NSD
measurements on the tuning dataset P4:
  - 2-sigma coverage  : fraction of NSD measurements within mean +/- 2*std (target ~95%)
  - spread-skill ratio: mean ensemble std / RMSE at measurement times   (target ~1.0)

Only P4 is used here; P1-P3 remain fully withheld for validation.

Usage:
    ./.venv/Scripts/python.exe scripts/tune_alpha.py
    ./.venv/Scripts/python.exe scripts/tune_alpha.py --alphas 0.02,0.03,0.046,0.06,0.1
    ./.venv/Scripts/python.exe scripts/tune_alpha.py --ensemble-size 100
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import warnings
import numpy as np
import pandas as pd
from scipy.linalg import LinAlgWarning

warnings.filterwarnings("ignore", category=LinAlgWarning)

import nsd_enkf.config as cfg
from nsd_enkf.data_loader import (
    select_datasets, load_dataset, get_initial_condition, build_schedule,
)
from nsd_enkf.model import compute_volume_results, simulate_dataset
from nsd_enkf.analysis import generate_measurement_ensembles
from nsd_enkf.enkf import run_enkf_single_with_ensemble_diagnostics

# ── CLI ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Re-derive universal ALPHA on P4")
parser.add_argument("--alphas", default="0.01,0.02,0.03,0.046,0.06,0.1")
parser.add_argument("--ensemble-size", default=cfg.ENSEMBLE_SIZE, type=int)
parser.add_argument("--seed", default=42, type=int)
args = parser.parse_args()

ALPHAS = [float(a) for a in args.alphas.split(",")]
ENSEMBLE_SIZE = args.ensemble_size
TUNING_DATASET = "P4"

print("=" * 70)
print(f"Re-deriving universal ALPHA on {TUNING_DATASET}  (Option B)")
print(f"  alphas         = {ALPHAS}")
print(f"  ensemble size  = {ENSEMBLE_SIZE}")
print("=" * 70)

# ── Time grid ────────────────────────────────────────────────────────────────
time_grid = np.arange(cfg.DT, cfg.T_END + cfg.DT, cfg.DT)
step_len = np.full(len(time_grid), cfg.DT)
N_model = int(cfg.T_END / cfg.DT)
T_model = np.linspace(0, cfg.T_END, N_model + 1)
dt_kf = cfg.DT
N_kf = len(T_model) - 1

# ── Fixed matrices ───────────────────────────────────────────────────────────
var_meas = np.array(list(cfg.MEASUREMENT_NOISE_VAR.values()))
R = np.diag(var_meas[:cfg.MEAS_NUM])
H = np.hstack((np.eye(cfg.MEAS_NUM),
               np.zeros((cfg.MEAS_NUM, cfg.STATE_NUM - cfg.MEAS_NUM))))

process_noise_cv = {cfg.STATE_NAMES.index(s): cv
                    for s, cv in cfg.PROCESS_NOISE_CV.items()}
no_update_indices = {cfg.STATE_NAMES.index(s) for s in cfg.NO_UPDATE_STATES}
clip_indices = {cfg.STATE_NAMES.index(s) for s in cfg.CLIP_STATES}

# Fixed climatological scale vector (per state); 0 for measured / unset states.
scale_vec = np.zeros(cfg.STATE_NUM)
for s, sc in cfg.PROCESS_NOISE_SCALE.items():
    scale_vec[cfg.STATE_NAMES.index(s)] = sc

# ── Shared data for P4 ───────────────────────────────────────────────────────
ALL = select_datasets("P1", "P2", "P3", "P4")
volume_results = compute_volume_results(ALL, cfg.INITIAL_VOLUMES, build_schedule, step_len)

data = load_dataset(TUNING_DATASET)
_, state_init = get_initial_condition(data["met_df"], data["nsd_df"])
state_init_by_dataset = {TUNING_DATASET: state_init}

T_meas = np.array(cfg.T_MEAS_FIXED)

np.random.seed(args.seed)
set_meas_ens = generate_measurement_ensembles(
    select_datasets(TUNING_DATASET), load_dataset,
    cfg.MEAS_NUM, ENSEMBLE_SIZE, var_meas,
)[TUNING_DATASET]

# ── NSD measurements (withheld from point-estimate tuning) ───────────────────
NSD_meas = (pd.DataFrame(data["NSD_meas"]).apply(pd.to_numeric, errors="coerce")
            .to_numpy())
n_nsd = NSD_meas.shape[1]
nsd_state_idx = list(range(cfg.STATE_NUM - n_nsd, cfg.STATE_NUM))
nsd_names = [cfg.STATE_NAMES[i] for i in nsd_state_idx]

# map each measurement time to its index on the KF grid
meas_grid_idx = [int(round(t / dt_kf)) for t in T_meas]

# P0: measured states from R, unmeasured from per-step Q (rebuilt per alpha)
P0_meas = np.array([cfg.MEASUREMENT_NOISE_VAR.get(s, 0.0) for s in cfg.STATE_NAMES])


def evaluate_alpha(alpha):
    """Run one EnKF pass on P4 with this alpha; return NSD spread diagnostics."""
    var_model = (alpha * scale_vec) ** 2
    Q = np.diag(var_model)

    P0_diag = var_model.copy()
    for i in range(cfg.MEAS_NUM):
        P0_diag[i] = P0_meas[i]
    P0 = np.diag(P0_diag)

    np.random.seed(args.seed)  # same ensemble draw across alphas for comparability
    _, std_traj, mean_traj = run_enkf_single_with_ensemble_diagnostics(
        dataset_name=TUNING_DATASET,
        load_dataset_fn=load_dataset, build_schedule_fn=build_schedule,
        state_init=state_init, volume_results=volume_results,
        set_meas_ens=set_meas_ens, T_meas=T_meas,
        state_num=cfg.STATE_NUM, meas_num=cfg.MEAS_NUM,
        ensemble_size=ENSEMBLE_SIZE,
        Q=Q, R=R, H=H, dt_kf=dt_kf, N_kf=N_kf,
        P0=P0, process_noise_cv=process_noise_cv,
        no_update_indices=no_update_indices, clip_indices=clip_indices,
    )

    cov, sskill, rmse_all, meanstd_all = {}, {}, {}, {}
    for col, si in enumerate(nsd_state_idx):
        meas = NSD_meas[:, col]
        m = mean_traj[meas_grid_idx, si]
        s = std_traj[meas_grid_idx, si]
        valid = ~np.isnan(meas) & (s > 0)
        if valid.sum() == 0:
            cov[nsd_names[col]] = np.nan; sskill[nsd_names[col]] = np.nan
            continue
        within = np.abs(meas[valid] - m[valid]) <= 2.0 * s[valid]
        cov[nsd_names[col]] = 100.0 * within.mean()
        rmse = np.sqrt(np.mean((meas[valid] - m[valid]) ** 2))
        mean_std = np.mean(s[valid])
        rmse_all[nsd_names[col]] = rmse
        meanstd_all[nsd_names[col]] = mean_std
        sskill[nsd_names[col]] = mean_std / rmse if rmse > 0 else np.nan
    return cov, sskill, rmse_all, meanstd_all


# ── Sweep ────────────────────────────────────────────────────────────────────
results = {}
for a in ALPHAS:
    print(f"\n-- alpha = {a:g} --", flush=True)
    results[a] = evaluate_alpha(a)
    cov, sskill, _, _ = results[a]
    mean_cov = np.nanmean(list(cov.values()))
    med_ss = np.nanmedian(list(sskill.values()))
    print(f"   mean NSD 2-sigma coverage = {mean_cov:5.1f}%   "
          f"median spread-skill = {med_ss:.2f}", flush=True)

# ── Summary tables ───────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("2-sigma coverage of NSD measurements (%)  [target ~95]")
print("=" * 70)
hdr = f"{'alpha':>7s} | " + " | ".join(f"{n[:9]:>9s}" for n in nsd_names) + " |  MEAN"
print(hdr); print("-" * len(hdr))
for a in ALPHAS:
    cov = results[a][0]
    row = f"{a:>7g} | " + " | ".join(f"{cov[n]:9.0f}" for n in nsd_names)
    row += f" | {np.nanmean(list(cov.values())):5.0f}"
    print(row)

print("\n" + "=" * 70)
print("Spread-skill ratio (mean ensemble std / RMSE)  [target ~1.0]")
print("=" * 70)
print(hdr); print("-" * len(hdr))
for a in ALPHAS:
    ss = results[a][1]
    row = f"{a:>7g} | " + " | ".join(f"{ss[n]:9.2f}" for n in nsd_names)
    row += f" | {np.nanmedian(list(ss.values())):5.2f}"
    print(row)

# ── Recommendation ───────────────────────────────────────────────────────────
scores = {a: abs(np.nanmean(list(results[a][0].values())) - 95.0) for a in ALPHAS}
best = min(scores, key=scores.get)
print("\n" + "=" * 70)
print(f"Recommended alpha (mean NSD coverage closest to 95%): {best:g}")
print(f"  mean coverage      = {np.nanmean(list(results[best][0].values())):.1f}%")
print(f"  median spread-skill= {np.nanmedian(list(results[best][1].values())):.2f}")
print("=" * 70)
print("\nNote: states whose OPEN-LOOP model is nearly flat (CMP-Neu5Ac, GDP-Man)")
print("have a tiny climatological scale, so their bands stay narrow regardless of")
print("alpha -- an honest model-fidelity limitation, not a tuning failure.")
