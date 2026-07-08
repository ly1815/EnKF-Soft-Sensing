"""
tune_alpha.py
=============
Calibrate the single universal process-noise scalar ALPHA (Option B) on P4 by
NSD NRMSE, then cross-validate the selected value on P1-P3.

Unmeasured states use additive noise Q_ii = (alpha * scale_i)^2, where scale_i is
the fixed median magnitude from config (PROCESS_NOISE_SCALE) and alpha is the one
tunable knob shared by all unmeasured states. Measured states keep their per-state
multiplicative CV noise (PROCESS_NOISE_CV), unaffected by alpha.

Selection metric: mean NRMSE over the 7 NSDs on P4, where
    NRMSE_i = RMSE_i / mean(NSD_meas_i)          (mean-normalised, dimensionless)
so states of very different magnitude are comparable. The alpha minimising mean
P4 NRMSE is selected, then re-evaluated on P1-P3 (cross-validation). Coverage and
spread-skill are reported as secondary uncertainty diagnostics.

Usage:
    ./.venv/Scripts/python.exe scripts/tune_alpha.py
    ./.venv/Scripts/python.exe scripts/tune_alpha.py --alphas 0.02,0.03,0.046,0.06,0.1,0.15
    ./.venv/Scripts/python.exe scripts/tune_alpha.py --tuning-dataset P4 --validate P1,P2,P3
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
from nsd_enkf.model import compute_volume_results
from nsd_enkf.analysis import generate_measurement_ensembles
from nsd_enkf.enkf import run_enkf_single_with_ensemble_diagnostics

# ── CLI ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Calibrate universal ALPHA on P4 (NRMSE), cross-validate")
parser.add_argument("--alphas", default="0.02,0.03,0.046,0.06,0.1,0.15")
parser.add_argument("--tuning-dataset", default="P4")
parser.add_argument("--validate", default="P1,P2,P3")
parser.add_argument("--ensemble-size", default=cfg.ENSEMBLE_SIZE, type=int)
parser.add_argument("--seed", default=42, type=int)
args = parser.parse_args()

ALPHAS = [float(a) for a in args.alphas.split(",")]
TUNE_DS = args.tuning_dataset
VAL_DS = [s for s in args.validate.split(",") if s]
ENS = args.ensemble_size

print("=" * 70)
print(f"Calibrate ALPHA on {TUNE_DS} by NSD NRMSE; cross-validate on {VAL_DS}")
print(f"  alphas = {ALPHAS} | ensemble size = {ENS}")
print("=" * 70)

# ── Grids / fixed matrices ───────────────────────────────────────────────────
time_grid = np.arange(cfg.DT, cfg.T_END + cfg.DT, cfg.DT)
step_len = np.full(len(time_grid), cfg.DT)
N_model = int(cfg.T_END / cfg.DT)
T_model = np.linspace(0, cfg.T_END, N_model + 1)
dt_kf = cfg.DT
N_kf = len(T_model) - 1

var_meas = np.array(list(cfg.MEASUREMENT_NOISE_VAR.values()))
R = np.diag(var_meas[:cfg.MEAS_NUM])
H = np.hstack((np.eye(cfg.MEAS_NUM),
               np.zeros((cfg.MEAS_NUM, cfg.STATE_NUM - cfg.MEAS_NUM))))
process_noise_cv = {cfg.STATE_NAMES.index(s): cv for s, cv in cfg.PROCESS_NOISE_CV.items()}
no_update_indices = {cfg.STATE_NAMES.index(s) for s in cfg.NO_UPDATE_STATES}
clip_indices = {cfg.STATE_NAMES.index(s) for s in cfg.CLIP_STATES}

scale_vec = np.zeros(cfg.STATE_NUM)
for s, sc in cfg.PROCESS_NOISE_SCALE.items():
    scale_vec[cfg.STATE_NAMES.index(s)] = sc

# Two-stage alpha: swept alpha applies to NSDs; observable Asn/Glu pinned at ALPHA_OBS.
ALPHA_OBS = getattr(cfg, "PROCESS_NOISE_ALPHA_OBS", 0.0)
obs_idx = [cfg.STATE_NAMES.index(s) for s in getattr(cfg, "ALPHA_OBS_STATES", [])]

def alpha_var(alpha_nsd):
    a = np.full(cfg.STATE_NUM, float(alpha_nsd))
    for i in obs_idx:
        a[i] = ALPHA_OBS
    return (a * scale_vec) ** 2

n_nsd = 7
nsd_state_idx = list(range(cfg.STATE_NUM - n_nsd, cfg.STATE_NUM))
nsd_names = [cfg.STATE_NAMES[i] for i in nsd_state_idx]
T_meas = np.array(cfg.T_MEAS_FIXED)
meas_grid_idx = [min(int(round(t / dt_kf)) + 1, N_kf) for t in T_meas]  # post-update index

ALL_DS = sorted(set([TUNE_DS] + VAL_DS))
volume_results = compute_volume_results(select_datasets(*ALL_DS), cfg.INITIAL_VOLUMES,
                                        build_schedule, step_len)

# Per-dataset static inputs (state_init, meas ensemble, NSD measurements)
_ds_cache = {}
def ds_data(name):
    if name not in _ds_cache:
        d = load_dataset(name)
        _, x0 = get_initial_condition(d["met_df"], d["nsd_df"])
        np.random.seed(args.seed)
        mens = generate_measurement_ensembles(select_datasets(name), load_dataset,
                                              cfg.MEAS_NUM, ENS, var_meas)[name]
        nsd = pd.DataFrame(d["NSD_meas"]).apply(pd.to_numeric, errors="coerce").to_numpy()
        _ds_cache[name] = dict(x0=x0, mens=mens, nsd=nsd)
    return _ds_cache[name]

P0_meas = np.array([cfg.MEASUREMENT_NOISE_VAR.get(s, 0.0) for s in cfg.STATE_NAMES])


def evaluate(name, alpha):
    """Run one EnKF pass; return per-NSD rmse, nrmse, coverage, spread-skill."""
    dd = ds_data(name)
    var_model = alpha_var(alpha)
    P0_diag = var_model.copy()
    P0_diag[:cfg.MEAS_NUM] = P0_meas[:cfg.MEAS_NUM]

    np.random.seed(args.seed)  # same ensemble draw across alphas/datasets
    _, std_traj, mean_traj = run_enkf_single_with_ensemble_diagnostics(
        dataset_name=name, load_dataset_fn=load_dataset, build_schedule_fn=build_schedule,
        state_init=dd["x0"], volume_results=volume_results,
        set_meas_ens=dd["mens"], T_meas=T_meas,
        state_num=cfg.STATE_NUM, meas_num=cfg.MEAS_NUM, ensemble_size=ENS,
        Q=np.diag(var_model), R=R, H=H, dt_kf=dt_kf, N_kf=N_kf,
        P0=np.diag(P0_diag), process_noise_cv=process_noise_cv,
        no_update_indices=no_update_indices, clip_indices=clip_indices,
    )
    out = {}
    for col, si in enumerate(nsd_state_idx):
        meas = dd["nsd"][:, col]
        m = mean_traj[meas_grid_idx, si]
        s = std_traj[meas_grid_idx, si]
        valid = ~np.isnan(meas) & (s > 0)
        if valid.sum() == 0:
            out[nsd_names[col]] = dict(rmse=np.nan, nrmse=np.nan, cov=np.nan, ss=np.nan)
            continue
        err = meas[valid] - m[valid]
        rmse = np.sqrt(np.mean(err ** 2))
        norm = np.mean(np.abs(meas[valid])) or 1.0
        cov = 100.0 * np.mean(np.abs(err) <= 2.0 * s[valid])
        ss = np.mean(s[valid]) / rmse if rmse > 0 else np.nan
        out[nsd_names[col]] = dict(rmse=rmse, nrmse=rmse / norm, cov=cov, ss=ss)
    return out


def mean_nrmse(res):
    return np.nanmean([res[n]["nrmse"] for n in nsd_names])


# ── 1) Sweep on tuning dataset ───────────────────────────────────────────────
print(f"\n### Calibration on {TUNE_DS} ###")
tune_res = {}
for a in ALPHAS:
    print(f"  alpha={a:g} ...", flush=True)
    tune_res[a] = evaluate(TUNE_DS, a)

def print_metric(title, key, fmt, res_by_alpha, agg=np.nanmean, aggfmt="{:6.3f}"):
    print("\n" + title)
    hdr = f"{'alpha':>7s} | " + " | ".join(f"{n[:9]:>9s}" for n in nsd_names) + " |   AGG"
    print(hdr); print("-" * len(hdr))
    for a in ALPHAS:
        r = res_by_alpha[a]
        row = f"{a:>7g} | " + " | ".join(fmt.format(r[n][key]) for n in nsd_names)
        row += " | " + aggfmt.format(agg([r[n][key] for n in nsd_names]))
        print(row)

print_metric(f"NRMSE (RMSE / mean meas)  [{TUNE_DS}, SELECTION METRIC — lower better]",
             "nrmse", "{:9.3f}", tune_res)
print_metric(f"2-sigma coverage (%)  [{TUNE_DS}, secondary]", "cov", "{:9.0f}", tune_res,
             aggfmt="{:6.0f}")
print_metric(f"Spread-skill (std/RMSE ~1)  [{TUNE_DS}, secondary]", "ss", "{:9.2f}", tune_res,
             agg=np.nanmedian, aggfmt="{:6.2f}")

best = min(ALPHAS, key=lambda a: mean_nrmse(tune_res[a]))
print("\n" + "=" * 70)
print(f"Selected alpha (min mean NSD NRMSE on {TUNE_DS}): {best:g}   "
      f"(mean NRMSE = {mean_nrmse(tune_res[best]):.3f})")
print("=" * 70)

# ── 2) Cross-validation ──────────────────────────────────────────────────────
if VAL_DS:
    print(f"\n### Cross-validation at alpha={best:g} ###")
    val_res = {TUNE_DS: tune_res[best]}
    for name in VAL_DS:
        print(f"  {name} ...", flush=True)
        val_res[name] = evaluate(name, best)

    print(f"\nNRMSE per dataset at alpha={best:g}  (train={TUNE_DS}, validate={VAL_DS})")
    hdr = f"{'dataset':>8s} | " + " | ".join(f"{n[:9]:>9s}" for n in nsd_names) + " |   MEAN"
    print(hdr); print("-" * len(hdr))
    for name in [TUNE_DS] + VAL_DS:
        r = val_res[name]
        row = f"{name:>8s} | " + " | ".join(f"{r[n]['nrmse']:9.3f}" for n in nsd_names)
        row += f" | {mean_nrmse(r):6.3f}"
        print(row)
    print(f"\nMean NRMSE  train({TUNE_DS})={mean_nrmse(val_res[TUNE_DS]):.3f}  "
          f"validate={np.mean([mean_nrmse(val_res[n]) for n in VAL_DS]):.3f}")

print(f"\nTo set this in config.py:  PROCESS_NOISE_ALPHA = {best:g}")
