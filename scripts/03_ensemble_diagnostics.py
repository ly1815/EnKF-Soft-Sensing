"""
03_ensemble_diagnostics.py
==========================
Generate ensemble diagnostic plots (std trajectory, violin snapshots,
forecast vs analysis spread).

Reads diagnostics from 01_run_enkf — no re-run needed.

Usage:
    poetry run python scripts/03_ensemble_diagnostics.py
    poetry run python scripts/03_ensemble_diagnostics.py --run run_v2

Outputs:
    results/{RUN_NAME}/03_ensemble_diagnostics/figures/
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

import nsd_enkf.config as cfg
from nsd_enkf.io_utils import set_dirs, has_results, load_pkl, fig_path
from nsd_enkf import plotting as pl

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Ensemble diagnostics plots")
parser.add_argument("--run", default=cfg.RUN_NAME, help=f"Run name (default: {cfg.RUN_NAME})")
args = parser.parse_args()

RUN_NAME = args.run
RESULTS_DIR = cfg.PROJECT_ROOT / "results" / RUN_NAME

S01_PKL = RESULTS_DIR / "01_run_enkf" / "pkl"
S03_FIG = RESULTS_DIR / "03_ensemble_diagnostics" / "figures"
S03_PKL = RESULTS_DIR / "03_ensemble_diagnostics" / "pkl"
set_dirs(S03_PKL, S03_FIG)

print("=" * 60)
print(f"Step 3: Ensemble Diagnostics  [{RUN_NAME}]")
print("=" * 60)

if not has_results(S01_PKL):
    print("ERROR: No results from step 01. Run 01_run_enkf.py first.")
    sys.exit(1)

N_model = int(cfg.T_END / cfg.DT)
T_kf = np.linspace(0, cfg.T_END, N_model + 1)

print("\nLoading diagnostics from 01_run_enkf ...")
T_meas_by_dataset = load_pkl('T_meas_by_dataset.pkl', subdir=S01_PKL)

# Load diagnostics — try bundled file first, fall back to per-dataset files
if (S01_PKL / "diagnostics_by_dataset.pkl").exists():
    diagnostics_by_dataset = load_pkl('diagnostics_by_dataset.pkl', subdir=S01_PKL)
else:
    # Per-dataset files saved incrementally during long runs
    set_model = load_pkl('set_model_by_dataset.pkl', subdir=S01_PKL)
    diagnostics_by_dataset = {}
    for name in set_model.keys():
        diagnostics_by_dataset[name] = load_pkl(f'diagnostics_{name}.pkl', subdir=S01_PKL)

dataset_list = list(diagnostics_by_dataset.keys())

for ds_name in dataset_list:
    print(f"\n{'─' * 40}")
    print(f"Dataset: {ds_name}")
    print(f"{'─' * 40}")

    diag_list = diagnostics_by_dataset[ds_name]
    # Support both old format (single dict) and new format (list of dicts)
    if isinstance(diag_list, dict):
        diag_list = [diag_list]
    diag_first = diag_list[0]
    T_meas = T_meas_by_dataset[ds_name]

    print(f"  [A] Ensemble std trajectory (run 1) ...")
    pl.plot_ensemble_std_trajectory(
        diag_first["std_trajectory"], T_kf, cfg.STATE_NAMES, cfg.AXIS_NAMES,
        T_meas=T_meas, dataset_name=ds_name,
        save_path=fig_path(f"ensemble_std_trajectory_{ds_name}.png"),
    )

    print(f"  [B] Ensemble distribution snapshots (run 1) ...")
    pl.plot_ensemble_snapshots(
        diag_first["ensemble_at_updates"], cfg.STATE_NAMES, cfg.AXIS_NAMES,
        state_indices=[0, 4, 8, 13], dataset_name=ds_name,
        save_path=fig_path(f"ensemble_snapshots_{ds_name}.png"),
    )

    print(f"  [C] Forecast vs analysis spread (run 1) ...")
    pl.plot_forecast_vs_analysis_spread(
        diag_first["ensemble_at_updates"], cfg.STATE_NAMES, cfg.AXIS_NAMES,
        state_indices=[0, 4, 8, 13], dataset_name=ds_name,
        save_path=fig_path(f"forecast_vs_analysis_{ds_name}.png"),
    )

print("\nStep 3 complete.")
