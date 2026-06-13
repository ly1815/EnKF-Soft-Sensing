"""
04_ensemble_uncertainty_profiles.py
===================================
Draw concentration time series with within-run ensemble (N=100) posterior
±1σ/±2σ uncertainty bands.

Reads pkl from 01_run_enkf only — diagnostics are saved there directly.

Usage:
    poetry run python scripts/04_ensemble_uncertainty_profiles.py
    poetry run python scripts/04_ensemble_uncertainty_profiles.py --run run_v2

Outputs:
    results/{RUN_NAME}/04_ensemble_uncertainty_profiles/figures/
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

import nsd_enkf.config as cfg
from nsd_enkf.data_loader import load_dataset
from nsd_enkf.io_utils import set_dirs, has_results, load_pkl, fig_path
from nsd_enkf import plotting as pl

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Concentration profiles with ensemble bands")
parser.add_argument("--run", default=cfg.RUN_NAME, help=f"Run name (default: {cfg.RUN_NAME})")
args = parser.parse_args()

RUN_NAME = args.run
RESULTS_DIR = cfg.PROJECT_ROOT / "results" / RUN_NAME

S01_PKL = RESULTS_DIR / "01_run_enkf" / "pkl"
S04_FIG = RESULTS_DIR / "04_ensemble_uncertainty_profiles" / "figures"
S04_PKL = RESULTS_DIR / "04_ensemble_uncertainty_profiles" / "pkl"
set_dirs(S04_PKL, S04_FIG)

print("=" * 60)
print(f"Step 4: Concentration profiles with ensemble bands  [{RUN_NAME}]")
print("=" * 60)

if not has_results(S01_PKL):
    print("ERROR: No results from step 01. Run 01_run_enkf.py first.")
    sys.exit(1)

N_model = int(cfg.T_END / cfg.DT)
T_model = np.linspace(0, cfg.T_END, N_model + 1)
T_kf = T_model

print("\nLoading results ...")
set_model_by_dataset = load_pkl('set_model_by_dataset.pkl', subdir=S01_PKL)
T_meas_by_dataset    = load_pkl('T_meas_by_dataset.pkl',    subdir=S01_PKL)

dataset_list = list(set_model_by_dataset.keys())

# Load diagnostics — try bundled file first, fall back to per-dataset files
if (S01_PKL / "diagnostics_by_dataset.pkl").exists():
    diagnostics_by_dataset = load_pkl('diagnostics_by_dataset.pkl', subdir=S01_PKL)
else:
    diagnostics_by_dataset = {}
    for name in dataset_list:
        diagnostics_by_dataset[name] = load_pkl(f'diagnostics_{name}.pkl', subdir=S01_PKL)

print("\n[A] Measured metabolites with posterior ensemble bands ...")
pl.plot_metabolites_with_ensemble_bands(
    set_model_by_dataset, diagnostics_by_dataset,
    load_dataset, T_model, T_kf, T_meas_by_dataset,
    cfg.STATE_NAMES, cfg.AXIS_NAMES, cfg.MEAS_NUM,
    dataset_list=dataset_list, dataset_colours=cfg.DATASET_COLOURS,
    save_path=fig_path("metabolites_posterior_bands.png"),
)

print("[B] NSD states with posterior ensemble bands ...")
pl.plot_nsd_with_ensemble_bands(
    set_model_by_dataset, diagnostics_by_dataset,
    load_dataset, T_model, T_kf, T_meas_by_dataset, cfg.AXIS_NAMES,
    dataset_list=dataset_list,
    save_path=fig_path("nsd_posterior_bands.png"),
)

print("[C] Asn + Glu with posterior ensemble bands ...")
pl.plot_asn_glu_with_ensemble_bands(
    set_model_by_dataset, diagnostics_by_dataset,
    load_dataset, T_model, T_kf, T_meas_by_dataset, cfg.AXIS_NAMES,
    dataset_list=dataset_list,
    save_path=fig_path("asn_glu_posterior_bands.png"),
)

print("\nStep 4 complete.")
