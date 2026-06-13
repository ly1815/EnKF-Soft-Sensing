"""
02_uncertainty_figures.py
========================
Generate uncertainty band figures (±1σ, ±2σ across independent runs).

Requires results from 01_run_enkf.py for the same --run.

Usage:
    poetry run python scripts/02_uncertainty_figures.py
    poetry run python scripts/02_uncertainty_figures.py --run run_v2

Outputs:
    results/{RUN_NAME}/02_uncertainty/figures/
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
parser = argparse.ArgumentParser(description="Across-run uncertainty figures")
parser.add_argument("--run", default=cfg.RUN_NAME, help=f"Run name (default: {cfg.RUN_NAME})")
args = parser.parse_args()

RUN_NAME = args.run
RESULTS_DIR = cfg.PROJECT_ROOT / "results" / RUN_NAME

S01_PKL = RESULTS_DIR / "01_run_enkf" / "pkl"
S02_FIG = RESULTS_DIR / "02_uncertainty" / "figures"
S02_PKL = RESULTS_DIR / "02_uncertainty" / "pkl"
set_dirs(S02_PKL, S02_FIG)

print("=" * 60)
print(f"Step 2: Uncertainty Band Figures  [{RUN_NAME}]")
print("=" * 60)

if not has_results(S01_PKL):
    print("ERROR: No results from step 01. Run 01_run_enkf.py first.")
    sys.exit(1)

N_model = int(cfg.T_END / cfg.DT)
T_model = np.linspace(0, cfg.T_END, N_model + 1)
T_kf = T_model

print("\nLoading results from 01_run_enkf ...")
set_model_by_dataset    = load_pkl('set_model_by_dataset.pkl',    subdir=S01_PKL)
enkf_results_by_dataset = load_pkl('enkf_results_by_dataset.pkl', subdir=S01_PKL)
enkf_runs_by_dataset    = load_pkl('enkf_runs_by_dataset.pkl',    subdir=S01_PKL)
T_meas_by_dataset       = load_pkl('T_meas_by_dataset.pkl',       subdir=S01_PKL)

dataset_list = list(enkf_runs_by_dataset.keys())

print("\n[A] Measured metabolites with uncertainty bands ...")
pl.plot_measured_metabolites_uncertainty(
    set_model_by_dataset, enkf_results_by_dataset, enkf_runs_by_dataset,
    load_dataset, T_model, T_kf, T_meas_by_dataset,
    cfg.STATE_NAMES, cfg.AXIS_NAMES, cfg.MEAS_NUM,
    dataset_list=dataset_list, dataset_colours=cfg.DATASET_COLOURS,
    save_path=fig_path("measured_metabolites_uncertainty.png"),
)

print("[B] NSD states with uncertainty bands ...")
pl.plot_nsd_uncertainty(
    set_model_by_dataset, enkf_results_by_dataset, enkf_runs_by_dataset,
    load_dataset, T_model, T_kf, T_meas_by_dataset, cfg.AXIS_NAMES,
    dataset_list=dataset_list,
    save_path=fig_path("nsd_uncertainty.png"),
)

print("[C] Unmeasured extracellular (Asn, Glu) with uncertainty bands ...")
pl.plot_unmeasured_extracellular_uncertainty(
    set_model_by_dataset, enkf_results_by_dataset, enkf_runs_by_dataset,
    load_dataset, T_model, T_kf, T_meas_by_dataset, cfg.AXIS_NAMES,
    state_indices=(8, 9), state_labels=("Asn", "Glu"),
    dataset_list=dataset_list,
    save_path=fig_path("asn_glu_uncertainty.png"),
)

print("[D] Ensemble spread evolution ...")
pl.plot_ensemble_spread_evolution(
    enkf_runs_by_dataset, T_kf, cfg.STATE_NAMES, cfg.AXIS_NAMES,
    state_indices=[0, 4, 8, 13],
    dataset_list=dataset_list,
    save_path=fig_path("ensemble_spread_evolution.png"),
)

print("\nStep 2 complete.")
