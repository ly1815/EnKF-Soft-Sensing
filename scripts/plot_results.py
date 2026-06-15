"""
plot_results.py
===============
Generate all figures from a completed EnKF run.

Reads pkl files from results/{RUN_NAME}/pkl/ and writes figures to
results/{RUN_NAME}/figures/.

Usage:
    poetry run python scripts/plot_results.py
    poetry run python scripts/plot_results.py --run tuned_v1
    poetry run python scripts/plot_results.py --run tuned_v1 --only trajectories
    poetry run python scripts/plot_results.py --run tuned_v1 --only uncertainty
    poetry run python scripts/plot_results.py --run tuned_v1 --only diagnostics

Figure groups:
    trajectories  — mean trajectory plots (metabolites, Asn, NSDs)
    uncertainty   — ±1σ/±2σ bands across independent runs
    diagnostics   — ensemble std, snapshots, forecast-vs-analysis spread
    posterior     — within-run posterior ensemble ±1σ/±2σ bands

Output structure:
    results/{RUN_NAME}/figures/
        measured_metabolites.png
        asparagine.png
        nsd_grid.png
        nsd_grid_with_runs.png
        uncertainty_metabolites.png
        uncertainty_nsd.png
        uncertainty_asn_glu.png
        spread_evolution.png
        ensemble_std_{P1..P4}.png
        ensemble_snapshots_{P1..P4}.png
        forecast_vs_analysis_{P1..P4}.png
        posterior_metabolites.png
        posterior_nsd.png
        posterior_asn_glu.png
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

import nsd_enkf.config as cfg
from nsd_enkf.data_loader import load_dataset
from nsd_enkf.io_utils import set_dirs, has_results, load_pkl, load_per_dataset, fig_path
from nsd_enkf import plotting as pl

# ── CLI ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Generate all figures from EnKF results")
parser.add_argument("--run",  default=cfg.RUN_NAME)
parser.add_argument("--only", default=None,
                    choices=["trajectories", "uncertainty", "diagnostics", "posterior"],
                    help="Generate only one figure group")
args = parser.parse_args()

RUN_NAME = args.run
RESULTS_DIR = cfg.PROJECT_ROOT / "results" / RUN_NAME
PKL_DIR = RESULTS_DIR / "pkl"
FIG_DIR = RESULTS_DIR / "figures"

# Support both new flat layout and old nested layout
if not PKL_DIR.exists():
    PKL_DIR_ALT = RESULTS_DIR / "01_run_enkf" / "pkl"
    if PKL_DIR_ALT.exists():
        PKL_DIR = PKL_DIR_ALT
        print(f"Using legacy pkl path: {PKL_DIR}")

set_dirs(PKL_DIR, FIG_DIR)

print("=" * 60)
print(f"Plotting  [{RUN_NAME}]")
if args.only:
    print(f"  Group: {args.only}")
print(f"  Input:  {PKL_DIR}")
print(f"  Output: {FIG_DIR}")
print("=" * 60)

if not has_results(PKL_DIR):
    print("ERROR: No results found. Run run_enkf.py first.")
    sys.exit(1)

# ── Load shared data ─────────────────────────────────────────────────────────
N_model = int(cfg.T_END / cfg.DT)
T_model = np.linspace(0, cfg.T_END, N_model + 1)
T_kf = T_model

run_cfg = load_pkl('run_config.pkl', subdir=PKL_DIR)
n_runs = run_cfg.get('N_RUNS', 1)

set_model_by_dataset = load_pkl('set_model_by_dataset.pkl', subdir=PKL_DIR)
T_meas_by_dataset    = load_pkl('T_meas_by_dataset.pkl',    subdir=PKL_DIR)
dataset_list = list(set_model_by_dataset.keys())

# Lazy loaders — only load heavy data when needed
_enkf_results = None
_enkf_runs = None
_diagnostics = None

def get_enkf_results():
    global _enkf_results
    if _enkf_results is None:
        _enkf_results = load_per_dataset("enkf_results", dataset_list, n_runs, PKL_DIR)
    return _enkf_results

def get_enkf_runs():
    global _enkf_runs
    if _enkf_runs is None:
        _enkf_runs = load_per_dataset("enkf_traj", dataset_list, n_runs, PKL_DIR)
    return _enkf_runs

def get_diagnostics():
    global _diagnostics
    if _diagnostics is None:
        _diagnostics = load_per_dataset("diagnostics", dataset_list, n_runs, PKL_DIR)
    return _diagnostics

do_all = args.only is None

# ── Trajectories: mean trajectory plots ──────────────────────────────────────
if do_all or args.only == "trajectories":
    print("\n── Trajectory plots ──")

    print("  Measured metabolites ...", flush=True)
    pl.plot_measured_metabolites_grid(
        set_model_by_dataset, get_enkf_results(), get_enkf_runs(), load_dataset,
        T_model, T_kf, T_meas_by_dataset, cfg.STATE_NAMES, cfg.AXIS_NAMES, cfg.MEAS_NUM,
        dataset_list=dataset_list, dataset_colours=cfg.DATASET_COLOURS,
        save_path=fig_path("measured_metabolites.png"),
    )

    print("  Asparagine ...", flush=True)
    pl.plot_asn_soft_sensing(
        set_model_by_dataset, get_enkf_results(), get_enkf_runs(), load_dataset,
        T_model, T_kf, T_meas_by_dataset, cfg.AXIS_NAMES,
        dataset_list=dataset_list,
        save_path=fig_path("asparagine.png"),
    )

    print("  NSD grid ...", flush=True)
    pl.plot_nsd_grid(
        set_model_by_dataset, get_enkf_results(), get_enkf_runs(), load_dataset,
        T_model, T_kf, T_meas_by_dataset, cfg.AXIS_NAMES,
        dataset_list=dataset_list, plot_individual_runs=False,
        save_path=fig_path("nsd_grid.png"),
    )

    print("  NSD grid (with individual runs) ...", flush=True)
    pl.plot_nsd_grid(
        set_model_by_dataset, get_enkf_results(), get_enkf_runs(), load_dataset,
        T_model, T_kf, T_meas_by_dataset, cfg.AXIS_NAMES,
        dataset_list=dataset_list, plot_individual_runs=True,
        save_path=fig_path("nsd_grid_with_runs.png"),
    )

# ── Uncertainty: ±σ bands across independent runs ────────────────────────────
if do_all or args.only == "uncertainty":
    print("\n── Uncertainty band plots ──")

    print("  Measured metabolites ...", flush=True)
    pl.plot_measured_metabolites_uncertainty(
        set_model_by_dataset, get_enkf_results(), get_enkf_runs(),
        load_dataset, T_model, T_kf, T_meas_by_dataset,
        cfg.STATE_NAMES, cfg.AXIS_NAMES, cfg.MEAS_NUM,
        dataset_list=dataset_list, dataset_colours=cfg.DATASET_COLOURS,
        save_path=fig_path("uncertainty_metabolites.png"),
    )

    print("  NSD states ...", flush=True)
    pl.plot_nsd_uncertainty(
        set_model_by_dataset, get_enkf_results(), get_enkf_runs(),
        load_dataset, T_model, T_kf, T_meas_by_dataset, cfg.AXIS_NAMES,
        dataset_list=dataset_list,
        save_path=fig_path("uncertainty_nsd.png"),
    )

    print("  Asn + Glu ...", flush=True)
    pl.plot_unmeasured_extracellular_uncertainty(
        set_model_by_dataset, get_enkf_results(), get_enkf_runs(),
        load_dataset, T_model, T_kf, T_meas_by_dataset, cfg.AXIS_NAMES,
        state_indices=(8, 9), state_labels=("Asn", "Glu"),
        dataset_list=dataset_list,
        save_path=fig_path("uncertainty_asn_glu.png"),
    )

    print("  Ensemble spread evolution ...", flush=True)
    pl.plot_ensemble_spread_evolution(
        get_enkf_runs(), T_kf, cfg.STATE_NAMES, cfg.AXIS_NAMES,
        state_indices=[0, 4, 8, 13],
        dataset_list=dataset_list,
        save_path=fig_path("spread_evolution.png"),
    )

# ── Diagnostics: per-dataset ensemble health ─────────────────────────────────
if do_all or args.only == "diagnostics":
    print("\n── Ensemble diagnostics ──")

    diagnostics = get_diagnostics()
    for ds_name in dataset_list:
        print(f"  {ds_name} ...", flush=True)
        diag_entry = diagnostics[ds_name]
        diag_first = diag_entry[0] if isinstance(diag_entry, list) else diag_entry
        T_meas = T_meas_by_dataset[ds_name]

        pl.plot_ensemble_std_trajectory(
            diag_first["std_trajectory"], T_kf, cfg.STATE_NAMES, cfg.AXIS_NAMES,
            T_meas=T_meas, dataset_name=ds_name,
            save_path=fig_path(f"ensemble_std_{ds_name}.png"),
        )

        if "ensemble_at_updates" in diag_first:
            pl.plot_ensemble_snapshots(
                diag_first["ensemble_at_updates"], cfg.STATE_NAMES, cfg.AXIS_NAMES,
                state_indices=[0, 4, 8, 13], dataset_name=ds_name,
                save_path=fig_path(f"ensemble_snapshots_{ds_name}.png"),
            )

            pl.plot_forecast_vs_analysis_spread(
                diag_first["ensemble_at_updates"], cfg.STATE_NAMES, cfg.AXIS_NAMES,
                state_indices=[0, 4, 8, 13], dataset_name=ds_name,
                save_path=fig_path(f"forecast_vs_analysis_{ds_name}.png"),
            )

# ── Posterior: within-run ensemble ±σ bands ──────────────────────────────────
if do_all or args.only == "posterior":
    print("\n── Posterior ensemble band plots ──")

    print("  Measured metabolites ...", flush=True)
    pl.plot_metabolites_with_ensemble_bands(
        set_model_by_dataset, get_diagnostics(),
        load_dataset, T_model, T_kf, T_meas_by_dataset,
        cfg.STATE_NAMES, cfg.AXIS_NAMES, cfg.MEAS_NUM,
        dataset_list=dataset_list, dataset_colours=cfg.DATASET_COLOURS,
        save_path=fig_path("posterior_metabolites.png"),
    )

    print("  NSD states ...", flush=True)
    pl.plot_nsd_with_ensemble_bands(
        set_model_by_dataset, get_diagnostics(),
        load_dataset, T_model, T_kf, T_meas_by_dataset, cfg.AXIS_NAMES,
        dataset_list=dataset_list,
        save_path=fig_path("posterior_nsd.png"),
    )

    print("  Asn + Glu ...", flush=True)
    pl.plot_asn_glu_with_ensemble_bands(
        set_model_by_dataset, get_diagnostics(),
        load_dataset, T_model, T_kf, T_meas_by_dataset, cfg.AXIS_NAMES,
        dataset_list=dataset_list,
        save_path=fig_path("posterior_asn_glu.png"),
    )

print("\nAll plots complete.")
