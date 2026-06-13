"""
01_run_enkf.py
==============
Main pipeline: load data, simulate, run EnKF, compute RMSE, and generate plots.

If results already exist (individual pkl files or the legacy results_all.pkl),
the computation is skipped and figures are produced directly.

Usage:
    # Use defaults from config.py
    poetry run python scripts/01_run_enkf.py

    # Override per-run parameters
    poetry run python scripts/01_run_enkf.py --run run_v2 --kq 1e-4 --n-runs 1
    poetry run python scripts/01_run_enkf.py --run run_v3 --kq 5e-5 --ensemble-size 200

Outputs:
    results/{RUN_NAME}/01_run_enkf/pkl/
    results/{RUN_NAME}/01_run_enkf/figures/
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

import nsd_enkf.config as cfg
from nsd_enkf.data_loader import (
    select_datasets, load_dataset, get_initial_condition, build_schedule,
)
from nsd_enkf.model import compute_volume_results, simulate_dataset
from nsd_enkf.enkf import run_enkf_multi_dataset
from nsd_enkf.analysis import generate_measurement_ensembles, compute_rmse_table
from nsd_enkf.io_utils import (
    set_dirs, has_results, has_legacy_results,
    load_legacy_results, save_pkl, load_pkl, fig_path,
)
from nsd_enkf import plotting as pl

# ── CLI overrides ─────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Run EnKF soft sensing pipeline")
parser.add_argument("--run",            default=cfg.RUN_NAME,       help=f"Run name (default: {cfg.RUN_NAME})")
parser.add_argument("--kq",             default=cfg.KQ,       type=float, help=f"Process noise scale (default: {cfg.KQ:.1e})")
parser.add_argument("--n-runs",         default=cfg.N_RUNS,   type=int,   help=f"Number of independent EnKF runs (default: {cfg.N_RUNS})")
parser.add_argument("--ensemble-size",  default=cfg.ENSEMBLE_SIZE, type=int, help=f"Ensemble size (default: {cfg.ENSEMBLE_SIZE})")
args = parser.parse_args()

RUN_NAME      = args.run
KQ            = args.kq
N_RUNS        = args.n_runs
ENSEMBLE_SIZE = args.ensemble_size

# ── Derived paths ─────────────────────────────────────────────────────────────
RESULTS_DIR = cfg.PROJECT_ROOT / "results" / RUN_NAME
S01_PKL = RESULTS_DIR / "01_run_enkf" / "pkl"
S01_FIG = RESULTS_DIR / "01_run_enkf" / "figures"
set_dirs(S01_PKL, S01_FIG)

LEGACY_DIR = cfg.PROJECT_ROOT / "results"

print("=" * 60)
print(f"Step 1: EnKF Soft Sensing Pipeline  [{RUN_NAME}]")
print(f"  KQ={KQ:.1e}  N_RUNS={N_RUNS}  ENSEMBLE_SIZE={ENSEMBLE_SIZE}")
print("=" * 60)

# ── Load datasets ─────────────────────────────────────────────────────────────
DATASETS = select_datasets("P1", "P2", "P3", "P4")

# ── Time grid ─────────────────────────────────────────────────────────────────
time_grid = np.arange(cfg.DT, cfg.T_END + cfg.DT, cfg.DT)
step_len = np.full(len(time_grid), cfg.DT)

dt_model = cfg.DT
N_model = int(cfg.T_END / dt_model)
T_model = np.linspace(0, cfg.T_END, N_model + 1)

# ── Noise matrices (uses CLI KQ) ─────────────────────────────────────────────
var_model = np.array(list(cfg.PROCESS_NOISE_VAR.values()))
var_meas = np.array(list(cfg.MEASUREMENT_NOISE_VAR.values()))

P = np.diag(var_model)
Q = KQ * P

A = np.identity(cfg.MEAS_NUM)
B = np.zeros((cfg.MEAS_NUM, cfg.STATE_NUM - cfg.MEAS_NUM))
H = np.hstack((A, B))
R = np.diag(var_meas[:cfg.MEAS_NUM])

np.random.seed(42)

T_kf = T_model
dt_kf = dt_model
N_kf = len(T_kf) - 1

# ── Load saved results or run from scratch ───────────────────────────────────
LOAD_FROM_PKL = (S01_PKL / "enkf_results_by_dataset.pkl").exists()
LOAD_FROM_LEGACY = (not LOAD_FROM_PKL) and RUN_NAME == "run_v1" and has_legacy_results(LEGACY_DIR)

if LOAD_FROM_PKL:
    print("\nLoading from individual pkl files ...")
    volume_results          = load_pkl('volume_results.pkl')
    state_init_by_dataset   = load_pkl('state_init_by_dataset.pkl')
    set_model_by_dataset    = load_pkl('set_model_by_dataset.pkl')
    T_meas_by_dataset       = load_pkl('T_meas_by_dataset.pkl')
    set_meas_ens_by_dataset = load_pkl('set_meas_ens_by_dataset.pkl')
    enkf_results_by_dataset = load_pkl('enkf_results_by_dataset.pkl')
    enkf_runs_by_dataset    = load_pkl('enkf_runs_by_dataset.pkl')
    rmse_df                 = load_pkl('rmse_df.pkl')

elif LOAD_FROM_LEGACY:
    print("\nLoading from legacy results_all.pkl ...")
    bundle = load_legacy_results(LEGACY_DIR)

    T_model                 = bundle['T_model']
    T_kf                    = bundle['T_kf']
    T_meas_by_dataset       = bundle['T_meas_by_dataset']
    set_model_by_dataset    = bundle['set_model_by_dataset']
    enkf_results_by_dataset = bundle['enkf_results_by_dataset']
    enkf_runs_by_dataset    = bundle['enkf_runs_by_dataset']
    volume_results          = bundle['volume_results_P']
    state_init_by_dataset   = bundle['state_init_by_dataset']
    set_meas_ens_by_dataset = bundle['set_meas_ens_by_dataset']
    rmse_df                 = bundle['rmse_df_all']

    print("\nSplitting legacy bundle into individual pkl files ...")
    save_pkl(volume_results,          'volume_results.pkl')
    save_pkl(state_init_by_dataset,   'state_init_by_dataset.pkl')
    save_pkl(set_model_by_dataset,    'set_model_by_dataset.pkl')
    save_pkl(T_meas_by_dataset,       'T_meas_by_dataset.pkl')
    save_pkl(set_meas_ens_by_dataset, 'set_meas_ens_by_dataset.pkl')
    save_pkl(enkf_results_by_dataset, 'enkf_results_by_dataset.pkl')
    save_pkl(enkf_runs_by_dataset,    'enkf_runs_by_dataset.pkl')
    save_pkl(rmse_df,                 'rmse_df.pkl')
    save_pkl({
        "RUN_NAME": RUN_NAME, "KQ": KQ, "N_RUNS": N_RUNS,
        "ENSEMBLE_SIZE": ENSEMBLE_SIZE,
    }, "run_config.pkl")

else:
    print("\nNo existing results — running from scratch ...")

    print("\nBuilding initial conditions ...")
    init_cond_by_dataset = {}
    state_init_by_dataset = {}
    for name in DATASETS:
        data = load_dataset(name)
        init_cond, state_init = get_initial_condition(data["met_df"], data["nsd_df"])
        init_cond_by_dataset[name] = init_cond
        state_init_by_dataset[name] = state_init

    print("\nComputing volume profiles ...")
    volume_results = compute_volume_results(DATASETS, cfg.INITIAL_VOLUMES, build_schedule, step_len)
    save_pkl(volume_results, 'volume_results.pkl')

    print("\nRunning nominal model simulation ...")
    set_model_by_dataset = {}
    for name in DATASETS:
        Fin, Fout, Gal_feed, Urd_feed = build_schedule(name)
        V_traj = volume_results[name][1:]
        traj = simulate_dataset(
            state_init_by_dataset[name], Fin, Fout, Gal_feed, Urd_feed,
            V_traj, time_grid, step_len, name=name,
        )
        set_model_by_dataset[name] = np.vstack([state_init_by_dataset[name], traj])
    save_pkl(set_model_by_dataset, 'set_model_by_dataset.pkl')

    print("\nPreparing measurement timing ...")
    T_meas_by_dataset = {}
    interval = int(24.0 / dt_model)
    for name in DATASETS:
        data = load_dataset(name)
        N_meas_time = data["set_meas"].shape[0]
        if len(cfg.T_MEAS_FIXED) == N_meas_time:
            T_meas_by_dataset[name] = np.array(cfg.T_MEAS_FIXED)
        else:
            T_index_meas = [i * interval for i in range(N_meas_time)]
            T_meas_by_dataset[name] = T_model[T_index_meas]
    save_pkl(T_meas_by_dataset, 'T_meas_by_dataset.pkl')

    print("\nGenerating measurement ensembles ...")
    set_meas_ens_by_dataset = generate_measurement_ensembles(
        DATASETS, load_dataset, cfg.MEAS_NUM, ENSEMBLE_SIZE, var_meas,
    )
    save_pkl(set_meas_ens_by_dataset, 'set_meas_ens_by_dataset.pkl')

    print(f"\nRunning EnKF ({N_RUNS} run(s), ensemble size {ENSEMBLE_SIZE}) ...")
    enkf_results_by_dataset, enkf_runs_by_dataset, diagnostics_by_dataset = run_enkf_multi_dataset(
        datasets_cfg=DATASETS, load_dataset_fn=load_dataset,
        build_schedule_fn=build_schedule,
        state_init_by_dataset=state_init_by_dataset,
        volume_results=volume_results,
        set_meas_ens_by_dataset=set_meas_ens_by_dataset,
        T_meas_by_dataset=T_meas_by_dataset,
        state_num=cfg.STATE_NUM, meas_num=cfg.MEAS_NUM,
        ensemble_size=ENSEMBLE_SIZE, n_runs=N_RUNS,
        Q=Q, R=R, H=H, dt_kf=dt_kf, N_kf=N_kf,
        save_fn=save_pkl,
    )
    save_pkl(enkf_results_by_dataset, 'enkf_results_by_dataset.pkl')
    save_pkl(enkf_runs_by_dataset,    'enkf_runs_by_dataset.pkl')
    save_pkl(state_init_by_dataset,   'state_init_by_dataset.pkl')
    save_pkl(diagnostics_by_dataset,  'diagnostics_by_dataset.pkl')
    save_pkl({
        "RUN_NAME": RUN_NAME, "KQ": KQ, "N_RUNS": N_RUNS,
        "ENSEMBLE_SIZE": ENSEMBLE_SIZE,
    }, "run_config.pkl")

    print("\nComputing RMSE ...")
    rmse_df = compute_rmse_table(
        DATASETS, load_dataset,
        set_model_by_dataset, enkf_results_by_dataset,
        T_model, T_kf, T_meas_by_dataset,
        cfg.AXIS_NAMES, cfg.STATE_NUM,
    )
    save_pkl(rmse_df, 'rmse_df.pkl')

# ── Print RMSE summary ───────────────────────────────────────────────────────
print("\nRMSE summary:")
print(rmse_df.to_string())

# ── Figures ───────────────────────────────────────────────────────────────────
dataset_list = list(DATASETS.keys())

print("\n[A] NSD grid plot ...")
pl.plot_nsd_grid(
    set_model_by_dataset, enkf_results_by_dataset, enkf_runs_by_dataset,
    load_dataset, T_model, T_kf, T_meas_by_dataset, cfg.AXIS_NAMES,
    dataset_list=dataset_list,
    save_path=fig_path("all_nsd_with_individual_runs.png"),
)

print("[B] Measured metabolites ...")
pl.plot_measured_metabolites_grid(
    set_model_by_dataset, enkf_results_by_dataset, enkf_runs_by_dataset,
    load_dataset, T_model, T_kf, T_meas_by_dataset,
    cfg.STATE_NAMES, cfg.AXIS_NAMES, cfg.MEAS_NUM,
    dataset_list=dataset_list, dataset_colours=cfg.DATASET_COLOURS,
    save_path=fig_path("measured_metabolites.png"),
)

print("[C] Asparagine soft sensing ...")
pl.plot_asn_soft_sensing(
    set_model_by_dataset, enkf_results_by_dataset, enkf_runs_by_dataset,
    load_dataset, T_model, T_kf, T_meas_by_dataset, cfg.AXIS_NAMES,
    dataset_list=dataset_list,
    save_path=fig_path("all_asn.png"),
)

print("\nStep 1 complete.")
