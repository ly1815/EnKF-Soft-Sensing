"""
run_enkf.py
===========
Run the EnKF pipeline: prepare shared data, run EnKF, compute RMSE.

All intermediate results are saved to results/{RUN_NAME}/pkl/.
Figures are generated separately by plot_results.py.

Usage:
    poetry run python scripts/run_enkf.py
    poetry run python scripts/run_enkf.py --run tuned_v2 --kq 0.5 --n-runs 5
    poetry run python scripts/run_enkf.py --datasets P4          # single dataset
    poetry run python scripts/run_enkf.py --force                # recompute everything

Output structure:
    results/{RUN_NAME}/
        pkl/
            run_config.pkl              — run parameters
            volume_results.pkl          — bioreactor volume profiles
            state_init_by_dataset.pkl   — initial conditions per dataset
            set_model_by_dataset.pkl    — open-loop model trajectories
            T_meas_by_dataset.pkl       — measurement time vectors
            set_meas_ens_by_dataset.pkl — perturbed measurement ensembles
            enkf_results_{P1..P4}.pkl   — mean EnKF trajectory (across runs)
            enkf_traj_{P1..P4}_run{i}.pkl  — per-run trajectories
            diagnostics_{P1..P4}_run{i}.pkl — per-run ensemble diagnostics
            rmse.pkl                    — RMSE table (DataFrame)
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
from nsd_enkf.io_utils import set_dirs, has_results, save_pkl, load_pkl

# ── CLI ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Run EnKF soft sensing pipeline")
parser.add_argument("--run",            default=cfg.RUN_NAME)
parser.add_argument("--kq",             default=cfg.KQ,            type=float)
parser.add_argument("--n-runs",         default=cfg.N_RUNS,        type=int)
parser.add_argument("--ensemble-size",  default=cfg.ENSEMBLE_SIZE, type=int)
parser.add_argument("--datasets",       default="P1,P2,P3,P4")
parser.add_argument("--force",          action="store_true", help="Recompute even if results exist")
args = parser.parse_args()

RUN_NAME      = args.run
KQ            = args.kq
N_RUNS        = args.n_runs
ENSEMBLE_SIZE = args.ensemble_size
DATASET_NAMES = [s.strip() for s in args.datasets.split(",")]

# ── Paths ────────────────────────────────────────────────────────────────────
RESULTS_DIR = cfg.PROJECT_ROOT / "results" / RUN_NAME
PKL_DIR = RESULTS_DIR / "pkl"
FIG_DIR = RESULTS_DIR / "figures"
set_dirs(PKL_DIR, FIG_DIR)

print("=" * 60)
print(f"EnKF Pipeline  [{RUN_NAME}]")
print(f"  KQ={KQ:.1e}  N_RUNS={N_RUNS}  ENSEMBLE_SIZE={ENSEMBLE_SIZE}")
print(f"  Datasets: {DATASET_NAMES}")
print(f"  Output:   {RESULTS_DIR}")
print("=" * 60)

# ── Time grids ───────────────────────────────────────────────────────────────
time_grid = np.arange(cfg.DT, cfg.T_END + cfg.DT, cfg.DT)
step_len = np.full(len(time_grid), cfg.DT)
N_model = int(cfg.T_END / cfg.DT)
T_model = np.linspace(0, cfg.T_END, N_model + 1)
T_kf = T_model
dt_kf = cfg.DT
N_kf = len(T_kf) - 1

# ── Noise matrices ───────────────────────────────────────────────────────────
var_model = np.array(list(cfg.PROCESS_NOISE_VAR.values()))
var_meas = np.array(list(cfg.MEASUREMENT_NOISE_VAR.values()))

Q = KQ * np.diag(var_model)
R = np.diag(var_meas[:cfg.MEAS_NUM])
H = np.hstack((np.eye(cfg.MEAS_NUM), np.zeros((cfg.MEAS_NUM, cfg.STATE_NUM - cfg.MEAS_NUM))))

# Multiplicative noise: build {state_index: cv} for states with CV-based noise
process_noise_cv = {}
if hasattr(cfg, 'PROCESS_NOISE_CV'):
    for s, cv in cfg.PROCESS_NOISE_CV.items():
        process_noise_cv[cfg.STATE_NAMES.index(s)] = cv

# Localization: states excluded from Kalman update (structurally unobservable)
no_update_indices = set()
if hasattr(cfg, 'NO_UPDATE_STATES'):
    for s in cfg.NO_UPDATE_STATES:
        no_update_indices.add(cfg.STATE_NAMES.index(s))

# Initial ensemble covariance (separate from per-step Q)
P0_diag = var_model.copy()
if hasattr(cfg, 'INITIAL_COV_OVERRIDE'):
    for i, s in enumerate(cfg.STATE_NAMES):
        if s in cfg.INITIAL_COV_OVERRIDE:
            P0_diag[i] = cfg.INITIAL_COV_OVERRIDE[s]
P0 = np.diag(P0_diag)

np.random.seed(42)

# ── Shared data (compute once, reuse across datasets) ────────────────────────
ALL_DATASETS = select_datasets("P1", "P2", "P3", "P4")

if args.force or not (PKL_DIR / "volume_results.pkl").exists():
    print("\nPreparing shared data ...")

    state_init_by_dataset = {}
    for name in ALL_DATASETS:
        data = load_dataset(name)
        _, state_init = get_initial_condition(data["met_df"], data["nsd_df"])
        state_init_by_dataset[name] = state_init
    save_pkl(state_init_by_dataset, 'state_init_by_dataset.pkl')

    volume_results = compute_volume_results(
        ALL_DATASETS, cfg.INITIAL_VOLUMES, build_schedule, step_len,
    )
    save_pkl(volume_results, 'volume_results.pkl')

    set_model_by_dataset = {}
    for name in ALL_DATASETS:
        Fin, Fout, Gal_feed, Urd_feed = build_schedule(name)
        V_traj = volume_results[name][1:]
        traj = simulate_dataset(
            state_init_by_dataset[name], Fin, Fout, Gal_feed, Urd_feed,
            V_traj, time_grid, step_len, name=name,
        )
        set_model_by_dataset[name] = np.vstack([state_init_by_dataset[name], traj])
    save_pkl(set_model_by_dataset, 'set_model_by_dataset.pkl')

    T_meas_by_dataset = {}
    interval = int(24.0 / cfg.DT)
    for name in ALL_DATASETS:
        data = load_dataset(name)
        N_meas_time = data["set_meas"].shape[0]
        if len(cfg.T_MEAS_FIXED) == N_meas_time:
            T_meas_by_dataset[name] = np.array(cfg.T_MEAS_FIXED)
        else:
            T_meas_by_dataset[name] = T_model[[i * interval for i in range(N_meas_time)]]
    save_pkl(T_meas_by_dataset, 'T_meas_by_dataset.pkl')

    set_meas_ens_by_dataset = generate_measurement_ensembles(
        ALL_DATASETS, load_dataset, cfg.MEAS_NUM, ENSEMBLE_SIZE, var_meas,
    )
    save_pkl(set_meas_ens_by_dataset, 'set_meas_ens_by_dataset.pkl')
else:
    print("\nLoading shared data ...")
    volume_results          = load_pkl('volume_results.pkl')
    state_init_by_dataset   = load_pkl('state_init_by_dataset.pkl')
    set_model_by_dataset    = load_pkl('set_model_by_dataset.pkl')
    T_meas_by_dataset       = load_pkl('T_meas_by_dataset.pkl')
    set_meas_ens_by_dataset = load_pkl('set_meas_ens_by_dataset.pkl')

# Save run config
save_pkl({
    "RUN_NAME": RUN_NAME, "KQ": KQ, "N_RUNS": N_RUNS,
    "ENSEMBLE_SIZE": ENSEMBLE_SIZE,
    "Q_diag": np.diag(Q).copy(),
    "R_diag": np.diag(R).copy(),
    "P0_diag": P0_diag.copy(),
    "PROCESS_NOISE_VAR": dict(cfg.PROCESS_NOISE_VAR),
    "PROCESS_NOISE_CV": dict(getattr(cfg, 'PROCESS_NOISE_CV', {})),
    "process_noise_cv_indices": dict(process_noise_cv),
    "no_update_indices": list(no_update_indices),
    "NO_UPDATE_STATES": list(getattr(cfg, 'NO_UPDATE_STATES', [])),
    "MEASUREMENT_NOISE_VAR": dict(cfg.MEASUREMENT_NOISE_VAR),
    "INITIAL_COV_OVERRIDE": dict(getattr(cfg, 'INITIAL_COV_OVERRIDE', {})),
    "STATE_NAMES": list(cfg.STATE_NAMES),
    "MEASURED_STATES": list(cfg.MEASURED_STATES),
}, "run_config.pkl")

# ── Run EnKF per dataset ─────────────────────────────────────────────────────
for name in DATASET_NAMES:
    if not args.force and (PKL_DIR / f"enkf_results_{name}.pkl").exists():
        print(f"\n{name}: already done, skipping.")
        continue

    print(f"\nRunning EnKF for {name} ({N_RUNS} run(s), N={ENSEMBLE_SIZE}) ...")
    run_enkf_multi_dataset(
        datasets_cfg=select_datasets(name),
        load_dataset_fn=load_dataset,
        build_schedule_fn=build_schedule,
        state_init_by_dataset={name: state_init_by_dataset[name]},
        volume_results=volume_results,
        set_meas_ens_by_dataset={name: set_meas_ens_by_dataset[name]},
        T_meas_by_dataset={name: T_meas_by_dataset[name]},
        state_num=cfg.STATE_NUM, meas_num=cfg.MEAS_NUM,
        ensemble_size=ENSEMBLE_SIZE, n_runs=N_RUNS,
        Q=Q, R=R, H=H, dt_kf=dt_kf, N_kf=N_kf,
        P0=P0, process_noise_cv=process_noise_cv,
        no_update_indices=no_update_indices, save_fn=save_pkl,
    )

# ── RMSE ─────────────────────────────────────────────────────────────────────
enkf_results_by_dataset = {}
completed = []
for name in DATASET_NAMES:
    path = PKL_DIR / f"enkf_results_{name}.pkl"
    if path.exists():
        enkf_results_by_dataset[name] = load_pkl(f"enkf_results_{name}.pkl")
        completed.append(name)

if completed:
    print(f"\nComputing RMSE for: {completed}")
    rmse_df = compute_rmse_table(
        select_datasets(*completed), load_dataset,
        set_model_by_dataset, enkf_results_by_dataset,
        T_model, T_kf, T_meas_by_dataset,
        cfg.AXIS_NAMES, cfg.STATE_NUM,
    )
    save_pkl(rmse_df, "rmse.pkl")
    print("\n" + rmse_df.to_string())

print("\nPipeline complete.")
