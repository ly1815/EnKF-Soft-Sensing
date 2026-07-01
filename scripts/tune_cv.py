"""
tune_cv.py
==========
Automated, systematic calibration of the 8 measured-state multiplicative CVs on P4.

Each measured state's per-step CV is chosen so that its normalised innovation
variance (filter consistency statistic)

    NIV_j = mean_over_updates( d_j^2 / S_jj ),   d = z - H x_forecast,  S = P_zz + R

equals 1 (an under-dispersed filter has NIV > 1, over-dispersed NIV < 1). Because
S_jj grows ~ CV_j^2, NIV_j is monotone decreasing in CV_j, so the fixed point

    CV_j  <-  CV_j * sqrt(NIV_j)

converges to NIV_j = 1. All 8 NIVs come from ONE EnKF pass, so a handful of passes
suffices. CVs are capped at CV_MAX; a state that stays under-dispersed at the cap
is flagged as structural model bias (NIV=1 there would only mask the bias with
inflated noise) and left at the cap.

The unmeasured-state additive noise uses the two-stage alpha from config
(PROCESS_NOISE_ALPHA for NSDs, PROCESS_NOISE_ALPHA_OBS for Asn/Glu).

Usage:
    ./.venv/Scripts/python.exe scripts/tune_cv.py
    ./.venv/Scripts/python.exe scripts/tune_cv.py --iters 12 --cv-max 0.05
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import warnings
import numpy as np
from scipy.linalg import LinAlgWarning

warnings.filterwarnings("ignore", category=LinAlgWarning)

import nsd_enkf.config as cfg
from nsd_enkf.data_loader import (
    select_datasets, load_dataset, get_initial_condition, build_schedule,
)
from nsd_enkf.model import compute_volume_results, model_step
from nsd_enkf.analysis import generate_measurement_ensembles
from nsd_enkf.enkf import EnsembleKalmanFilter

parser = argparse.ArgumentParser(description="Systematic per-state CV calibration to NIV=1 on P4")
parser.add_argument("--dataset", default="P4")
parser.add_argument("--ensemble-size", default=cfg.ENSEMBLE_SIZE, type=int)
parser.add_argument("--iters", default=12, type=int)
parser.add_argument("--cv-max", default=0.05, type=float)
parser.add_argument("--cv-min", default=1e-4, type=float)
parser.add_argument("--tol", default=0.15, type=float, help="stop when all |NIV-1| < tol (uncapped states)")
parser.add_argument("--seed", default=42, type=int)
parser.add_argument("--run", default="cv_tuning", help="results/<run>/ for checkpoint + final CVs")
parser.add_argument("--resume", action="store_true", help="resume from checkpoint if present")
args = parser.parse_args()

DS = args.dataset
ENS = args.ensemble_size
meas_names = cfg.MEASURED_STATES
n_meas = cfg.MEAS_NUM

# Checkpoint / output files (written every iteration so a crash never loses work)
OUT_DIR = cfg.PROJECT_ROOT / "results" / args.run
OUT_DIR.mkdir(parents=True, exist_ok=True)
CKPT = OUT_DIR / "tune_cv_checkpoint.json"
FINAL = OUT_DIR / "cv_final.json"

def write_json(path, obj):
    """Atomic write: temp file then replace, so a kill mid-write can't corrupt it."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)

print("=" * 70)
print(f"Systematic CV calibration to NIV=1 on {DS}  (N={ENS}, cap={args.cv_max})")
print("=" * 70)

# ── Grids / matrices ─────────────────────────────────────────────────────────
time_grid = np.arange(cfg.DT, cfg.T_END + cfg.DT, cfg.DT)
step_len = np.full(len(time_grid), cfg.DT)
N_model = int(cfg.T_END / cfg.DT)
T_model = np.linspace(0, cfg.T_END, N_model + 1)
dt_kf = cfg.DT
N_kf = len(T_model) - 1

var_meas = np.array(list(cfg.MEASUREMENT_NOISE_VAR.values()))
R = np.diag(var_meas[:n_meas])
H = np.hstack((np.eye(n_meas), np.zeros((n_meas, cfg.STATE_NUM - n_meas))))
clip_indices = {cfg.STATE_NAMES.index(s) for s in cfg.CLIP_STATES}
no_update_indices = {cfg.STATE_NAMES.index(s) for s in cfg.NO_UPDATE_STATES}

# Fixed additive var for unmeasured states (two-stage alpha from config)
scale_vec = np.zeros(cfg.STATE_NUM)
for s, sc in cfg.PROCESS_NOISE_SCALE.items():
    scale_vec[cfg.STATE_NAMES.index(s)] = sc
ALPHA_OBS = getattr(cfg, "PROCESS_NOISE_ALPHA_OBS", 0.0)
obs_idx = [cfg.STATE_NAMES.index(s) for s in getattr(cfg, "ALPHA_OBS_STATES", [])]
alpha_full = np.full(cfg.STATE_NUM, cfg.PROCESS_NOISE_ALPHA)
for i in obs_idx:
    alpha_full[i] = ALPHA_OBS
var_model = (alpha_full * scale_vec) ** 2
Q = np.diag(var_model)

# ── Data ─────────────────────────────────────────────────────────────────────
volume_results = compute_volume_results(select_datasets(DS), cfg.INITIAL_VOLUMES,
                                        build_schedule, step_len)
data = load_dataset(DS)
_, state_init = get_initial_condition(data["met_df"], data["nsd_df"])
set_meas = data["set_meas"][:, :n_meas].astype(float)
T_meas = np.array(cfg.T_MEAS_FIXED)
np.random.seed(args.seed)
set_meas_ens = generate_measurement_ensembles(select_datasets(DS), load_dataset,
                                              n_meas, ENS, var_meas)[DS]
Fin, Fout, Gal_feed, Urd_feed = build_schedule(DS)
V_traj = volume_results[DS][1:]

P0_diag = var_model.copy()
P0_diag[:n_meas] = var_meas[:n_meas]
P0 = np.diag(P0_diag)

time_steps_A = [round(i * dt_kf, 2) for i in range(N_kf)]
meas_time_to_index = {round(t, 2): i for i, t in enumerate(T_meas.tolist())}


def run_niv(cv_by_name):
    """One EnKF pass on DS; return per-measured-state NIV = mean(d^2 / S_jj)."""
    cv_idx = {cfg.STATE_NAMES.index(s): cv for s, cv in cv_by_name.items()}
    np.random.seed(args.seed)
    enkf = EnsembleKalmanFilter(cfg.STATE_NUM, n_meas)
    enkf.x = state_init.copy(); enkf.Q = Q.copy(); enkf.R = R.copy(); enkf.H = H.copy()
    enkf.fx = model_step; enkf.dt = dt_kf
    enkf.process_noise_cv = dict(cv_idx)
    enkf.no_update_indices = set(no_update_indices)
    enkf.clip_indices = set(clip_indices)
    enkf.create_ensemble(ENS, P0)

    sq_norm = [[] for _ in range(n_meas)]   # squared normalised innovations
    for idx_A, step_A in enumerate(time_steps_A):
        enkf.predict({"Fin": Fin[idx_A], "Fout": Fout[idx_A], "V": V_traj[idx_A],
                      "Gal_feed": Gal_feed[idx_A], "Urd_feed": Urd_feed[idx_A]})
        if step_A in meas_time_to_index:
            b = meas_time_to_index[step_A]
            Z = enkf.X @ enkf.H.T                 # (N, n_meas) forecast in meas space
            z_mean = Z.mean(axis=0)
            Ez = Z - z_mean
            S = (Ez.T @ Ez) / (ENS - 1) + R
            d = set_meas[b] - z_mean
            for j in range(n_meas):
                if not np.isnan(set_meas[b, j]) and S[j, j] > 0:
                    sq_norm[j].append(d[j] ** 2 / S[j, j])
            enkf.update(set_meas_ens[b])
    return np.array([np.mean(v) if v else np.nan for v in sq_norm])


# ── Fixed-point iteration (checkpointed every iteration) ─────────────────────
cv = {s: cfg.PROCESS_NOISE_CV[s] for s in meas_names}
capped = {s: False for s in meas_names}
start_iter = 0

if args.resume and CKPT.exists():
    ck = json.load(open(CKPT))
    cv = {s: float(ck["cv"][s]) for s in meas_names}
    capped = {s: bool(ck["capped"][s]) for s in meas_names}
    start_iter = int(ck["iter"]) + 1
    print(f"\nResuming from checkpoint at iteration {start_iter} "
          f"(CVs = {[round(cv[s],4) for s in meas_names]})")

hdr = f"{'iter':>4s} | " + " | ".join(f"{s:>6s}" for s in meas_names)
print("\nNIV per iteration (target 1.0):"); print(hdr); print("-" * len(hdr))

converged = False
for it in range(start_iter, args.iters):
    niv = run_niv(cv)
    print(f"{it:>4d} | " + " | ".join(f"{v:6.2f}" for v in niv), flush=True)

    done = True
    for j, s in enumerate(meas_names):
        if np.isnan(niv[j]):
            continue
        new = cv[s] * np.sqrt(niv[j])
        new_c = float(np.clip(new, args.cv_min, args.cv_max))
        capped[s] = bool(new > args.cv_max)
        # convergence check only for states not pinned at the cap
        if not capped[s] and abs(niv[j] - 1.0) > args.tol:
            done = False
        cv[s] = new_c

    # checkpoint AFTER every iteration (atomic) so a crash resumes here
    write_json(CKPT, {"iter": it, "cv": cv, "capped": capped,
                      "niv_last": {s: float(niv[j]) for j, s in enumerate(meas_names)}})
    if done:
        converged = True
        print(f"\nConverged after iteration {it} (all uncapped states within tol).")
        break

# Final NIV with converged CVs
niv_final = run_niv(cv)

print("\n" + "=" * 70)
print("Final systematic CVs (NIV -> 1 on P4):")
print(f"{'state':>6s} | {'CV_old':>8s} | {'CV_new':>8s} | {'NIV':>6s} | note")
print("-" * 58)
for j, s in enumerate(meas_names):
    note = "STRUCTURAL BIAS (capped)" if capped[s] else ("well-calibrated" if abs(niv_final[j]-1) < 0.3 else "")
    print(f"{s:>6s} | {cfg.PROCESS_NOISE_CV[s]:8.4f} | {cv[s]:8.4f} | {niv_final[j]:6.2f} | {note}")

print("\nPaste into config.py PROCESS_NOISE_CV:")
print("PROCESS_NOISE_CV = {")
for s in meas_names:
    tag = "  # capped (structural bias)" if capped[s] else f"  # NIV={niv_final[meas_names.index(s)]:.2f}"
    print(f"    '{s}': {cv[s]:.4f},{tag}")
print("}")

# Persist final result (survives even if the terminal scrollback is lost)
write_json(FINAL, {
    "dataset": DS, "ensemble_size": ENS, "cv_max": args.cv_max,
    "converged": converged,
    "cv": {s: round(cv[s], 5) for s in meas_names},
    "niv_final": {s: round(float(niv_final[j]), 3) for j, s in enumerate(meas_names)},
    "capped": capped,
})
print(f"\nSaved final CVs to: {FINAL}")
print(f"Checkpoint (for --resume): {CKPT}")
