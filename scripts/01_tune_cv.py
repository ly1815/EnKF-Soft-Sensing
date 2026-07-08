"""
01_tune_cv.py  —  Stage 3: measured-state CV calibration (NIV = 1)
=================================================================
Automated, systematic calibration of the 8 measured-state multiplicative CVs on P4.

Each measured state's per-step CV is chosen so that its normalised innovation
variance (filter consistency statistic)

    NIV_j = mean_over_updates( d_j^2 / S_jj ),   d = z - H x_forecast,  S = P_zz + R

equals 1 (an under-dispersed filter has NIV > 1, over-dispersed NIV < 1). Because
S_jj grows ~ CV_j^2, NIV_j is monotone decreasing in CV_j, so the fixed point

    CV_j  <-  CV_j * sqrt(NIV_j)

converges to NIV_j = 1. All 8 NIVs come from ONE EnKF pass, so a handful of passes
suffices.

Termination (general, no per-state special-casing): iterate until all |NIV-1| < tol, or
--iters is reached (a max-iteration safeguard). CVs are clamped to [CV_MIN, CV_MAX]; a
state that would leave that range stays pinned at the bound and is FLAGGED (for reporting
only — the flags do NOT gate convergence):
  - CAPPED at CV_MAX: still under-dispersed at the cap -> structural model bias (NIV=1
    there would only mask the bias with inflated noise).
  - FLOORED at CV_MIN: still over-dispersed at the floor -> here S ~ R, so the residual
    over-dispersion is driven by the measurement noise R, not process noise, and cannot be
    removed by lowering CV. Flags R as likely overestimated.
When any state is pinned it keeps the loop running to --iters; the pinned CV/NIV are
unchanged by the extra iterations, so the final CVs match an early stop.

The unmeasured-state additive noise uses the two-stage alpha from config
(PROCESS_NOISE_ALPHA for NSDs, PROCESS_NOISE_ALPHA_OBS for Asn/Glu).

Outputs (results/<run>/):
  - tune_cv_checkpoint.json : per-iteration state (atomic, --resume safe)
  - cv_final.json           : final CVs, NIV, capped/floored flags, NIV history
  - pkl/cv_tuned_<DS>.pkl    : final all-state mean + std trajectories and ±1σ/±2σ bands,
                               plus the open-loop model trajectory (uncertainty saved)
  - figures/cv_niv_convergence_<DS>.png : NIV per iteration -> 1.0
  - figures/cv_tuned_states_<DS>.png    : all-17-state grid (mean, bands, model, meas)

The CV cap CV_MAX (default 0.006) is a physical band-plausibility ceiling, NOT a tuned
value: 0.006/step compounds to ~0.006*sqrt(2400) ~ 0.29, i.e. ~30% accumulated model error
over a 24h measurement interval, which keeps each state's uncertainty band within
physically plausible metabolite ranges. 0.006 sits just above mAb's NIV=1 CV (~0.0058), so
it caps only the bias-limited states (Glc, Urd, Lac) and leaves the rest at NIV~1. Filter consistency (NIV=1) is the primary target,
but where reaching it would demand a larger CV -- driven by structural model bias, chiefly
glucose (whose NIV=1 band climbed toward ~200 mM against a ~144 mM feed, i.e. physically
impossible) -- the state is pinned at the cap and left UNDER-dispersed (NIV>1). A physically
bounded band is preferred over a statistically consistent but physically implausible one.
Glc pins hardest; Urd and Lac also pin (their NIV=1 CVs, ~0.019 and ~0.011, exceed the cap).
(Earlier caps of 0.05 and 0.02 let glucose's multiplicative noise blow the band up to
~1000 and ~200 mM respectively.)

Usage (macOS venv):
    ./.venv/bin/python scripts/01_tune_cv.py
    ./.venv/bin/python scripts/01_tune_cv.py --iters 10 --cv-max 0.02
    ./.venv/bin/python scripts/01_tune_cv.py --resume        # continue from checkpoint
"""

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import warnings
import numpy as np
import pandas as pd
from scipy.linalg import LinAlgWarning
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

warnings.filterwarnings("ignore", category=LinAlgWarning)

import nsd_enkf.config as cfg
from nsd_enkf.data_loader import (
    select_datasets, load_dataset, get_initial_condition, build_schedule,
)
from nsd_enkf.model import compute_volume_results, model_step, simulate_dataset
from nsd_enkf.analysis import generate_measurement_ensembles
from nsd_enkf.enkf import (
    EnsembleKalmanFilter, run_enkf_single_with_ensemble_diagnostics,
)

parser = argparse.ArgumentParser(description="Systematic per-state CV calibration to NIV=1 on P4")
parser.add_argument("--dataset", default="P4")
parser.add_argument("--ensemble-size", default=cfg.ENSEMBLE_SIZE, type=int)
parser.add_argument("--iters", default=12, type=int)
parser.add_argument("--cv-max", default=0.006, type=float,
                    help="per-step CV cap; 0.006 ~ 30%%/24h accumulated model error, a "
                         "physical band-plausibility ceiling. Sits just above mAb's NIV=1 "
                         "CV (~0.0058), so it caps only the bias-limited states (Glc, Urd, "
                         "Lac) at NIV>1 and leaves the rest at NIV~1")
parser.add_argument("--cv-min", default=1e-4, type=float)
parser.add_argument("--tol", default=0.15, type=float, help="stop when all |NIV-1| < tol (uncapped states)")
parser.add_argument("--seed", default=42, type=int)
parser.add_argument("--run", default="cv_tuning", help="results/<run>/ for checkpoint + final CVs")
parser.add_argument("--resume", action="store_true", help="resume from checkpoint if present")
parser.add_argument("--no-plots", action="store_true", help="skip figure generation")
parser.add_argument("--traj-down", default=20, type=int, help="downsample factor for plotted trajectories")
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
floored = {s: False for s in meas_names}
niv_history = []   # list of per-iteration NIV arrays (for the convergence plot)
start_iter = 0

if args.resume and CKPT.exists():
    ck = json.load(open(CKPT))
    cv = {s: float(ck["cv"][s]) for s in meas_names}
    capped = {s: bool(ck["capped"][s]) for s in meas_names}
    floored = {s: bool(ck.get("floored", {}).get(s, False)) for s in meas_names}
    niv_history = [list(map(float, h)) for h in ck.get("niv_history", [])]
    start_iter = int(ck["iter"]) + 1
    print(f"\nResuming from checkpoint at iteration {start_iter} "
          f"(CVs = {[round(cv[s],4) for s in meas_names]})")

hdr = f"{'iter':>4s} | " + " | ".join(f"{s:>6s}" for s in meas_names)
print("\nNIV per iteration (target 1.0):"); print(hdr); print("-" * len(hdr))

converged = False
for it in range(start_iter, args.iters):
    niv = run_niv(cv)
    niv_history.append([float(v) for v in niv])
    print(f"{it:>4d} | " + " | ".join(f"{v:6.2f}" for v in niv), flush=True)

    done = True
    for j, s in enumerate(meas_names):
        if np.isnan(niv[j]):
            continue
        new = cv[s] * np.sqrt(niv[j])
        new_c = float(np.clip(new, args.cv_min, args.cv_max))
        # bound flags are recorded for REPORTING only (tables/plots/JSON); they do NOT
        # gate convergence — the method stays general with no per-state special-casing.
        # A state pinned at a bound (Glc capped by structural bias, Gln floored / R-driven)
        # simply keeps `done` False, so the loop runs to the --iters safeguard. Those
        # extra iterations leave the pinned CV/NIV unchanged, so the final CVs are the same.
        capped[s] = bool(new > args.cv_max)
        floored[s] = bool(new < args.cv_min)
        if abs(niv[j] - 1.0) > args.tol:
            done = False
        cv[s] = new_c

    # checkpoint AFTER every iteration (atomic) so a crash resumes here
    write_json(CKPT, {"iter": it, "cv": cv, "capped": capped, "floored": floored,
                      "niv_history": niv_history,
                      "niv_last": {s: float(niv[j]) for j, s in enumerate(meas_names)}})
    if done:
        converged = True
        print(f"\nConverged after iteration {it} (all states within tol).")
        break

# Final NIV with converged CVs
niv_final = run_niv(cv)

def state_note(j, s):
    if capped[s]:
        return "STRUCTURAL BIAS (capped)"
    if floored[s]:
        return "OVER-DISPERSED at floor (R likely too big)"
    return "well-calibrated" if abs(niv_final[j] - 1) < 0.3 else ""

print("\n" + "=" * 70)
print(f"Final systematic CVs (NIV -> 1 on {DS}):")
print(f"{'state':>6s} | {'CV_old':>8s} | {'CV_new':>8s} | {'NIV':>6s} | note")
print("-" * 66)
for j, s in enumerate(meas_names):
    print(f"{s:>6s} | {cfg.PROCESS_NOISE_CV[s]:8.4f} | {cv[s]:8.4f} | {niv_final[j]:6.2f} | {state_note(j, s)}")

print("\nPaste into config.py PROCESS_NOISE_CV:")
print("PROCESS_NOISE_CV = {")
for j, s in enumerate(meas_names):
    if capped[s]:
        tag = "  # capped (structural bias)"
    elif floored[s]:
        tag = "  # floored (R likely too big)"
    else:
        tag = f"  # NIV={niv_final[j]:.2f}"
    print(f"    '{s}': {cv[s]:.4f},{tag}")
print("}")

# Persist final result (survives even if the terminal scrollback is lost)
write_json(FINAL, {
    "dataset": DS, "ensemble_size": ENS, "cv_max": args.cv_max, "cv_min": args.cv_min,
    "converged": converged,
    "cv": {s: round(cv[s], 5) for s in meas_names},
    "niv_final": {s: round(float(niv_final[j]), 3) for j, s in enumerate(meas_names)},
    "capped": capped, "floored": floored,
    "niv_history": niv_history,
})
print(f"\nSaved final CVs to: {FINAL}")
print(f"Checkpoint (for --resume): {CKPT}")

# ── Artifacts: full trajectories with uncertainty + visualization plots ──────
# One production-config EnKF pass at the final CVs, recording the ensemble mean and
# std at every step for ALL 17 states (measured multiplicative-CV noise + unmeasured
# two-stage-alpha additive noise, IQR clipping) — same runner the Option-B pipeline
# uses, so figures/pkls are directly comparable across the tuning pipeline.
PKL_DIR = OUT_DIR / "pkl"; FIG_DIR = OUT_DIR / "figures"
PKL_DIR.mkdir(parents=True, exist_ok=True); FIG_DIR.mkdir(parents=True, exist_ok=True)

cv_idx_final = {cfg.STATE_NAMES.index(s): cv[s] for s in meas_names}
np.random.seed(args.seed)
_, std_traj, mean_traj = run_enkf_single_with_ensemble_diagnostics(
    dataset_name=DS, load_dataset_fn=load_dataset, build_schedule_fn=build_schedule,
    state_init=state_init, volume_results=volume_results,
    set_meas_ens=set_meas_ens, T_meas=T_meas,
    state_num=cfg.STATE_NUM, meas_num=n_meas, ensemble_size=ENS,
    Q=Q, R=R, H=H, dt_kf=dt_kf, N_kf=N_kf,
    P0=P0, process_noise_cv=cv_idx_final,
    no_update_indices=set(no_update_indices), clip_indices=set(clip_indices),
)

# Open-loop model trajectory (no assimilation) for overlay/comparison
model_traj = simulate_dataset(state_init, Fin, Fout, Gal_feed, Urd_feed,
                              V_traj, time_grid, step_len, name=DS)
model_traj = np.vstack([state_init, model_traj])

with open(PKL_DIR / f"cv_tuned_{DS}.pkl", "wb") as f:
    pickle.dump({
        "dataset": DS, "ensemble_size": ENS, "seed": args.seed, "converged": converged,
        "cv_final": {s: cv[s] for s in meas_names},
        "niv_final": {s: float(niv_final[j]) for j, s in enumerate(meas_names)},
        "capped": capped, "floored": floored, "niv_history": niv_history,
        "T": T_model, "state_names": list(cfg.STATE_NAMES),
        "mean_trajectory": mean_traj, "std_trajectory": std_traj,
        "band_1sigma_lo": np.maximum(mean_traj - std_traj, 0.0),
        "band_1sigma_hi": mean_traj + std_traj,
        "band_2sigma_lo": np.maximum(mean_traj - 2 * std_traj, 0.0),
        "band_2sigma_hi": mean_traj + 2 * std_traj,
        "model_trajectory": model_traj,
    }, f)
print(f"Saved trajectories + uncertainty bands: {PKL_DIR / f'cv_tuned_{DS}.pkl'}")

if not args.no_plots:
    # (1) NIV convergence to 1.0 per iteration
    hist = np.array(niv_history)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for j, s in enumerate(meas_names):
        ax.plot(np.arange(hist.shape[0]), hist[:, j], marker="o", ms=4, label=s)
    ax.axhline(1.0, color="k", ls="--", lw=1)
    ax.axhspan(1 - args.tol, 1 + args.tol, color="green", alpha=0.08, label=f"|NIV-1|<{args.tol}")
    ax.set_xlabel("iteration"); ax.set_ylabel(r"NIV = mean($d^2/S$)")
    ax.set_title(f"CV calibration convergence to NIV=1  ({DS}, N={ENS})")
    ax.legend(ncol=3, fontsize=8); ax.grid(alpha=0.2)
    fig.tight_layout(); fig.savefig(FIG_DIR / f"cv_niv_convergence_{DS}.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {FIG_DIR / f'cv_niv_convergence_{DS}.png'}")

    # (2) all-state grid: EnKF mean + 1/2-sigma bands + open-loop model + measurements
    d = load_dataset(DS)
    set_meas_all = d["set_meas"].astype(float); set_err = d["set_meas_errorbar"].astype(float)
    nsd_meas = pd.DataFrame(d["NSD_meas"]).apply(pd.to_numeric, errors="coerce").to_numpy()
    nsd_err = pd.DataFrame(d["NSD_meas_errorbar"]).apply(pd.to_numeric, errors="coerce").to_numpy()
    asn_col = set_meas_all.shape[1] - 1
    n_nsd = 7; nsd_state_idx = list(range(cfg.STATE_NUM - n_nsd, cfg.STATE_NUM))
    meas_by_state = {i: (set_meas_all[:, i], set_err[:, i]) for i in range(n_meas)}
    meas_by_state[cfg.STATE_NAMES.index("Asn")] = (set_meas_all[:, asn_col], set_err[:, asn_col])
    for j in range(n_nsd):
        meas_by_state[nsd_state_idx[j]] = (nsd_meas[:, j], nsd_err[:, j])

    DOWN = max(int(args.traj_down), 1)
    tds = T_model[::DOWN]
    fig, axes = plt.subplots(5, 4, figsize=(20, 15)); axes = axes.flatten()
    for si in range(cfg.STATE_NUM):
        ax = axes[si]; m = mean_traj[::DOWN, si]; s_ = std_traj[::DOWN, si]
        ax.fill_between(tds, np.maximum(m - 2 * s_, 0), m + 2 * s_, color="steelblue", alpha=0.15)
        ax.fill_between(tds, np.maximum(m - s_, 0), m + s_, color="steelblue", alpha=0.30)
        ax.plot(tds, m, color="steelblue", lw=2.0)
        ax.plot(tds, model_traj[::DOWN, si], color="red", lw=1.6)
        if si in meas_by_state:
            v, e = meas_by_state[si]
            ax.errorbar(T_meas, v, yerr=e, fmt="o", color="darkorange", markersize=4,
                        capsize=2, elinewidth=1, alpha=0.9, zorder=5)
        sn = cfg.STATE_NAMES[si]
        if sn in capped and capped[sn]:
            tag = f"  [capped, NIV={niv_final[meas_names.index(sn)]:.2f}]"
        elif sn in floored and floored[sn]:
            tag = f"  [floored, NIV={niv_final[meas_names.index(sn)]:.2f}]"
        elif si not in meas_by_state:
            tag = "  (no meas)"
        else:
            tag = ""
        ax.set_title(f"{sn}{tag}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Time (h)", fontsize=9); ax.grid(alpha=0.15)
    for k in range(cfg.STATE_NUM, len(axes)):
        axes[k].set_visible(False)
    fig.legend(handles=[
        Line2D([0], [0], color="red", lw=1.8, label="Open-loop model"),
        Line2D([0], [0], color="steelblue", lw=2.0, label="EnKF mean"),
        Patch(facecolor="steelblue", alpha=0.30, label=r"$\pm1\sigma$"),
        Patch(facecolor="steelblue", alpha=0.15, label=r"$\pm2\sigma$"),
        Line2D([0], [0], color="darkorange", marker="o", lw=0, markersize=6, label="Measurements"),
    ], loc="lower center", ncol=5, fontsize=12, frameon=False, bbox_to_anchor=(0.5, -0.005))
    fig.suptitle(f"Automated CV calibration (NIV->1) — {DS}, N={ENS}",
                 fontsize=15, fontweight="bold", y=1.005)
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(FIG_DIR / f"cv_tuned_states_{DS}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {FIG_DIR / f'cv_tuned_states_{DS}.png'}")
