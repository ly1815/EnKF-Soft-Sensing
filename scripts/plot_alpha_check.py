"""
plot_alpha_check.py
===================
Quick visual check of Option-B tuning: run ONE EnKF diagnostic pass on a dataset
for a given ALPHA and plot all 17 states in a single grid, each with the
open-loop model, EnKF posterior mean, +/-1 sigma and +/-2 sigma ensemble bands,
and all available measurements (metabolites, Asn, and the 7 NSDs) with error bars.

Lets you eyeball where the +/-2 sigma bands cover the data and where they do not
(e.g. GDP-Man / GDP-Fuc / CMP-Neu5Ac, whose open-loop climatological scale is
tiny so their bands stay narrow regardless of alpha).

Usage:
    ./.venv/Scripts/python.exe scripts/plot_alpha_check.py --alpha 0.05
    ./.venv/Scripts/python.exe scripts/plot_alpha_check.py --alpha 0.1 --dataset P4
"""

import argparse
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
from nsd_enkf.model import compute_volume_results, simulate_dataset
from nsd_enkf.analysis import generate_measurement_ensembles
from nsd_enkf.enkf import run_enkf_single_with_ensemble_diagnostics

# ── CLI ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Plot all states vs measurements for one ALPHA")
parser.add_argument("--alpha", default=cfg.PROCESS_NOISE_ALPHA, type=float)
parser.add_argument("--dataset", default="P4")
parser.add_argument("--ensemble-size", default=cfg.ENSEMBLE_SIZE, type=int)
parser.add_argument("--seed", default=42, type=int)
parser.add_argument("--downsample", default=20, type=int)
args = parser.parse_args()

ALPHA = args.alpha
DS = args.dataset
DOWN = args.downsample

print(f"Alpha check: alpha={ALPHA:g}, dataset={DS}, N={args.ensemble_size}")

# ── Grids / matrices ─────────────────────────────────────────────────────────
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

# Two-stage alpha: ALPHA applies to NSDs; observable Asn/Glu pinned at ALPHA_OBS.
ALPHA_OBS = getattr(cfg, "PROCESS_NOISE_ALPHA_OBS", 0.0)
obs_idx = [cfg.STATE_NAMES.index(s) for s in getattr(cfg, "ALPHA_OBS_STATES", [])]
alpha_per_state = np.full(cfg.STATE_NUM, ALPHA)
for _i in obs_idx:
    alpha_per_state[_i] = ALPHA_OBS
var_model = (alpha_per_state * scale_vec) ** 2

# ── Data ─────────────────────────────────────────────────────────────────────
volume_results = compute_volume_results(select_datasets(DS), cfg.INITIAL_VOLUMES,
                                        build_schedule, step_len)
data = load_dataset(DS)
_, state_init = get_initial_condition(data["met_df"], data["nsd_df"])
T_meas = np.array(cfg.T_MEAS_FIXED)

# Open-loop model
Fin, Fout, Gal_feed, Urd_feed = build_schedule(DS)
model_traj = simulate_dataset(state_init, Fin, Fout, Gal_feed, Urd_feed,
                             volume_results[DS][1:], time_grid, step_len, name=DS)
model_traj = np.vstack([state_init, model_traj])

# Measurement ensemble
np.random.seed(args.seed)
set_meas_ens = generate_measurement_ensembles(
    select_datasets(DS), load_dataset, cfg.MEAS_NUM, args.ensemble_size, var_meas)[DS]

# EnKF single pass
P0_diag = var_model.copy()
for i in range(cfg.MEAS_NUM):
    P0_diag[i] = var_meas[i]
np.random.seed(args.seed)
_, std_traj, mean_traj = run_enkf_single_with_ensemble_diagnostics(
    dataset_name=DS, load_dataset_fn=load_dataset, build_schedule_fn=build_schedule,
    state_init=state_init, volume_results=volume_results,
    set_meas_ens=set_meas_ens, T_meas=T_meas,
    state_num=cfg.STATE_NUM, meas_num=cfg.MEAS_NUM, ensemble_size=args.ensemble_size,
    Q=np.diag(var_model), R=R, H=H, dt_kf=dt_kf, N_kf=N_kf,
    P0=np.diag(P0_diag), process_noise_cv=process_noise_cv,
    no_update_indices=no_update_indices, clip_indices=clip_indices,
)

# ── Measurements per state index ─────────────────────────────────────────────
set_meas = data["set_meas"].astype(float)
set_meas_err = data["set_meas_errorbar"].astype(float)
NSD_meas = pd.DataFrame(data["NSD_meas"]).apply(pd.to_numeric, errors="coerce").to_numpy()
NSD_err = pd.DataFrame(data["NSD_meas_errorbar"]).apply(pd.to_numeric, errors="coerce").to_numpy()
n_nsd = NSD_meas.shape[1]
asn_col = set_meas.shape[1] - 1  # Asn is the last metabolite column

meas_by_state = {}                      # state_idx -> (values, errors)
for i in range(cfg.MEAS_NUM):           # measured metabolites
    meas_by_state[i] = (set_meas[:, i], set_meas_err[:, i])
meas_by_state[cfg.STATE_NAMES.index("Asn")] = (set_meas[:, asn_col], set_meas_err[:, asn_col])
for j in range(n_nsd):                   # NSDs
    meas_by_state[cfg.STATE_NUM - n_nsd + j] = (NSD_meas[:, j], NSD_err[:, j])

# ── Plot 17-state grid ───────────────────────────────────────────────────────
tds = T_model[::DOWN]
ncol, nrow = 4, 5
fig, axes = plt.subplots(nrow, ncol, figsize=(5.0 * ncol, 3.0 * nrow))
axes = axes.flatten()

for si in range(cfg.STATE_NUM):
    ax = axes[si]
    m = mean_traj[::DOWN, si]
    s = std_traj[::DOWN, si]
    ax.fill_between(tds, np.maximum(m - 2 * s, 0), m + 2 * s, color="steelblue", alpha=0.15)
    ax.fill_between(tds, np.maximum(m - s, 0), m + s, color="steelblue", alpha=0.30)
    ax.plot(tds, m, color="steelblue", lw=2.0)
    ax.plot(tds, model_traj[::DOWN, si], color="red", lw=1.6)
    if si in meas_by_state:
        vals, errs = meas_by_state[si]
        ax.errorbar(T_meas, vals, yerr=errs, fmt="o", color="darkorange",
                    markersize=4.0, capsize=2, elinewidth=1, alpha=0.9, zorder=5)
    tag = "" if si in meas_by_state else "  (no meas)"
    ax.set_title(f"{cfg.STATE_NAMES[si]}{tag}", fontsize=11, fontweight="bold")
    ax.set_xlabel("Time (h)", fontsize=9)
    ax.grid(alpha=0.15)

for k in range(cfg.STATE_NUM, len(axes)):
    axes[k].set_visible(False)

legend = [
    Line2D([0], [0], color="red", lw=1.8, label="Open-loop model"),
    Line2D([0], [0], color="steelblue", lw=2.0, label="EnKF mean"),
    Patch(facecolor="steelblue", alpha=0.30, label=r"$\pm1\sigma$"),
    Patch(facecolor="steelblue", alpha=0.15, label=r"$\pm2\sigma$"),
    Line2D([0], [0], color="darkorange", marker="o", lw=0, markersize=6, label="Measurements"),
]
fig.legend(handles=legend, loc="lower center", ncol=5, fontsize=12, frameon=False,
           bbox_to_anchor=(0.5, -0.005))
fig.suptitle(f"Option B — {DS}, alpha={ALPHA:g}  (all states vs measurements)",
             fontsize=15, fontweight="bold", y=1.005)
plt.tight_layout(rect=[0, 0.03, 1, 1])

out_dir = cfg.PROJECT_ROOT / "results" / "alpha_check" / "figures"
out_dir.mkdir(parents=True, exist_ok=True)
out = out_dir / f"alpha_{ALPHA:g}_{DS}.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out}")
