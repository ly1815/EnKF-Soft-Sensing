"""
02_tune_alpha_asn.py  —  Stage 4a: observable-tier alpha (Asn & Glu)
====================================================================
Sweep the shared observable-tier additive-noise alpha (Asn & Glu) on P4, scored on Asn,
and uncertainty diagnostics. This is a focused, Asn-only look — the NSD alpha is NOT
swept here because the NSD pathway is DOWNSTREAM of Asn (NSDs do not feed back into Asn
dynamics), so Asn's calibration is independent of the NSD alpha. NSDs keep their config
alpha; the two observable states are ONE tier (config PROCESS_NOISE_ALPHA_OBS), so Asn AND
Glu both take the swept alpha:

    Q_i = (alpha_obs * scale_i)^2      for i in {Asn, Glu}

Only Asn is scored (Glu is never measured). Asn is measured but held out of the update,
so we score directly against the Asn
measurements: RMSE, NRMSE (RMSE / mean measurement), 2-sigma coverage, and spread-skill
(mean std / RMSE, ideal ~1). Measured states use the adopted multiplicative CVs
(config.PROCESS_NOISE_CV). Nothing is auto-adopted — this prints/plots for inspection.

Outputs (results/<run>/):
  pkl/asn_alpha_<a>.pkl              : Asn mean/std trajectory + bands + measurements
  figures/asn_alpha_trajectories.png : Asn mean +/-1/2 sigma per alpha vs measurements
  figures/asn_metrics_vs_alpha.png   : NRMSE / coverage / spread-skill vs alpha

Usage (macOS venv):
    caffeinate -i ./.venv/bin/python scripts/02_tune_alpha_asn.py
    ./.venv/bin/python scripts/02_tune_alpha_asn.py --alphas 0.001,0.002,0.004,0.006,0.008,0.01
"""

import argparse
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import warnings
import numpy as np
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
p = argparse.ArgumentParser(description="Sweep Asn additive-noise alpha on P4 (Asn only)")
p.add_argument("--alphas", default="0.001,0.002,0.004,0.006,0.008,0.01")
p.add_argument("--dataset", default="P4")
p.add_argument("--ensemble-size", default=cfg.ENSEMBLE_SIZE, type=int)
p.add_argument("--seed", default=42, type=int)
p.add_argument("--run", default="alpha_asn")
p.add_argument("--traj-down", default=20, type=int)
p.add_argument("--no-plots", action="store_true")
p.add_argument("--replot", action="store_true",
               help="skip the EnKF sweep; regenerate figures from existing pkls (fast)")
args = p.parse_args()

ALPHAS = [float(a) for a in args.alphas.split(",")]
DS = args.dataset
ENS = args.ensemble_size

RESULTS_DIR = cfg.PROJECT_ROOT / "results" / args.run
PKL_DIR = RESULTS_DIR / "pkl"; FIG_DIR = RESULTS_DIR / "figures"
PKL_DIR.mkdir(parents=True, exist_ok=True); FIG_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print(f"Observable-tier alpha sweep on {DS}  (N={ENS}, alphas={ALPHAS})")
print("Asn AND Glu share the swept alpha; NSD alpha held fixed (downstream of Asn).")
print("Only Asn is scored (Glu has no measurements).")
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

# NSDs stay at the config NSD alpha (they are downstream of Asn — no feedback). The two
# observable states are one tier (config PROCESS_NOISE_ALPHA_OBS), so Asn AND Glu both take
# the swept alpha. Only Asn is scored below (Glu is never measured).
scale_vec = np.zeros(cfg.STATE_NUM)
for s, sc in cfg.PROCESS_NOISE_SCALE.items():
    scale_vec[cfg.STATE_NAMES.index(s)] = sc
obs_idx = [cfg.STATE_NAMES.index(s) for s in getattr(cfg, "ALPHA_OBS_STATES", ["Asn", "Glu"])]
ASN = cfg.STATE_NAMES.index("Asn")

def var_model_for(alpha_obs):
    a = np.full(cfg.STATE_NUM, cfg.PROCESS_NOISE_ALPHA)   # NSDs at config alpha
    for i in obs_idx:                                     # Asn AND Glu share the swept alpha
        a[i] = alpha_obs
    return (a * scale_vec) ** 2

P0_meas = np.array([cfg.MEASUREMENT_NOISE_VAR.get(s, 0.0) for s in cfg.STATE_NAMES])

# ── Data (P4) ────────────────────────────────────────────────────────────────
volume_results = compute_volume_results(select_datasets(DS), cfg.INITIAL_VOLUMES,
                                        build_schedule, step_len)
d = load_dataset(DS)
_, x0 = get_initial_condition(d["met_df"], d["nsd_df"])
np.random.seed(args.seed)
mens = generate_measurement_ensembles(select_datasets(DS), load_dataset,
                                      cfg.MEAS_NUM, ENS, var_meas)[DS]
set_meas = d["set_meas"].astype(float)
set_err = d["set_meas_errorbar"].astype(float)
asn_col = set_meas.shape[1] - 1            # Asn is the last measured column in the data
asn_meas = set_meas[:, asn_col]
asn_err = set_err[:, asn_col]
T_meas = np.array(cfg.T_MEAS_FIXED)
meas_grid_idx = [min(int(round(t / dt_kf)) + 1, N_kf) for t in T_meas]   # post-update

# Open-loop model (no assimilation) — identical for every alpha; overlaid for reference.
_Fin, _Fout, _Gf, _Uf = build_schedule(DS)
asn_model = np.vstack([x0, simulate_dataset(x0, _Fin, _Fout, _Gf, _Uf,
                                            volume_results[DS][1:], time_grid, step_len,
                                            name=DS)])[:, ASN]


def run_pass(alpha_obs):
    var_model = var_model_for(alpha_obs)
    P0_diag = var_model.copy()
    P0_diag[:cfg.MEAS_NUM] = P0_meas[:cfg.MEAS_NUM]
    np.random.seed(args.seed)
    _, std_traj, mean_traj = run_enkf_single_with_ensemble_diagnostics(
        dataset_name=DS, load_dataset_fn=load_dataset, build_schedule_fn=build_schedule,
        state_init=x0, volume_results=volume_results, set_meas_ens=mens, T_meas=T_meas,
        state_num=cfg.STATE_NUM, meas_num=cfg.MEAS_NUM, ensemble_size=ENS,
        Q=np.diag(var_model), R=R, H=H, dt_kf=dt_kf, N_kf=N_kf,
        P0=np.diag(P0_diag), process_noise_cv=process_noise_cv,
        no_update_indices=no_update_indices, clip_indices=clip_indices,
    )
    return mean_traj, std_traj


def asn_metrics(mean_traj, std_traj):
    m = mean_traj[meas_grid_idx, ASN]
    s = std_traj[meas_grid_idx, ASN]
    valid = ~np.isnan(asn_meas) & (s > 0)
    err = asn_meas[valid] - m[valid]
    rmse = float(np.sqrt(np.mean(err ** 2)))
    norm = float(np.mean(np.abs(asn_meas[valid]))) or 1.0
    cov = 100.0 * float(np.mean(np.abs(err) <= 2.0 * s[valid]))
    ss = float(np.mean(s[valid]) / rmse) if rmse > 0 else np.nan
    return dict(rmse=rmse, nrmse=rmse / norm, cov=cov, ss=ss)


# ── Sweep (or reload metrics in --replot) ────────────────────────────────────
results = {}
if args.replot:
    for a in ALPHAS:
        results[a] = pickle.load(open(PKL_DIR / f"asn_alpha_{a:g}.pkl", "rb"))["metrics"]
    print("Replot mode: metrics loaded from existing pkls (EnKF sweep skipped).")
else:
    for a in ALPHAS:
        print(f"  alpha_obs={a:g} (Asn & Glu) ...", flush=True)
        mt, st = run_pass(a)
        met = asn_metrics(mt, st)
        results[a] = met
        with open(PKL_DIR / f"asn_alpha_{a:g}.pkl", "wb") as f:
            pickle.dump({
                "dataset": DS, "alpha_obs": a, "state": "Asn", "T": T_model,
                "mean_trajectory": mt[:, ASN], "std_trajectory": st[:, ASN],
                "band_1sigma_lo": np.maximum(mt[:, ASN] - st[:, ASN], 0.0),
                "band_1sigma_hi": mt[:, ASN] + st[:, ASN],
                "band_2sigma_lo": np.maximum(mt[:, ASN] - 2 * st[:, ASN], 0.0),
                "band_2sigma_hi": mt[:, ASN] + 2 * st[:, ASN],
                "model_trajectory": asn_model,
                "asn_meas": asn_meas, "asn_err": asn_err, "T_meas": T_meas,
                "metrics": met,
            }, f)
        del mt, st

# ── Table ──────────────────────────────────────────────────────────────────
print(f"\nAsn metrics vs alpha ({DS}):")
hdr = f"{'alpha':>8} | {'RMSE':>8} | {'NRMSE':>8} | {'cov2s%':>7} | {'ss=std/RMSE':>12}"
print(hdr); print("-" * len(hdr))
for a in ALPHAS:
    r = results[a]
    print(f"{a:>8g} | {r['rmse']:8.3f} | {r['nrmse']:8.3f} | {r['cov']:7.0f} | {r['ss']:12.2f}")
best = min(ALPHAS, key=lambda a: results[a]["nrmse"])
print(f"\nmin-NRMSE alpha = {best:g}  "
      f"(NRMSE={results[best]['nrmse']:.3f}, cov={results[best]['cov']:.0f}%, ss={results[best]['ss']:.2f})")
print("Reported for inspection — nothing adopted automatically.")

# ── Plots ────────────────────────────────────────────────────────────────
if not args.no_plots:
    DOWN = max(int(args.traj_down), 1)
    tds = T_model[::DOWN]

    # (1) Asn trajectory + bands per alpha, with measurements
    ncol = 3
    nrow = int(np.ceil(len(ALPHAS) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 3.4 * nrow))
    axes = np.atleast_1d(axes).flatten()
    for k, a in enumerate(ALPHAS):
        ax = axes[k]
        dd = pickle.load(open(PKL_DIR / f"asn_alpha_{a:g}.pkl", "rb"))
        ax.fill_between(tds, dd["band_2sigma_lo"][::DOWN], dd["band_2sigma_hi"][::DOWN],
                        color="steelblue", alpha=0.15)
        ax.fill_between(tds, dd["band_1sigma_lo"][::DOWN], dd["band_1sigma_hi"][::DOWN],
                        color="steelblue", alpha=0.30)
        ax.plot(tds, dd["mean_trajectory"][::DOWN], color="steelblue", lw=2)
        ax.plot(tds, asn_model[::DOWN], color="red", lw=1.6)
        ax.errorbar(T_meas, asn_meas, yerr=asn_err, fmt="o", color="darkorange",
                    ms=4, capsize=2, elinewidth=1, zorder=5)
        r = results[a]
        ax.set_title(f"alpha={a:g}  NRMSE={r['nrmse']:.2f} cov={r['cov']:.0f}% ss={r['ss']:.2f}",
                     fontsize=10)
        ax.set_xlabel("Time (h)"); ax.set_ylabel("Asn (mM)"); ax.grid(alpha=0.15)
    for k in range(len(ALPHAS), len(axes)):
        axes[k].set_visible(False)
    fig.suptitle(f"Asn additive-noise alpha sweep — {DS}", fontsize=14, fontweight="bold")
    fig.legend(handles=[
        Line2D([0], [0], color="red", lw=1.6, label="Open-loop model"),
        Line2D([0], [0], color="steelblue", lw=2, label="EnKF mean"),
        Patch(facecolor="steelblue", alpha=0.30, label=r"$\pm1\sigma$"),
        Patch(facecolor="steelblue", alpha=0.15, label=r"$\pm2\sigma$"),
        Line2D([0], [0], color="darkorange", marker="o", lw=0, ms=6, label="Asn measurements"),
    ], loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.02))
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    fig.savefig(FIG_DIR / "asn_alpha_trajectories.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {FIG_DIR / 'asn_alpha_trajectories.png'}")

    # (2) metrics vs alpha
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    al = ALPHAS
    axs[0].plot(al, [results[a]["nrmse"] for a in al], "o-", color="tab:red")
    axs[0].set_title("NRMSE (lower better)")
    axs[1].plot(al, [results[a]["cov"] for a in al], "s-", color="tab:blue")
    axs[1].axhline(95, ls=":", color="gray"); axs[1].set_title("2σ coverage % (target ~95)")
    axs[2].plot(al, [results[a]["ss"] for a in al], "^-", color="tab:green")
    axs[2].axhline(1.0, ls=":", color="gray"); axs[2].set_title("spread-skill std/RMSE (target ~1)")
    for ax in axs:
        ax.set_xlabel("alpha_obs (Asn & Glu)"); ax.set_xscale("log"); ax.grid(alpha=0.2)
    fig.suptitle(f"Asn metrics vs alpha — {DS}", fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(FIG_DIR / "asn_metrics_vs_alpha.png", dpi=150)
    plt.close(fig)
    print(f"saved: {FIG_DIR / 'asn_metrics_vs_alpha.png'}")

print(f"\nDone. Results in {RESULTS_DIR}")
