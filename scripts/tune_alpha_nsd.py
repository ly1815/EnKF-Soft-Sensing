"""
tune_alpha_nsd.py
=================
Sweep the NSD additive process-noise scalar (PROCESS_NOISE_ALPHA) on P4 and show, for each
alpha, the 7 NSD uncertainty bands against the NSD measurements — the NSD analogue of
tune_alpha_asn.py. This is for choosing the NSD alpha by inspection (bands + metrics), NOT
auto-selecting by argmin-NRMSE (cf. run_option_b.py, which auto-picks and plots only the
winner).

Only the 7 NSD states take the swept alpha; the observable Asn/Glu keep their calibrated
config alpha (PROCESS_NOISE_ALPHA_OBS), and the measured states keep their multiplicative
CVs (PROCESS_NOISE_CV). Additive variance:  Q_i = (alpha_nsd * scale_i)^2  for the 7 NSDs.

The 7 NSDs are measured (held out of the update), so each is scored against its
measurements: RMSE, NRMSE (RMSE / mean measurement), 2-sigma coverage, spread-skill
(mean std / RMSE, ideal ~1). Nothing is auto-adopted — prints/plots for inspection.

Outputs (results/<run>/):
  pkl/nsd_alpha_<a>.pkl            : all-7-NSD mean/std trajectory + bands + meas + model
  figures/nsd_bands_alpha_<a>.png  : 7-NSD band grid (mean, bands, model, meas) per alpha
  figures/nsd_metrics_vs_alpha.png : mean NRMSE / mean coverage / median spread-skill vs alpha

Usage (macOS venv):
    mkdir -p results/alpha_nsd
    caffeinate -i ./.venv/bin/python scripts/tune_alpha_nsd.py 2>&1 | tee results/alpha_nsd/sweep.log
    ./.venv/bin/python scripts/tune_alpha_nsd.py --replot         # redraw from existing pkls
"""

import argparse
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
from nsd_enkf.model import compute_volume_results, simulate_dataset
from nsd_enkf.analysis import generate_measurement_ensembles
from nsd_enkf.enkf import run_enkf_single_with_ensemble_diagnostics

# ── CLI ──────────────────────────────────────────────────────────────────────
p = argparse.ArgumentParser(description="Sweep NSD additive-noise alpha on P4; show 7-NSD bands")
p.add_argument("--alphas", default="0.005,0.0075,0.01,0.02,0.03,0.04")
p.add_argument("--dataset", default="P4")
p.add_argument("--ensemble-size", default=cfg.ENSEMBLE_SIZE, type=int)
p.add_argument("--seed", default=42, type=int)
p.add_argument("--run", default="alpha_nsd")
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
print(f"NSD alpha sweep on {DS}  (N={ENS}, alphas={ALPHAS})")
print("Only the 7 NSDs take the swept alpha; Asn/Glu keep config ALPHA_OBS.")
print("Choose by inspection (bands + metrics) — nothing auto-adopted.")
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

scale_vec = np.zeros(cfg.STATE_NUM)
for s, sc in cfg.PROCESS_NOISE_SCALE.items():
    scale_vec[cfg.STATE_NAMES.index(s)] = sc
ALPHA_OBS = getattr(cfg, "PROCESS_NOISE_ALPHA_OBS", 0.0)
obs_idx = [cfg.STATE_NAMES.index(s) for s in getattr(cfg, "ALPHA_OBS_STATES", ["Asn", "Glu"])]

n_nsd = 7
nsd_state_idx = list(range(cfg.STATE_NUM - n_nsd, cfg.STATE_NUM))
nsd_names = [cfg.STATE_NAMES[i] for i in nsd_state_idx]


def var_model_for(alpha_nsd):
    # NSDs take the swept alpha; Asn/Glu keep config ALPHA_OBS; measured states have
    # scale=0 so contribute no additive variance (they use multiplicative CV noise).
    a = np.full(cfg.STATE_NUM, float(alpha_nsd))
    for i in obs_idx:
        a[i] = ALPHA_OBS
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
nsd_meas = pd.DataFrame(d["NSD_meas"]).apply(pd.to_numeric, errors="coerce").to_numpy()
nsd_err = pd.DataFrame(d["NSD_meas_errorbar"]).apply(pd.to_numeric, errors="coerce").to_numpy()
T_meas = np.array(cfg.T_MEAS_FIXED)
meas_grid_idx = [min(int(round(t / dt_kf)) + 1, N_kf) for t in T_meas]   # post-update

# Open-loop model (no assimilation) — identical for every alpha; overlaid for reference.
_Fin, _Fout, _Gf, _Uf = build_schedule(DS)
model_full = np.vstack([x0, simulate_dataset(x0, _Fin, _Fout, _Gf, _Uf,
                                             volume_results[DS][1:], time_grid, step_len,
                                             name=DS)])


def run_pass(alpha_nsd):
    var_model = var_model_for(alpha_nsd)
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


def nsd_metrics(mean_traj, std_traj):
    out = {}
    for col, si in enumerate(nsd_state_idx):
        meas = nsd_meas[:, col]
        m = mean_traj[meas_grid_idx, si]; s = std_traj[meas_grid_idx, si]
        valid = ~np.isnan(meas) & (s > 0)
        if valid.sum() == 0:
            out[nsd_names[col]] = dict(rmse=np.nan, nrmse=np.nan, cov=np.nan, ss=np.nan); continue
        err = meas[valid] - m[valid]
        rmse = float(np.sqrt(np.mean(err ** 2)))
        norm = float(np.mean(np.abs(meas[valid]))) or 1.0
        cov = 100.0 * float(np.mean(np.abs(err) <= 2.0 * s[valid]))
        ss = float(np.mean(s[valid]) / rmse) if rmse > 0 else np.nan
        out[nsd_names[col]] = dict(rmse=rmse, nrmse=rmse / norm, cov=cov, ss=ss)
    return out


# ── Sweep (or reload metrics in --replot) ────────────────────────────────────
results = {}   # alpha -> {nsd_name: metrics}
if args.replot:
    for a in ALPHAS:
        results[a] = pickle.load(open(PKL_DIR / f"nsd_alpha_{a:g}.pkl", "rb"))["metrics"]
    print("Replot mode: metrics loaded from existing pkls (EnKF sweep skipped).")
else:
    for a in ALPHAS:
        print(f"  alpha_nsd={a:g} ...", flush=True)
        mt, st = run_pass(a)
        met = nsd_metrics(mt, st)
        results[a] = met
        payload = {
            "dataset": DS, "alpha_nsd": a, "T": T_model, "nsd_names": nsd_names,
            "metrics": met, "T_meas": T_meas,
        }
        for col, si in enumerate(nsd_state_idx):
            nm = nsd_names[col]
            payload[nm] = {
                "mean_trajectory": mt[:, si], "std_trajectory": st[:, si],
                "band_1sigma_lo": np.maximum(mt[:, si] - st[:, si], 0.0),
                "band_1sigma_hi": mt[:, si] + st[:, si],
                "band_2sigma_lo": np.maximum(mt[:, si] - 2 * st[:, si], 0.0),
                "band_2sigma_hi": mt[:, si] + 2 * st[:, si],
                "model_trajectory": model_full[:, si],
                "meas": nsd_meas[:, col], "err": nsd_err[:, col],
            }
        with open(PKL_DIR / f"nsd_alpha_{a:g}.pkl", "wb") as f:
            pickle.dump(payload, f)
        del mt, st


# ── Tables ───────────────────────────────────────────────────────────────
def print_table(title, key, fmt, agg=np.nanmean, aggfmt="{:7.3f}"):
    print("\n" + title)
    hdr = f"{'alpha':>7} | " + " | ".join(f"{n[:9]:>9}" for n in nsd_names) + " |     AGG"
    print(hdr); print("-" * len(hdr))
    for a in ALPHAS:
        r = results[a]
        row = f"{a:>7g} | " + " | ".join(fmt.format(r[n][key]) for n in nsd_names)
        row += " | " + aggfmt.format(agg([r[n][key] for n in nsd_names]))
        print(row)

print_table("NRMSE per NSD (RMSE / mean meas; lower better)", "nrmse", "{:9.3f}")
print_table("2-sigma coverage % per NSD (target ~95)", "cov", "{:9.0f}", aggfmt="{:7.0f}")
print_table("spread-skill std/RMSE per NSD (target ~1)", "ss", "{:9.2f}", np.nanmedian, "{:7.2f}")

mean_nrmse = {a: np.nanmean([results[a][n]["nrmse"] for n in nsd_names]) for a in ALPHAS}
best = min(ALPHAS, key=lambda a: mean_nrmse[a])
print(f"\nmin mean-NRMSE alpha = {best:g} (mean NRMSE = {mean_nrmse[best]:.3f}) — "
      f"reference only, choose by inspecting the bands.")

# ── Plots ────────────────────────────────────────────────────────────────
if not args.no_plots:
    DOWN = max(int(args.traj_down), 1)
    tds = T_model[::DOWN]

    # (1) per-alpha: 7-NSD band grid
    for a in ALPHAS:
        dd = pickle.load(open(PKL_DIR / f"nsd_alpha_{a:g}.pkl", "rb"))
        fig, axes = plt.subplots(2, 4, figsize=(20, 8)); axes = axes.flatten()
        for col, nm in enumerate(nsd_names):
            ax = axes[col]; b = dd[nm]
            ax.fill_between(tds, b["band_2sigma_lo"][::DOWN], b["band_2sigma_hi"][::DOWN],
                            color="steelblue", alpha=0.15)
            ax.fill_between(tds, b["band_1sigma_lo"][::DOWN], b["band_1sigma_hi"][::DOWN],
                            color="steelblue", alpha=0.30)
            ax.plot(tds, b["mean_trajectory"][::DOWN], color="steelblue", lw=2)
            ax.plot(tds, b["model_trajectory"][::DOWN], color="red", lw=1.6)
            ax.errorbar(T_meas, b["meas"], yerr=b["err"], fmt="o", color="darkorange",
                        ms=4, capsize=2, elinewidth=1, zorder=5)
            r = results[a][nm]
            ax.set_title(f"{nm}  NRMSE={r['nrmse']:.2f} cov={r['cov']:.0f}% ss={r['ss']:.2f}",
                         fontsize=10, fontweight="bold")
            ax.set_xlabel("Time (h)"); ax.set_ylabel("mM"); ax.grid(alpha=0.15)
        for k in range(n_nsd, len(axes)):
            axes[k].set_visible(False)
        fig.suptitle(f"NSD bands at alpha_nsd={a:g} — {DS}", fontsize=15, fontweight="bold")
        fig.legend(handles=[
            Line2D([0], [0], color="red", lw=1.6, label="Open-loop model"),
            Line2D([0], [0], color="steelblue", lw=2, label="EnKF mean"),
            Patch(facecolor="steelblue", alpha=0.30, label=r"$\pm1\sigma$"),
            Patch(facecolor="steelblue", alpha=0.15, label=r"$\pm2\sigma$"),
            Line2D([0], [0], color="darkorange", marker="o", lw=0, ms=6, label="Measurements"),
        ], loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.02))
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        fig.savefig(FIG_DIR / f"nsd_bands_alpha_{a:g}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"saved: {FIG_DIR / f'nsd_bands_alpha_{a:g}.png'}")

    # (2) summary metrics vs alpha (aggregated over the 7 NSDs)
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    al = ALPHAS
    axs[0].plot(al, [mean_nrmse[a] for a in al], "o-", color="tab:red")
    axs[0].set_title("mean NSD NRMSE (lower better)")
    axs[1].plot(al, [np.nanmean([results[a][n]["cov"] for n in nsd_names]) for a in al],
                "s-", color="tab:blue")
    axs[1].axhline(95, ls=":", color="gray"); axs[1].set_title("mean 2σ coverage % (target ~95)")
    axs[2].plot(al, [np.nanmedian([results[a][n]["ss"] for n in nsd_names]) for a in al],
                "^-", color="tab:green")
    axs[2].axhline(1.0, ls=":", color="gray"); axs[2].set_title("median spread-skill (target ~1)")
    for ax in axs:
        ax.set_xlabel("alpha_nsd"); ax.set_xscale("log"); ax.grid(alpha=0.2)
    fig.suptitle(f"NSD metrics vs alpha (mean over 7 NSDs) — {DS}", fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(FIG_DIR / "nsd_metrics_vs_alpha.png", dpi=150); plt.close(fig)
    print(f"saved: {FIG_DIR / 'nsd_metrics_vs_alpha.png'}")

print(f"\nDone. Results in {RESULTS_DIR}")
