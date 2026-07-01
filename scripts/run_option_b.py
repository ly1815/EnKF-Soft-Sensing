"""
run_option_b.py
===============
One-command Option-B pipeline:

  1. Sweep the universal additive-noise scalar ALPHA on the tuning dataset (P4)
     and select the value minimising mean NSD NRMSE.
  2. Cross-validate the selected ALPHA on the other datasets (P1-P3).
  3. Re-use the per-dataset EnKF passes at the selected ALPHA to save, for ALL 17
     states, the ensemble MEAN trajectory and STD trajectory (the uncertainty
     band) to results/<run>/pkl/, plus an open-loop model trajectory and a
     summary of the calibration. Per-dataset all-state figures are also written.

Noise model (unchanged design):
  - measured states  -> multiplicative CV noise (config.PROCESS_NOISE_CV)
  - unmeasured states -> additive Q_ii = (ALPHA * scale_i)^2, scale = median
    magnitude (config.PROCESS_NOISE_SCALE); one universal ALPHA for all of them.

Usage:
    ./.venv/Scripts/python.exe scripts/run_option_b.py
    ./.venv/Scripts/python.exe scripts/run_option_b.py --alphas 0.03,0.05,0.07 --run option_b
    ./.venv/Scripts/python.exe scripts/run_option_b.py --fixed-alpha 0.05   # skip sweep
    ./.venv/Scripts/python.exe scripts/run_option_b.py --no-plots
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
p = argparse.ArgumentParser(description="Option-B: calibrate ALPHA, save mean/std bands for all states")
p.add_argument("--alphas", default="0.02,0.03,0.046,0.06,0.1,0.15")
p.add_argument("--fixed-alpha", default=None, type=float, help="Skip sweep, use this ALPHA")
p.add_argument("--tuning-dataset", default="P4")
p.add_argument("--validate", default="P1,P2,P3")
p.add_argument("--ensemble-size", default=cfg.ENSEMBLE_SIZE, type=int)
p.add_argument("--seed", default=42, type=int)
p.add_argument("--run", default="option_b")
p.add_argument("--no-plots", action="store_true")
args = p.parse_args()

ALPHAS = [float(a) for a in args.alphas.split(",")]
TUNE_DS = args.tuning_dataset
VAL_DS = [s for s in args.validate.split(",") if s]
ALL_DS = [TUNE_DS] + [d for d in VAL_DS if d != TUNE_DS]
ENS = args.ensemble_size

RESULTS_DIR = cfg.PROJECT_ROOT / "results" / args.run
PKL_DIR = RESULTS_DIR / "pkl"
FIG_DIR = RESULTS_DIR / "figures"
PKL_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

def save_pkl(obj, name):
    with open(PKL_DIR / name, "wb") as f:
        pickle.dump(obj, f)
    print(f"  saved: {PKL_DIR / name}")

print("=" * 70)
print(f"Option-B pipeline  [run={args.run}]")
print(f"  tune={TUNE_DS}  validate={VAL_DS}  ensemble={ENS}")
print(f"  alphas={ALPHAS}" + (f"  fixed-alpha={args.fixed_alpha}" if args.fixed_alpha else ""))
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
P0_meas = np.array([cfg.MEASUREMENT_NOISE_VAR.get(s, 0.0) for s in cfg.STATE_NAMES])

n_nsd = 7
nsd_state_idx = list(range(cfg.STATE_NUM - n_nsd, cfg.STATE_NUM))
nsd_names = [cfg.STATE_NAMES[i] for i in nsd_state_idx]
T_meas = np.array(cfg.T_MEAS_FIXED)
meas_grid_idx = [min(int(round(t / dt_kf)) + 1, N_kf) for t in T_meas]

volume_results = compute_volume_results(select_datasets(*ALL_DS), cfg.INITIAL_VOLUMES,
                                        build_schedule, step_len)

# Per-dataset static inputs (state_init, measurement ensemble, NSD meas, model traj)
_cache = {}
def ds_data(name):
    if name not in _cache:
        d = load_dataset(name)
        _, x0 = get_initial_condition(d["met_df"], d["nsd_df"])
        np.random.seed(args.seed)
        mens = generate_measurement_ensembles(select_datasets(name), load_dataset,
                                              cfg.MEAS_NUM, ENS, var_meas)[name]
        nsd = pd.DataFrame(d["NSD_meas"]).apply(pd.to_numeric, errors="coerce").to_numpy()
        Fin, Fout, Gf, Uf = build_schedule(name)
        model = simulate_dataset(x0, Fin, Fout, Gf, Uf, volume_results[name][1:],
                                time_grid, step_len, name=name)
        model = np.vstack([x0, model])
        _cache[name] = dict(x0=x0, mens=mens, nsd=nsd, model=model)
    return _cache[name]


def run_pass(name, alpha):
    """One EnKF diagnostic pass -> (mean_traj, std_traj)."""
    dd = ds_data(name)
    var_model = (alpha * scale_vec) ** 2
    P0_diag = var_model.copy()
    P0_diag[:cfg.MEAS_NUM] = P0_meas[:cfg.MEAS_NUM]
    np.random.seed(args.seed)
    _, std_traj, mean_traj = run_enkf_single_with_ensemble_diagnostics(
        dataset_name=name, load_dataset_fn=load_dataset, build_schedule_fn=build_schedule,
        state_init=dd["x0"], volume_results=volume_results,
        set_meas_ens=dd["mens"], T_meas=T_meas,
        state_num=cfg.STATE_NUM, meas_num=cfg.MEAS_NUM, ensemble_size=ENS,
        Q=np.diag(var_model), R=R, H=H, dt_kf=dt_kf, N_kf=N_kf,
        P0=np.diag(P0_diag), process_noise_cv=process_noise_cv,
        no_update_indices=no_update_indices, clip_indices=clip_indices,
    )
    return mean_traj, std_traj


def nsd_metrics(name, mean_traj, std_traj):
    dd = ds_data(name)
    out = {}
    for col, si in enumerate(nsd_state_idx):
        meas = dd["nsd"][:, col]
        m = mean_traj[meas_grid_idx, si]; s = std_traj[meas_grid_idx, si]
        valid = ~np.isnan(meas) & (s > 0)
        if valid.sum() == 0:
            out[nsd_names[col]] = dict(rmse=np.nan, nrmse=np.nan, cov=np.nan, ss=np.nan); continue
        err = meas[valid] - m[valid]
        rmse = np.sqrt(np.mean(err ** 2)); norm = np.mean(np.abs(meas[valid])) or 1.0
        out[nsd_names[col]] = dict(
            rmse=rmse, nrmse=rmse / norm,
            cov=100.0 * np.mean(np.abs(err) <= 2 * s[valid]),
            ss=(np.mean(s[valid]) / rmse if rmse > 0 else np.nan))
    return out

def mean_nrmse(mt):
    return np.nanmean([mt[n]["nrmse"] for n in nsd_names])


# ── 1) Select ALPHA ──────────────────────────────────────────────────────────
traj_store = {}   # (name, alpha) -> (mean, std)
sweep_metrics = {}

if args.fixed_alpha is not None:
    best = args.fixed_alpha
    print(f"\nUsing fixed alpha = {best:g} (sweep skipped)")
    mt, st = run_pass(TUNE_DS, best); traj_store[(TUNE_DS, best)] = (mt, st)
    sweep_metrics[best] = nsd_metrics(TUNE_DS, mt, st)
else:
    print(f"\n### Calibration sweep on {TUNE_DS} ###")
    for a in ALPHAS:
        print(f"  alpha={a:g} ...", flush=True)
        mt, st = run_pass(TUNE_DS, a)
        traj_store[(TUNE_DS, a)] = (mt, st)
        sweep_metrics[a] = nsd_metrics(TUNE_DS, mt, st)

    def show(title, key, fmt, agg=np.nanmean, aggfmt="{:7.3f}"):
        print("\n" + title)
        hdr = f"{'alpha':>7s} | " + " | ".join(f"{n[:9]:>9s}" for n in nsd_names) + " |    AGG"
        print(hdr); print("-" * len(hdr))
        for a in ALPHAS:
            r = sweep_metrics[a]
            print(f"{a:>7g} | " + " | ".join(fmt.format(r[n][key]) for n in nsd_names)
                  + " | " + aggfmt.format(agg([r[n][key] for n in nsd_names])))
    show(f"NRMSE  [{TUNE_DS}, SELECTION METRIC — lower better]", "nrmse", "{:9.3f}")
    show(f"2-sigma coverage %  [{TUNE_DS}]", "cov", "{:9.0f}", aggfmt="{:7.0f}")
    show(f"Spread-skill std/RMSE~1  [{TUNE_DS}]", "ss", "{:9.2f}", np.nanmedian, "{:7.2f}")

    best = min(ALPHAS, key=lambda a: mean_nrmse(sweep_metrics[a]))
    print("\n" + "=" * 70)
    print(f"Selected alpha (min mean NSD NRMSE on {TUNE_DS}): {best:g}  "
          f"(mean NRMSE={mean_nrmse(sweep_metrics[best]):.3f})")
    print("=" * 70)

# ── 2) Cross-validate + gather all-dataset trajectories at best alpha ────────
final_metrics = {TUNE_DS: sweep_metrics[best] if best in sweep_metrics else
                 nsd_metrics(TUNE_DS, *traj_store[(TUNE_DS, best)])}
if VAL_DS:
    print(f"\n### Cross-validation at alpha={best:g} ###")
for name in VAL_DS:
    if name == TUNE_DS:
        continue
    print(f"  {name} ...", flush=True)
    mt, st = run_pass(name, best)
    traj_store[(name, best)] = (mt, st)
    final_metrics[name] = nsd_metrics(name, mt, st)

datasets_saved = [TUNE_DS] + [d for d in VAL_DS if d != TUNE_DS]
print(f"\nNRMSE per dataset at alpha={best:g}")
hdr = f"{'dataset':>8s} | " + " | ".join(f"{n[:9]:>9s}" for n in nsd_names) + " |   MEAN"
print(hdr); print("-" * len(hdr))
for name in datasets_saved:
    r = final_metrics[name]
    print(f"{name:>8s} | " + " | ".join(f"{r[n]['nrmse']:9.3f}" for n in nsd_names)
          + f" | {mean_nrmse(r):6.3f}")
if VAL_DS:
    print(f"\nMean NRMSE  train({TUNE_DS})={mean_nrmse(final_metrics[TUNE_DS]):.3f}  "
          f"validate={np.mean([mean_nrmse(final_metrics[n]) for n in VAL_DS if n in final_metrics]):.3f}")

# ── 3) Save mean + std (uncertainty band) for ALL states, per dataset ───────
print(f"\n### Saving mean/std trajectories (alpha={best:g}) ###")
for name in datasets_saved:
    mt, st = traj_store[(name, best)]
    save_pkl({
        "dataset": name, "alpha": best, "T": T_model,
        "state_names": list(cfg.STATE_NAMES),
        "mean_trajectory": mt,            # (N_kf+1, 17)  ensemble mean
        "std_trajectory": st,             # (N_kf+1, 17)  ensemble std = uncertainty band
        "band_1sigma_lo": np.maximum(mt - st, 0.0),
        "band_1sigma_hi": mt + st,
        "band_2sigma_lo": np.maximum(mt - 2 * st, 0.0),
        "band_2sigma_hi": mt + 2 * st,
        "model_trajectory": ds_data(name)["model"],
    }, f"option_b_{name}.pkl")

save_pkl({
    "alpha_selected": best,
    "alphas_swept": ALPHAS if args.fixed_alpha is None else [best],
    "tuning_dataset": TUNE_DS, "validate_datasets": VAL_DS,
    "scale_median": dict(cfg.PROCESS_NOISE_SCALE),
    "process_noise_cv": dict(cfg.PROCESS_NOISE_CV),
    "clip_states": list(cfg.CLIP_STATES), "no_update_states": list(cfg.NO_UPDATE_STATES),
    "sweep_metrics": sweep_metrics, "final_metrics": final_metrics,
    "ensemble_size": ENS, "seed": args.seed,
}, "option_b_summary.pkl")

# ── 4) Per-dataset all-state figures ─────────────────────────────────────────
if not args.no_plots:
    print("\n### Plotting ###")
    DOWN = 20
    tds = T_model[::DOWN]
    for name in datasets_saved:
        mt, st = traj_store[(name, best)]
        dd = ds_data(name)
        d = load_dataset(name)
        set_meas = d["set_meas"].astype(float); set_err = d["set_meas_errorbar"].astype(float)
        nsd_err = pd.DataFrame(d["NSD_meas_errorbar"]).apply(pd.to_numeric, errors="coerce").to_numpy()
        asn_col = set_meas.shape[1] - 1
        meas_by_state = {i: (set_meas[:, i], set_err[:, i]) for i in range(cfg.MEAS_NUM)}
        meas_by_state[cfg.STATE_NAMES.index("Asn")] = (set_meas[:, asn_col], set_err[:, asn_col])
        for j in range(n_nsd):
            meas_by_state[nsd_state_idx[j]] = (dd["nsd"][:, j], nsd_err[:, j])

        fig, axes = plt.subplots(5, 4, figsize=(20, 15)); axes = axes.flatten()
        for si in range(cfg.STATE_NUM):
            ax = axes[si]
            m = mt[::DOWN, si]; s = st[::DOWN, si]
            ax.fill_between(tds, np.maximum(m - 2 * s, 0), m + 2 * s, color="steelblue", alpha=0.15)
            ax.fill_between(tds, np.maximum(m - s, 0), m + s, color="steelblue", alpha=0.30)
            ax.plot(tds, m, color="steelblue", lw=2.0)
            ax.plot(tds, dd["model"][::DOWN, si], color="red", lw=1.6)
            if si in meas_by_state:
                v, e = meas_by_state[si]
                ax.errorbar(T_meas, v, yerr=e, fmt="o", color="darkorange", markersize=4,
                            capsize=2, elinewidth=1, alpha=0.9, zorder=5)
            tag = "" if si in meas_by_state else "  (no meas)"
            ax.set_title(f"{cfg.STATE_NAMES[si]}{tag}", fontsize=11, fontweight="bold")
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
        role = "train" if name == TUNE_DS else "validate"
        fig.suptitle(f"Option B — {name} ({role}), alpha={best:g}", fontsize=15,
                     fontweight="bold", y=1.005)
        plt.tight_layout(rect=[0, 0.03, 1, 1])
        out = FIG_DIR / f"option_b_{name}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
        print(f"  saved: {out}")

print(f"\nDone. alpha={best:g}. Results in {RESULTS_DIR}")
print(f"To adopt: set PROCESS_NOISE_ALPHA = {best:g} in config.py")
