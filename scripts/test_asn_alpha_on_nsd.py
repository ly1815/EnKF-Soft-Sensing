"""
test_asn_alpha_on_nsd.py  —  does a LOOSER Asn/Glu (bigger alpha_obs) improve the NSDs?
========================================================================================
HYPOTHESIS (quick test, not part of the tuning pipeline): Asn/Glu are unmeasured but
UPSTREAM of the NSD pathway. Injecting more process noise into them (bigger alpha_obs)
lets them move more, which — through the shared dynamics / cross-covariance — might
improve the downstream NSD estimates even though the NSDs' own alpha_nsd is unchanged.

This runs 4 quick EnKF passes (one per dataset P1-P4) at:
    alpha_obs = 0.01   (loose Asn/Glu)      [--alpha-obs to change]
    alpha_nsd = 0.02   (held fixed)         [--alpha-nsd to change]
    measured CVs      = each fold's CALIBRATED values from results_single_sweep/fold_*/cv/cv_final.json

and compares the resulting NSD metrics to the existing baseline sweep pkl
    results_single_sweep/fold_<P>/alpha_nsd/pkl/alpha_0.02.pkl   (same CVs, same alpha_nsd, alpha_obs=0.002)
so the ONLY thing that differs between test and baseline is alpha_obs.

Self-contained: reuses nsd_enkf exactly as 04_cross_validate.py does (same seed, same
matrices), but touches none of the numbered scripts. Saves everything (trajectory + std +
bands + metrics pkl per dataset, NSD paper-style figure, comparison table) under
    results_asn_alpha0.01/

Usage (macOS venv):
    ./.venv/bin/python scripts/test_asn_alpha_on_nsd.py
    ./.venv/bin/python scripts/test_asn_alpha_on_nsd.py --alpha-obs 0.01 --alpha-nsd 0.02
"""

import argparse
import json
import os
import pickle
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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

# ── CLI ────────────────────────────────────────────────────────────────────────
p = argparse.ArgumentParser(description="Quick test: does bigger alpha_obs improve the NSDs?")
p.add_argument("--datasets", default="P1,P2,P3,P4")
p.add_argument("--alpha-obs", default=0.01, type=float)
p.add_argument("--alpha-nsd", default=0.02, type=float)
p.add_argument("--cv-run", default="results_single_sweep", help="run dir holding fold_*/cv/cv_final.json")
p.add_argument("--out", default="results_asn_alpha0.01")
p.add_argument("--ensemble-size", default=cfg.ENSEMBLE_SIZE, type=int)
p.add_argument("--seed", default=42, type=int)
p.add_argument("--traj-down", default=10, type=int)
p.add_argument("--dpi", default=200, type=int)
p.add_argument("--no-plots", action="store_true")
args = p.parse_args()

DATASETS = [d for d in args.datasets.split(",") if d]
ENS = args.ensemble_size
A_OBS = args.alpha_obs
A_NSD = args.alpha_nsd
CV_RUN = cfg.PROJECT_ROOT / args.cv_run if not Path(args.cv_run).is_absolute() else Path(args.cv_run)
OUT = cfg.PROJECT_ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out)
DOWN = max(int(args.traj_down), 1)

# ── Fixed grids / matrices (identical to 04_cross_validate.py) ──────────────────
meas_names = cfg.MEASURED_STATES
MEAS = cfg.MEAS_NUM
n_nsd = 7
nsd_state_idx = list(range(cfg.STATE_NUM - n_nsd, cfg.STATE_NUM))
nsd_names = [cfg.STATE_NAMES[i] for i in nsd_state_idx]
ASN = cfg.STATE_NAMES.index("Asn")
AX = cfg.AXIS_NAMES

time_grid = np.arange(cfg.DT, cfg.T_END + cfg.DT, cfg.DT)
step_len = np.full(len(time_grid), cfg.DT)
N_model = int(cfg.T_END / cfg.DT)
T_model = np.linspace(0, cfg.T_END, N_model + 1)
dt_kf = cfg.DT
N_kf = len(T_model) - 1
var_meas = np.array(list(cfg.MEASUREMENT_NOISE_VAR.values()))
R = np.diag(var_meas[:MEAS])
H = np.hstack((np.eye(MEAS), np.zeros((MEAS, cfg.STATE_NUM - MEAS))))
no_update_indices = {cfg.STATE_NAMES.index(s) for s in cfg.NO_UPDATE_STATES}
clip_indices = {cfg.STATE_NAMES.index(s) for s in cfg.CLIP_STATES}
scale_vec = np.zeros(cfg.STATE_NUM)
for s, sc in cfg.PROCESS_NOISE_SCALE.items():
    scale_vec[cfg.STATE_NAMES.index(s)] = sc
obs_idx = [cfg.STATE_NAMES.index(s) for s in getattr(cfg, "ALPHA_OBS_STATES", ["Asn", "Glu"])]
P0_meas = np.array([cfg.MEASUREMENT_NOISE_VAR.get(s, 0.0) for s in cfg.STATE_NAMES])
T_meas = np.array(cfg.T_MEAS_FIXED)
meas_grid_idx = [min(int(round(t / dt_kf)) + 1, N_kf) for t in T_meas]
volume_results = compute_volume_results(select_datasets(*DATASETS), cfg.INITIAL_VOLUMES,
                                        build_schedule, step_len)


def build_Q(alpha_obs, alpha_nsd):
    a = np.full(cfg.STATE_NUM, float(alpha_nsd))
    for i in obs_idx:
        a[i] = float(alpha_obs)
    return np.diag((a * scale_vec) ** 2)


def P0_from(Q):
    d = np.diag(Q).copy(); d[:MEAS] = P0_meas[:MEAS]
    return np.diag(d)


_cache = {}
def ds_data(name):
    if name not in _cache:
        d = load_dataset(name)
        _, x0 = get_initial_condition(d["met_df"], d["nsd_df"])
        sm = d["set_meas"].astype(float); se = d["set_meas_errorbar"].astype(float)
        asn_col = sm.shape[1] - 1
        np.random.seed(args.seed)
        mens = generate_measurement_ensembles(select_datasets(name), load_dataset,
                                              MEAS, ENS, var_meas)[name]
        Fin, Fout, Gf, Uf = build_schedule(name)
        model = np.vstack([x0, simulate_dataset(x0, Fin, Fout, Gf, Uf,
                                                volume_results[name][1:], time_grid, step_len,
                                                name=name)])
        _cache[name] = dict(
            x0=x0, set_meas=sm[:, :MEAS], asn_meas=sm[:, asn_col], asn_err=se[:, asn_col],
            set_err=se, mens=mens, model=model,
            nsd_meas=pd.DataFrame(d["NSD_meas"]).apply(pd.to_numeric, errors="coerce").to_numpy(),
            nsd_err=pd.DataFrame(d["NSD_meas_errorbar"]).apply(pd.to_numeric, errors="coerce").to_numpy())
    return _cache[name]


def enkf_pass(name, cv_idx, Q, P0):
    np.random.seed(args.seed)
    _, std_traj, mean_traj = run_enkf_single_with_ensemble_diagnostics(
        dataset_name=name, load_dataset_fn=load_dataset, build_schedule_fn=build_schedule,
        state_init=ds_data(name)["x0"], volume_results=volume_results,
        set_meas_ens=ds_data(name)["mens"], T_meas=T_meas,
        state_num=cfg.STATE_NUM, meas_num=MEAS, ensemble_size=ENS,
        Q=Q, R=R, H=H, dt_kf=dt_kf, N_kf=N_kf, P0=P0,
        process_noise_cv=dict(cv_idx), no_update_indices=set(no_update_indices),
        clip_indices=set(clip_indices))
    return mean_traj, std_traj


def nsd_asn_metrics(name, mean_traj, std_traj):
    dd = ds_data(name); out = {}
    for col, si in enumerate(nsd_state_idx):
        meas = dd["nsd_meas"][:, col]; m = mean_traj[meas_grid_idx, si]; s = std_traj[meas_grid_idx, si]
        valid = ~np.isnan(meas) & (s > 0)
        if valid.sum() == 0:
            out[nsd_names[col]] = dict(rmse=np.nan, nrmse=np.nan, cov=np.nan, ss=np.nan); continue
        err = meas[valid] - m[valid]; rmse = float(np.sqrt(np.mean(err ** 2)))
        norm = float(np.mean(np.abs(meas[valid]))) or 1.0
        out[nsd_names[col]] = dict(rmse=rmse, nrmse=rmse / norm,
                                   cov=100.0 * float(np.mean(np.abs(err) <= 2 * s[valid])),
                                   ss=(float(np.mean(s[valid]) / rmse) if rmse > 0 else np.nan))
    meas = dd["asn_meas"]; m = mean_traj[meas_grid_idx, ASN]; s = std_traj[meas_grid_idx, ASN]
    valid = ~np.isnan(meas) & (s > 0); err = meas[valid] - m[valid]
    rmse = float(np.sqrt(np.mean(err ** 2))); norm = float(np.mean(np.abs(meas[valid]))) or 1.0
    out["Asn"] = dict(rmse=rmse, nrmse=rmse / norm,
                      cov=100.0 * float(np.mean(np.abs(err) <= 2 * s[valid])),
                      ss=(float(np.mean(s[valid]) / rmse) if rmse > 0 else np.nan))
    return out


def save_pkl(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(obj, f)
    os.replace(tmp, path)


# ── paper-style NSD figure (mirrors nsd_enkf.plotting / plot_v1_sweeps) ─────────
def _style_ax(ax):
    ax.grid(alpha=0.15); ax.tick_params(axis="both", labelsize=10)
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_fontweight("bold")
    for sp in ax.spines.values():
        sp.set_linewidth(1.5)

LEGEND = [
    Line2D([0], [0], color="red", lw=1.8, label="Mechanistic model"),
    Line2D([0], [0], color="steelblue", lw=2.0, label="EnKF mean"),
    Patch(facecolor="steelblue", alpha=0.30, label=r"EnKF $\pm 1\sigma$"),
    Patch(facecolor="steelblue", alpha=0.15, label=r"EnKF $\pm 2\sigma$"),
    Line2D([0], [0], color="darkorange", marker="o", lw=0, markersize=5, label="Measurements"),
]

def nsd_figure(name, mt, st, model, out):
    t = T_model[::DOWN]; dd = ds_data(name)
    fig, axes = plt.subplots(2, 4, figsize=(5.0 * 4, 3.4 * 2)); axes = axes.flatten()
    for k, (nm, si) in enumerate(zip(nsd_names, nsd_state_idx)):
        ax = axes[k]; m = mt[::DOWN, si]; s = st[::DOWN, si]
        ax.fill_between(t, np.maximum(m - 2 * s, 0), m + 2 * s, color="steelblue", alpha=0.15)
        ax.fill_between(t, np.maximum(m - s, 0), m + s, color="steelblue", alpha=0.30)
        ax.plot(t, m, color="steelblue", lw=2.0)
        ax.plot(t, model[::DOWN, si], color="red", lw=1.8)
        v = dd["nsd_meas"][:, k]; e = dd["nsd_err"][:, k]
        ax.errorbar(T_meas, v, yerr=e, fmt="o", color="darkorange", markersize=4.0,
                    capsize=2, elinewidth=1, alpha=0.9, zorder=5)
        ax.set_ylabel(AX[si], fontsize=10, fontweight="bold")
        ax.set_xlabel("Time (hours)", fontsize=11, fontweight="bold")
        ax.set_title(nm, fontsize=13, fontweight="bold"); _style_ax(ax)
    axes[7].set_visible(False)
    fig.legend(handles=LEGEND, loc="lower center", ncol=5, fontsize=11, frameon=False,
               bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(rf"NSDs — {name},  $\alpha_{{obs}}$ = {A_OBS:g},  $\alpha_{{nsd}}$ = {A_NSD:g}",
                 fontsize=15, fontweight="bold")
    plt.tight_layout(rect=[0, 0.04, 1, 0.99])
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight"); plt.close(fig)


# ── run ─────────────────────────────────────────────────────────────────────────
print("=" * 78)
print(f"TEST: alpha_obs={A_OBS:g}  alpha_nsd={A_NSD:g}  N={ENS}  -> {OUT.name}")
print(f"baseline for comparison: {CV_RUN.name}/fold_*/alpha_nsd/pkl/alpha_{A_NSD:g}.pkl "
      f"(same CVs, alpha_nsd={A_NSD:g}, alpha_obs=0.002)")
print("=" * 78)

Q = build_Q(A_OBS, A_NSD); P0 = P0_from(Q)
rows = {}
for name in DATASETS:
    cv_json = CV_RUN / f"fold_{name}" / "cv" / "cv_final.json"
    if not cv_json.exists():
        raise SystemExit(f"missing calibrated CVs: {cv_json} — run 04 --stage sweep --train {name} first")
    cv = json.load(open(cv_json))["cv"]
    cv_idx = {cfg.STATE_NAMES.index(s): cv[s] for s in meas_names}
    print(f"\n[{name}] calibrated CVs loaded; running EnKF (alpha_obs={A_OBS:g}, alpha_nsd={A_NSD:g}) ...",
          flush=True)
    mt, st = enkf_pass(name, cv_idx, Q, P0)
    met = nsd_asn_metrics(name, mt, st)
    save_pkl({"dataset": name, "alpha_obs": A_OBS, "alpha_nsd": A_NSD, "cv": cv, "T": T_model,
              "state_names": list(cfg.STATE_NAMES),
              "mean_trajectory": mt, "std_trajectory": st,
              "band_2sigma_lo": np.maximum(mt - 2 * st, 0.0), "band_2sigma_hi": mt + 2 * st,
              "model_trajectory": ds_data(name)["model"], "metrics": met},
             OUT / "pkl" / f"{name}.pkl")
    if not args.no_plots:
        nsd_figure(name, mt, st, ds_data(name)["model"], OUT / "figures" / f"nsd_{name}.png")

    # baseline (alpha_obs=0.002) from the existing sweep
    base_pkl = CV_RUN / f"fold_{name}" / "alpha_nsd" / "pkl" / f"alpha_{A_NSD:g}.pkl"
    base_met = pickle.load(open(base_pkl, "rb"))["metrics"] if base_pkl.exists() else None
    rows[name] = (met, base_met)
    print(f"[{name}] done -> {OUT / 'pkl' / (name + '.pkl')}", flush=True)

# ── comparison table ────────────────────────────────────────────────────────────
def mean_nsd(met, key):
    return float(np.nanmean([met[n][key] for n in nsd_names]))

print("\n" + "=" * 78)
print(f"NSD RESULT:  TEST (alpha_obs={A_OBS:g})  vs  BASELINE (alpha_obs=0.002),  alpha_nsd={A_NSD:g}")
print("=" * 78)
hdr = f"{'dataset':8} | {'NRMSE test/base':>18} | {'cov% test/base':>18} | {'spread-skill test/base':>24}"
print(hdr); print("-" * len(hdr))
agg = {"nrmse": [[], []], "cov": [[], []], "ss": [[], []]}
for name in DATASETS:
    met, base = rows[name]
    tn, tc, ts = mean_nsd(met, "nrmse"), mean_nsd(met, "cov"), mean_nsd(met, "ss")
    if base is not None:
        bn, bc, bs = mean_nsd(base, "nrmse"), mean_nsd(base, "cov"), mean_nsd(base, "ss")
        agg["nrmse"][0].append(tn); agg["nrmse"][1].append(bn)
        agg["cov"][0].append(tc); agg["cov"][1].append(bc)
        agg["ss"][0].append(ts); agg["ss"][1].append(bs)
        print(f"{name:8} | {tn:7.3f} / {bn:<7.3f}  | {tc:7.1f} / {bc:<7.1f}  | {ts:9.3f} / {bs:<9.3f}")
    else:
        print(f"{name:8} | {tn:7.3f} / {'--':<7}  | {tc:7.1f} / {'--':<7}  | {ts:9.3f} / {'--':<9}  (no baseline pkl)")
if agg["nrmse"][0]:
    print("-" * len(hdr))
    print(f"{'MEAN':8} | {np.mean(agg['nrmse'][0]):7.3f} / {np.mean(agg['nrmse'][1]):<7.3f}  "
          f"| {np.mean(agg['cov'][0]):7.1f} / {np.mean(agg['cov'][1]):<7.1f}  "
          f"| {np.mean(agg['ss'][0]):9.3f} / {np.mean(agg['ss'][1]):<9.3f}")

# per-NSD detail (test only) so structural states (UDPGalNAc / GDPMan) are visible
print("\nPer-NSD coverage% (test):")
for name in DATASETS:
    met = rows[name][0]
    print(f"  {name}: " + "  ".join(f"{n}={met[n]['cov']:.0f}" for n in nsd_names))

save_pkl({"alpha_obs": A_OBS, "alpha_nsd": A_NSD,
          "test": {n: rows[n][0] for n in DATASETS},
          "baseline": {n: rows[n][1] for n in DATASETS}},
         OUT / "comparison_metrics.pkl")
print(f"\nSaved metrics + trajectories + figures under: {OUT}")
print("Done.")
