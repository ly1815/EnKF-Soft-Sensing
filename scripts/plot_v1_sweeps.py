"""
plot_v1_sweeps.py  —  one paper-style figure per swept alpha (mean + shaded bands)
==================================================================================
For each fold and EACH swept alpha, emit a single figure in the nsd_enkf.plotting paper
style (identical fonts / legend / band shading):
  * alpha_nsd/figures/nsd_alpha_<a>.png  — the 7 NSDs (2x4), mean + ±1/2σ bands + model + meas
  * alpha_obs/figures/asn_alpha_<a>.png  — Asn, same style
Plotted from the results_v1 sweep pkls (training-set trajectory + measurements). No re-run.

Usage:
    ./.venv/bin/python scripts/plot_v1_sweeps.py --run results_v1
"""

import argparse
import glob
import pickle
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import nsd_enkf.config as cfg
from nsd_enkf.data_loader import load_dataset

p = argparse.ArgumentParser(description="One paper-style figure per swept alpha (NSD + Asn)")
p.add_argument("--run", default="results_v1")
p.add_argument("--downsample", default=10, type=int)
p.add_argument("--dpi", default=200, type=int)
args = p.parse_args()

RUN = cfg.PROJECT_ROOT / args.run if not Path(args.run).is_absolute() else Path(args.run)
D = max(int(args.downsample), 1)
AX = cfg.AXIS_NAMES
T_meas = np.array(cfg.T_MEAS_FIXED)
n_nsd = 7
nsd_state_idx = list(range(cfg.STATE_NUM - n_nsd, cfg.STATE_NUM))
nsd_names = [cfg.STATE_NAMES[i] for i in nsd_state_idx]
ASN = cfg.STATE_NAMES.index("Asn")

# ── exact paper style (mirrors nsd_enkf.plotting) ─────────────────────────────
def _style_ax(ax):
    ax.grid(alpha=0.15)
    ax.tick_params(axis="both", labelsize=10)
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_fontweight("bold")
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

LEGEND = [
    Line2D([0], [0], color="red", lw=1.8, label="Mechanistic model"),
    Line2D([0], [0], color="steelblue", lw=2.0, label="EnKF mean"),
    Patch(facecolor="steelblue", alpha=0.30, label=r"EnKF $\pm 1\sigma$"),
    Patch(facecolor="steelblue", alpha=0.15, label=r"EnKF $\pm 2\sigma$"),
    Line2D([0], [0], color="darkorange", marker="o", lw=0, markersize=5, label="Measurements"),
]

_mc = {}
def meas(name, si):
    if name not in _mc:
        d = load_dataset(name)
        sm = d["set_meas"].astype(float); se = d["set_meas_errorbar"].astype(float)
        nsd = pd.DataFrame(d["NSD_meas"]).apply(pd.to_numeric, errors="coerce").to_numpy()
        nse = pd.DataFrame(d["NSD_meas_errorbar"]).apply(pd.to_numeric, errors="coerce").to_numpy()
        ac = sm.shape[1] - 1
        m = {ASN: (sm[:, ac], se[:, ac])}
        for j in range(n_nsd):
            m[nsd_state_idx[j]] = (nsd[:, j], nse[:, j])
        _mc[name] = m
    return _mc[name].get(si)


def band_panel(ax, T, mt, st, model, si, name):
    t = T[::D]; m = mt[::D, si]; s = st[::D, si]
    ax.fill_between(t, np.maximum(m - 2 * s, 0), m + 2 * s, color="steelblue", alpha=0.15)
    ax.fill_between(t, np.maximum(m - s, 0), m + s, color="steelblue", alpha=0.30)
    ax.plot(t, m, color="steelblue", lw=2.0)
    ax.plot(t, model[::D, si], color="red", lw=1.8)
    mv = meas(name, si)
    if mv is not None:
        v, e = mv
        ax.errorbar(T_meas, v, yerr=e, fmt="o", color="darkorange", markersize=4.0,
                    capsize=2, elinewidth=1, alpha=0.9, zorder=5)
    ax.set_ylabel(AX[si], fontsize=10, fontweight="bold")
    ax.set_xlabel("Time (hours)", fontsize=11, fontweight="bold")
    _style_ax(ax)


def _sorted_pkls(d):
    return sorted(glob.glob(str(d / "pkl" / "alpha_*.pkl")),
                  key=lambda x: float(re.search(r"alpha_([0-9.]+)\.pkl", x).group(1)))


fold_dirs = sorted(RUN.glob("fold_*"))
if not fold_dirs:
    raise SystemExit(f"No fold_* dirs in {RUN}")

for fdir in fold_dirs:
    fold = fdir.name.replace("fold_", "")
    # ---- NSD: one figure per alpha (7 NSDs, 2x4) ----
    for fp in _sorted_pkls(fdir / "alpha_nsd"):
        dd = pickle.load(open(fp, "rb"))
        a = dd["alpha"]; T = np.asarray(dd["T"]); mt = np.asarray(dd["mean_trajectory"])
        st = np.asarray(dd["std_trajectory"]); model = np.asarray(dd["model_trajectory"]); name = dd["train"]
        fig, axes = plt.subplots(2, 4, figsize=(5.0 * 4, 3.4 * 2))
        axes = axes.flatten()
        for k, (nm, si) in enumerate(zip(nsd_names, nsd_state_idx)):
            band_panel(axes[k], T, mt, st, model, si, name)
            axes[k].set_title(nm, fontsize=13, fontweight="bold")
        axes[7].set_visible(False)
        fig.legend(handles=LEGEND, loc="lower center", ncol=5, fontsize=11, frameon=False,
                   bbox_to_anchor=(0.5, -0.01))
        fig.suptitle(rf"NSDs — trained on {name},  $\alpha_{{nsd}}$ = {a:g}", fontsize=15, fontweight="bold")
        plt.tight_layout(rect=[0, 0.04, 1, 0.99])
        out = fdir / "alpha_nsd" / "figures" / f"nsd_alpha_{a:g}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=args.dpi, bbox_inches="tight"); plt.close(fig); print(f"saved: {out}")
    # ---- Asn: one figure per alpha ----
    for fp in _sorted_pkls(fdir / "alpha_obs"):
        dd = pickle.load(open(fp, "rb"))
        a = dd["alpha"]; T = np.asarray(dd["T"]); mt = np.asarray(dd["mean_trajectory"])
        st = np.asarray(dd["std_trajectory"]); model = np.asarray(dd["model_trajectory"]); name = dd["train"]
        fig, ax = plt.subplots(figsize=(7.5, 4.6))
        band_panel(ax, T, mt, st, model, ASN, name)
        ax.set_title(rf"Asn — trained on {name},  $\alpha_{{obs}}$ = {a:g}", fontsize=13, fontweight="bold")
        fig.legend(handles=LEGEND, loc="lower center", ncol=5, fontsize=10, frameon=False,
                   bbox_to_anchor=(0.5, -0.06))
        plt.tight_layout(rect=[0, 0.06, 1, 1])
        out = fdir / "alpha_obs" / "figures" / f"asn_alpha_{a:g}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=args.dpi, bbox_inches="tight"); plt.close(fig); print(f"saved: {out}")

print("Done.")
