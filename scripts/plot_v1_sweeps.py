"""
plot_v1_sweeps.py  —  per-fold sweep grids (mean + spread for every alpha)
==========================================================================
Reads the results_v1 sweep pkls and draws, for each fold, a grid showing the EnKF posterior
mean and +/-1/2 sigma uncertainty band at EVERY swept alpha, against the training-set
measurements. Two grids per fold:
  * alpha_nsd:  rows = 7 NSDs, cols = alpha  (scan a row to see the band widen with alpha)
  * alpha_obs:  Asn across alpha (single row)
Pure plotting from the saved pkls (no EnKF re-run).

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

p = argparse.ArgumentParser(description="Per-fold alpha-sweep grids (mean + spread for every alpha)")
p.add_argument("--run", default="results_v1")
p.add_argument("--traj-down", default=20, type=int)
args = p.parse_args()

RUN = cfg.PROJECT_ROOT / args.run if not Path(args.run).is_absolute() else Path(args.run)
DOWN = max(int(args.traj_down), 1)
T_meas = np.array(cfg.T_MEAS_FIXED)
n_nsd = 7
nsd_state_idx = list(range(cfg.STATE_NUM - n_nsd, cfg.STATE_NUM))
nsd_names = [cfg.STATE_NAMES[i] for i in nsd_state_idx]
ASN = cfg.STATE_NAMES.index("Asn")

_mcache = {}
def meas_for(name, si):
    if name not in _mcache:
        d = load_dataset(name)
        sm = d["set_meas"].astype(float); se = d["set_meas_errorbar"].astype(float)
        nsd = pd.DataFrame(d["NSD_meas"]).apply(pd.to_numeric, errors="coerce").to_numpy()
        nse = pd.DataFrame(d["NSD_meas_errorbar"]).apply(pd.to_numeric, errors="coerce").to_numpy()
        asn_col = sm.shape[1] - 1
        m = {ASN: (sm[:, asn_col], se[:, asn_col])}
        for j in range(n_nsd):
            m[nsd_state_idx[j]] = (nsd[:, j], nse[:, j])
        _mcache[name] = m
    return _mcache[name].get(si)

LEGEND = [Line2D([0], [0], color="steelblue", lw=2, label="EnKF mean"),
          Patch(facecolor="steelblue", alpha=0.30, label=r"$\pm1\sigma$"),
          Patch(facecolor="steelblue", alpha=0.15, label=r"$\pm2\sigma$"),
          Line2D([0], [0], color="red", lw=1.2, ls="--", label="open-loop model"),
          Line2D([0], [0], color="darkorange", marker="o", lw=0, ms=5, label="measurements")]


def load_sweep(fold_dir, kind):
    files = sorted(glob.glob(str(fold_dir / f"alpha_{kind}" / "pkl" / "alpha_*.pkl")),
                   key=lambda x: float(re.search(r"alpha_([0-9.]+)\.pkl", x).group(1)))
    return [pickle.load(open(f, "rb")) for f in files]


def panel(ax, dd, si, tds):
    mt = np.asarray(dd["mean_trajectory"]); st = np.asarray(dd["std_trajectory"])
    m = mt[::DOWN, si]; s = st[::DOWN, si]
    ax.fill_between(tds, np.maximum(m - 2 * s, 0), m + 2 * s, color="steelblue", alpha=0.15)
    ax.fill_between(tds, np.maximum(m - s, 0), m + s, color="steelblue", alpha=0.30)
    ax.plot(tds, m, color="steelblue", lw=1.8)
    ax.plot(tds, np.asarray(dd["model_trajectory"])[::DOWN, si], color="red", lw=1.1, ls="--")
    mv = meas_for(dd["train"], si)
    if mv is not None:
        v, e = mv
        ax.errorbar(T_meas, v, yerr=e, fmt="o", color="darkorange", ms=3.5, capsize=2, elinewidth=0.9, zorder=5)
    ax.grid(alpha=0.15)


fold_dirs = sorted(RUN.glob("fold_*"))
if not fold_dirs:
    raise SystemExit(f"No fold_* dirs in {RUN}")

for fdir in fold_dirs:
    fold = fdir.name.replace("fold_", "")
    # ---- alpha_nsd grid: 7 NSD rows x alpha cols ----
    sweep = load_sweep(fdir, "nsd")
    if sweep:
        alphas = [d["alpha"] for d in sweep]; tds = np.asarray(sweep[0]["T"])[::DOWN]
        nr, nc = n_nsd, len(alphas)
        fig, axes = plt.subplots(nr, nc, figsize=(3.2 * nc, 2.1 * nr), sharey="row", squeeze=False)
        for r, (nm, si) in enumerate(zip(nsd_names, nsd_state_idx)):
            for c, dd in enumerate(sweep):
                ax = axes[r][c]; panel(ax, dd, si, tds)
                cov = dd["metrics"][nm]["cov"]
                if r == 0:
                    ax.set_title(f"α={alphas[c]:g}", fontsize=11, fontweight="bold")
                ax.text(0.02, 0.92, f"cov {cov:.0f}%", transform=ax.transAxes, fontsize=7, va="top")
                if c == 0:
                    ax.set_ylabel(nm, fontsize=10, fontweight="bold")
                if r == nr - 1:
                    ax.set_xlabel("Time (h)", fontsize=8)
        fig.legend(handles=LEGEND, loc="lower center", ncol=5, fontsize=11, frameon=False, bbox_to_anchor=(0.5, -0.01))
        fig.suptitle(f"α_nsd sweep — fold trained on {fold} (mean + spread, all α)", fontsize=15, fontweight="bold")
        plt.tight_layout(rect=[0, 0.02, 1, 0.99])
        out = fdir / "alpha_nsd" / "figures" / "nsd_bands_grid.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=110, bbox_inches="tight"); plt.close(fig); print(f"saved: {out}")
    # ---- alpha_obs grid: Asn across alpha ----
    sweep = load_sweep(fdir, "obs")
    if sweep:
        alphas = [d["alpha"] for d in sweep]; tds = np.asarray(sweep[0]["T"])[::DOWN]
        fig, axes = plt.subplots(1, len(alphas), figsize=(3.4 * len(alphas), 3.2), sharey=True, squeeze=False)
        for c, dd in enumerate(sweep):
            ax = axes[0][c]; panel(ax, dd, ASN, tds)
            ax.set_title(f"α_obs={alphas[c]:g}\nAsn cov {dd['metrics']['Asn']['cov']:.0f}%", fontsize=10, fontweight="bold")
            ax.set_xlabel("Time (h)", fontsize=8)
            if c == 0:
                ax.set_ylabel("Asn (mM)", fontsize=10)
        fig.legend(handles=LEGEND, loc="lower center", ncol=5, fontsize=10, frameon=False, bbox_to_anchor=(0.5, -0.06))
        fig.suptitle(f"α_obs sweep — fold trained on {fold} (Asn mean + spread, all α)", fontsize=14, fontweight="bold")
        plt.tight_layout(rect=[0, 0.04, 1, 0.97])
        out = fdir / "alpha_obs" / "figures" / "asn_bands_grid.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=120, bbox_inches="tight"); plt.close(fig); print(f"saved: {out}")

print("Done.")
