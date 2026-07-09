"""
plot_cross_validation.py  —  plot held-out trajectories from a 04_cross_validate run
====================================================================================
Reads results/<run>/<mode>/fold_*.pkl and draws, for every (training fold x held-out
dataset), an all-17-state grid: EnKF posterior mean, +/-1 sigma and +/-2 sigma ensemble
bands, the open-loop model, and the measurements. Pure plotting from the saved pkls — no
EnKF re-run. This is the cross-validation analogue of the per-dataset Option-B figures:
each panel shows how a filter tuned on one batch predicts a batch it never saw.

Usage (macOS venv):
    ./.venv/bin/python scripts/plot_cross_validation.py --run cross_validation --mode cv
    ./.venv/bin/python scripts/plot_cross_validation.py --run cross_validation --mode all
"""

import argparse
import pickle
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

p = argparse.ArgumentParser(description="Plot held-out trajectories from a cross-validation run")
p.add_argument("--run", default="cross_validation")
p.add_argument("--mode", default="cv", choices=["cv", "all"])
p.add_argument("--traj-down", default=20, type=int)
args = p.parse_args()

RES = cfg.PROJECT_ROOT / "results" / args.run / args.mode
FIG = RES / "figures"
FIG.mkdir(parents=True, exist_ok=True)
DOWN = max(int(args.traj_down), 1)
T_meas = np.array(cfg.T_MEAS_FIXED)
n_nsd = 7
nsd_state_idx = list(range(cfg.STATE_NUM - n_nsd, cfg.STATE_NUM))
ASN = cfg.STATE_NAMES.index("Asn")

# Measurements per held-out dataset (measured extracellular + Asn + 7 NSDs), cached.
_mcache = {}
def meas_by_state(name):
    if name in _mcache:
        return _mcache[name]
    d = load_dataset(name)
    sm = d["set_meas"].astype(float); se = d["set_meas_errorbar"].astype(float)
    nsd = pd.DataFrame(d["NSD_meas"]).apply(pd.to_numeric, errors="coerce").to_numpy()
    nse = pd.DataFrame(d["NSD_meas_errorbar"]).apply(pd.to_numeric, errors="coerce").to_numpy()
    asn_col = sm.shape[1] - 1
    mbs = {i: (sm[:, i], se[:, i]) for i in range(cfg.MEAS_NUM)}
    mbs[ASN] = (sm[:, asn_col], se[:, asn_col])
    for j in range(n_nsd):
        mbs[nsd_state_idx[j]] = (nsd[:, j], nse[:, j])
    _mcache[name] = mbs
    return mbs

LEGEND = [
    Line2D([0], [0], color="red", lw=1.8, label="Open-loop model"),
    Line2D([0], [0], color="steelblue", lw=2.0, label="EnKF mean (held-out)"),
    Patch(facecolor="steelblue", alpha=0.30, label=r"$\pm1\sigma$"),
    Patch(facecolor="steelblue", alpha=0.15, label=r"$\pm2\sigma$"),
    Line2D([0], [0], color="darkorange", marker="o", lw=0, markersize=6, label="Measurements"),
]

fold_pkls = sorted(RES.glob("fold_*.pkl"))
if not fold_pkls:
    raise SystemExit(f"No fold_*.pkl in {RES} — run scripts/04_cross_validate.py --retune {args.mode} first.")

print(f"Plotting {args.mode} cross-validation from {len(fold_pkls)} folds in {RES}")
for fp in fold_pkls:
    P = pickle.load(open(fp, "rb"))
    T = np.asarray(P["T"]); tds = T[::DOWN]
    train = "+".join(P["train"])
    for val, vd in P["val"].items():
        mt = np.asarray(vd["mean_trajectory"]); st = np.asarray(vd["std_trajectory"])
        model = np.asarray(vd["model_trajectory"])
        mbs = meas_by_state(val)
        fig, axes = plt.subplots(5, 4, figsize=(20, 15)); axes = axes.flatten()
        for si in range(cfg.STATE_NUM):
            ax = axes[si]; m = mt[::DOWN, si]; s = st[::DOWN, si]
            ax.fill_between(tds, np.maximum(m - 2 * s, 0), m + 2 * s, color="steelblue", alpha=0.15)
            ax.fill_between(tds, np.maximum(m - s, 0), m + s, color="steelblue", alpha=0.30)
            ax.plot(tds, m, color="steelblue", lw=2.0)
            ax.plot(tds, model[::DOWN, si], color="red", lw=1.6)
            if si in mbs:
                v, e = mbs[si]
                ax.errorbar(T_meas, v, yerr=e, fmt="o", color="darkorange", markersize=4,
                            capsize=2, elinewidth=1, alpha=0.9, zorder=5)
            tag = "" if si in mbs else "  (no meas)"
            ax.set_title(f"{cfg.STATE_NAMES[si]}{tag}", fontsize=11, fontweight="bold")
            ax.set_xlabel("Time (h)", fontsize=9); ax.grid(alpha=0.15)
        for k in range(cfg.STATE_NUM, len(axes)):
            axes[k].set_visible(False)
        met = vd["metrics"]
        nsd_nrmse = np.nanmean([met[n]["nrmse"] for n in met if n != "Asn"])
        fig.legend(handles=LEGEND, loc="lower center", ncol=5, fontsize=12, frameon=False,
                   bbox_to_anchor=(0.5, -0.005))
        fig.suptitle(f"Cross-validation [{args.mode}] — trained on {train}, held-out {val}   "
                     f"(α_nsd={P['alpha_nsd']:g}, α_obs={P['alpha_obs']:g}; "
                     f"held-out NSD NRMSE={nsd_nrmse:.2f}, Asn NRMSE={met['Asn']['nrmse']:.2f})",
                     fontsize=15, fontweight="bold", y=1.005)
        plt.tight_layout(rect=[0, 0.03, 1, 1])
        out = FIG / f"cv_train{train}_heldout{val}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
        print(f"  saved: {out}")

print("Done.")
