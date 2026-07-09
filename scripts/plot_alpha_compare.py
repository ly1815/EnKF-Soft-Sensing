"""
plot_alpha_compare.py  —  overlay NSD uncertainty bands across all swept alpha values
=====================================================================================
Reads the per-alpha pkls from an 03_tune_alpha_nsd.py sweep and draws ONE figure: a grid of
the 7 NSDs, each panel overlaying the +/-2 sigma band (upper & lower envelopes) for every
alpha, colored low->high, with the open-loop model and measurements. Lets you compare at a
glance how the band widens with alpha and where it covers the data.

Usage (macOS venv):
    ./.venv/bin/python scripts/plot_alpha_compare.py --pkl-dir results/legacy/alpha_nsd/pkl
"""

import argparse
import glob
import pickle
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import nsd_enkf.config as cfg

p = argparse.ArgumentParser(description="Overlay NSD +/-2 sigma bands across swept alpha values")
p.add_argument("--pkl-dir", default="results/legacy/alpha_nsd/pkl")
p.add_argument("--out", default=None)
p.add_argument("--traj-down", default=20, type=int)
args = p.parse_args()

pkl_dir = (cfg.PROJECT_ROOT / args.pkl_dir) if not Path(args.pkl_dir).is_absolute() else Path(args.pkl_dir)
files = sorted(glob.glob(str(pkl_dir / "nsd_alpha_*.pkl")),
               key=lambda x: float(re.search(r"alpha_([0-9.]+)\.pkl", x).group(1)))
if not files:
    raise SystemExit(f"No nsd_alpha_*.pkl in {pkl_dir}")
data = [pickle.load(open(f, "rb")) for f in files]
alphas = [d["alpha_nsd"] for d in data]
nsd_names = data[0]["nsd_names"]
T = np.asarray(data[0]["T"]); DOWN = max(int(args.traj_down), 1); tds = T[::DOWN]
T_meas = np.asarray(data[0]["T_meas"])
colors = plt.cm.viridis(np.linspace(0, 1, len(alphas)))

print(f"Overlaying alphas {alphas} from {pkl_dir}")
fig, axes = plt.subplots(2, 4, figsize=(22, 9)); axes = axes.flatten()
for j, nm in enumerate(nsd_names):
    ax = axes[j]
    b0 = data[0][nm]
    for i in range(len(alphas)):
        b = data[i][nm]
        ax.plot(tds, b["band_2sigma_hi"][::DOWN], color=colors[i], lw=1.1)
        ax.plot(tds, b["band_2sigma_lo"][::DOWN], color=colors[i], lw=1.1)
    ax.plot(tds, b0["model_trajectory"][::DOWN], color="red", lw=1.3, ls="--")
    ax.errorbar(T_meas, b0["meas"], yerr=b0["err"], fmt="o", color="darkorange",
                ms=4, capsize=2, elinewidth=1, zorder=6)
    # coverage per alpha in the title, so you see the calibration trend
    covs = " ".join(f"{data[i]['metrics'][nm]['cov']:.0f}" for i in range(len(alphas)))
    ax.set_title(f"{nm}\ncov% per α: {covs}", fontsize=9, fontweight="bold")
    ax.set_xlabel("Time (h)"); ax.set_ylabel("mM"); ax.grid(alpha=0.15)
for k in range(len(nsd_names), len(axes)):
    axes[k].set_visible(False)

handles = [Line2D([0], [0], color=colors[i], lw=4, label=f"α={alphas[i]:g}") for i in range(len(alphas))]
handles += [Line2D([0], [0], color="red", lw=1.3, ls="--", label="open-loop model"),
            Line2D([0], [0], color="darkorange", marker="o", lw=0, ms=6, label="measurements")]
fig.legend(handles=handles, loc="lower center", ncol=len(alphas) + 2, fontsize=11, frameon=False,
           bbox_to_anchor=(0.5, -0.02))
fig.suptitle(r"NSD $\pm2\sigma$ bands vs $\alpha_{nsd}$ (P4) — envelopes widen with $\alpha$",
             fontsize=15, fontweight="bold")
plt.tight_layout(rect=[0, 0.04, 1, 0.98])
out = Path(args.out) if args.out else (pkl_dir.parent / "figures" / "nsd_bands_alpha_COMPARE.png")
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
print(f"saved: {out}")
