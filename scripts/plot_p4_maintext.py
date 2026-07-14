"""
plot_p4_maintext.py  —  P4-only main-text figures with multi-seed uncertainty bands
====================================================================================
Reproduces the three main-text figure styles (Fig. 1 measured metabolites, the Asn
soft-sensing panel, and the NSD figure) but for the **P4 fold only** and with the
seed-averaged 2-sigma uncertainty band shaded in — which the originals lacked.

Only the states already shown in Fig. 1 are plotted:
  measured : Xv (viable cell density), Gal, Urd, Glc, Gln
  Asn      : asparagine (observable-unmeasured)
  NSDs     : UDP-Gal, UDP-Glc, UDP-GlcNAc (the reliably-measured reported NSDs)

Source (default): results_multirun_nsd/fold_P4/agg/alpha_0.02.pkl — the P4 experiment
under the finalised filter (alpha_obs=0.002, alpha_nsd=0.02), averaged over the 10
clean seeds, divergent replicates rejected. That pkl carries all 17 states plus the
2-sigma band, so all three figures come from one consistent P4 run. Pass --agg to
point at a different agg pkl (e.g. a held-out validation run).

Style matches the pasted references: solid = mechanistic model, dashed = EnKF mean,
filled diamonds w/ error bars = measurements, all in the P4 colour (purple); the new
shaded band is the seed-averaged EnKF 2-sigma envelope.

Usage:
    ./.venv/bin/python scripts/plot_p4_maintext.py
    ./.venv/bin/python scripts/plot_p4_maintext.py --agg results_multirun_validation/fold_P3/agg/heldout_P4.pkl
"""

import argparse
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import nsd_enkf.config as cfg
from nsd_enkf.data_loader import load_dataset

# ── CLI ───────────────────────────────────────────────────────────────────────
p = argparse.ArgumentParser(description="P4-only main-text figures with uncertainty bands")
p.add_argument("--agg", default="results_multirun_nsd/fold_P4/agg/alpha_0.02.pkl",
               help="seed-averaged agg pkl for the P4 run (all 17 states + 2sigma band)")
p.add_argument("--dataset", default="P4")
p.add_argument("--outdir", default="/Users/luxi.yu/Research/Soft_Sensing_Paper/Figs",
               help="where the P4 main-text figures are written")
p.add_argument("--downsample", default=10, type=int)
p.add_argument("--dpi", default=300, type=int)
args = p.parse_args()

DS = args.dataset
DOWN = max(int(args.downsample), 1)
OUTDIR = Path(args.outdir)
OUTDIR.mkdir(parents=True, exist_ok=True)

PURPLE = "#9C179E"           # P4 colour, matched to the reference figures
MODEL_LW, ENKF_LW = 2.0, 2.0
MARKER = "D"                 # P4 marker (diamond), as in the references

# ── Load the seed-averaged P4 run ───────────────────────────────────────────────
AGG = Path(args.agg) if Path(args.agg).is_absolute() else cfg.PROJECT_ROOT / args.agg
with open(AGG, "rb") as f:
    agg = pickle.load(f)

T = np.asarray(agg["T"])
state_names = list(agg["state_names"])
model = np.asarray(agg["model_trajectory"])            # (28801, 17) open-loop model
enkf = np.asarray(agg["avg_mean_trajectory"])          # (28801, 17) seed-averaged EnKF mean
band_lo = np.asarray(agg["band_2sigma_lo"])            # (28801, 17)
band_hi = np.asarray(agg["band_2sigma_hi"])            # (28801, 17)
n_seed = len(agg.get("seeds", []))
print(f"Loaded {AGG.name}: {n_seed} clean seeds, alpha_obs={agg.get('alpha_obs')}, "
      f"alpha_nsd={agg.get('alpha_nsd')}")

# measurements (experimental) for the overlay
data = load_dataset(DS)
set_meas = np.asarray(data["set_meas"], dtype=float)          # (17, 9): Xv..Lac, Asn
set_meas_err = np.asarray(data["set_meas_errorbar"], dtype=float)
NSD_meas = np.asarray(data["NSD_meas"], dtype=float)          # (17, 7): local NSD idx
NSD_meas_err = np.asarray(data["NSD_meas_errorbar"], dtype=float)
T_meas = np.asarray(cfg.T_MEAS_FIXED, dtype=float)

sl = slice(None, None, DOWN)
Td = T[sl]


def draw_panel(ax, state_idx, ylabel, meas_vals=None, meas_err=None, letter=None):
    """One state panel: model (solid) + EnKF mean (dashed) + 2sigma band + measurements."""
    lo = np.maximum(band_lo[sl, state_idx], 0.0)           # concentrations are non-negative
    hi = band_hi[sl, state_idx]
    ax.fill_between(Td, lo, hi, color=PURPLE, alpha=0.18, lw=0, zorder=1)
    ax.plot(Td, model[sl, state_idx], color=PURPLE, lw=MODEL_LW, ls="-", zorder=3)
    ax.plot(Td, enkf[sl, state_idx], color=PURPLE, lw=ENKF_LW, ls="--", zorder=4)
    if meas_vals is not None:
        ax.errorbar(T_meas, meas_vals, yerr=meas_err, fmt=MARKER, color=PURPLE,
                    markersize=6, markeredgecolor="black", markeredgewidth=0.7,
                    capsize=2.5, elinewidth=1.1, ecolor="black", ls="none", zorder=5)
    ax.set_ylabel(ylabel, fontsize=11, fontweight="bold")
    ax.set_xlabel("Time (hours)", fontsize=11, fontweight="bold")
    ax.set_xlim(-8, 296)
    ax.grid(alpha=0.13)
    ax.tick_params(axis="both", labelsize=9)
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_fontweight("bold")
    for spine in ax.spines.values():
        spine.set_linewidth(1.6)
    if letter is not None:
        ax.text(0.03, 0.93, letter, transform=ax.transAxes, fontsize=13,
                fontweight="bold", va="top", ha="left")
    ax.text(0.03, 0.83, DS, transform=ax.transAxes, fontsize=11,
            fontweight="bold", va="top", ha="left")


def bottom_legend(fig, measurement_label, ncol=4):
    handles = [
        Line2D([0], [0], color=PURPLE, lw=MODEL_LW, ls="-", label=f"{DS} – Model"),
        Line2D([0], [0], color=PURPLE, lw=ENKF_LW, ls="--", label=f"{DS} – EnKF"),
        Patch(facecolor=PURPLE, alpha=0.18, label=r"EnKF $\pm2\sigma$ (10 seeds)"),
        Line2D([0], [0], color=PURPLE, marker=MARKER, lw=0, markersize=7,
               markeredgecolor="black", markeredgewidth=0.7, label=f"{DS} – {measurement_label}"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=ncol, fontsize=11,
               frameon=False, bbox_to_anchor=(0.5, -0.02))


LETTERS = "abcdefghijklmnop"

# ── Figure 1: measured metabolites (Xv, Gal, Urd, Glc, Gln) ─────────────────────
MEAS_PANELS = [
    ("Xv",  "Viable Cell Density (cell $L^{-1}$)"),
    ("Gal", "Galactose Concentration (mM)"),
    ("Urd", "Urdine Concentration (mM)"),
    ("Glc", "Glucose Concentration (mM)"),
    ("Gln", "Glutamine Concentration (mM)"),
]
fig, axes = plt.subplots(3, 2, figsize=(11, 12))
axes = axes.flatten()
for i, (name, ylab) in enumerate(MEAS_PANELS):
    si = state_names.index(name)
    draw_panel(axes[i], si, ylab, meas_vals=set_meas[:, si], meas_err=set_meas_err[:, si],
               letter=f"({LETTERS[i]})")
axes[-1].set_visible(False)                                  # 6th cell unused (5 states)
bottom_legend(fig, "Measurement Update", ncol=4)
plt.tight_layout(rect=[0, 0.045, 1, 1])
out = OUTDIR / "measured_metabolites_P4_bands.png"
fig.savefig(out, dpi=args.dpi, bbox_inches="tight"); plt.close(fig)
print(f"Saved: {out}")

# ── Figure 2: Asparagine (observable-unmeasured) ────────────────────────────────
fig, ax = plt.subplots(1, 1, figsize=(7.5, 5.2))
si = state_names.index("Asn")
draw_panel(ax, si, "Asparagine Concentration (mM)",
           meas_vals=set_meas[:, si], meas_err=set_meas_err[:, si], letter=None)
bottom_legend(fig, "Measurement", ncol=4)
plt.tight_layout(rect=[0, 0.07, 1, 1])
out = OUTDIR / "asn_P4_bands.png"
fig.savefig(out, dpi=args.dpi, bbox_inches="tight"); plt.close(fig)
print(f"Saved: {out}")

# ── Figure 3: reported NSDs (UDP-Gal, UDP-Glc, UDP-GlcNAc) ───────────────────────
NSD_PANELS = [
    ("UDPGal",    0, "UDP-Gal Concentration (mM)"),
    ("UDPGlc",    2, "UDP-Glc Concentration (mM)"),
    ("UDPGlcNAc", 3, "UDP-GlcNAc Concentration (mM)"),
]
fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.6))
for i, (name, jloc, ylab) in enumerate(NSD_PANELS):
    si = state_names.index(name)
    draw_panel(axes[i], si, ylab, meas_vals=NSD_meas[:, jloc], meas_err=NSD_meas_err[:, jloc],
               letter=f"({LETTERS[i]})")
bottom_legend(fig, "Validation Measurement", ncol=4)
plt.tight_layout(rect=[0, 0.07, 1, 1])
out = OUTDIR / "main_nsd_P4_bands.png"
fig.savefig(out, dpi=args.dpi, bbox_inches="tight"); plt.close(fig)
print(f"Saved: {out}")

print("Done.")
