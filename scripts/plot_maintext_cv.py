"""
plot_maintext_cv.py  —  main-text figures, full P1-P4 grid, tuning + validation
=================================================================================
Reproduces the three main-text figure layouts EXACTLY as the references
(measured metabolites 5x4, Asn 2x2, reported NSDs 3x4) across all four batches,
with the seed-averaged EnKF 2-sigma uncertainty band shaded in.

The cross-validation story shown is **train on P4, validate on P1-P3**:
  P4  (Tuning set)    -> filter tuned on P4, applied to P4 (in-sample)
                         results_multirun_nsd/fold_P4/agg/alpha_0.02.pkl
  P1,P2,P3 (Validation) -> the P4-tuned filter applied to each held-out batch
                         results_multirun_validation/fold_P4/agg/heldout_<k>.pkl

All runs use the finalised filter (alpha_obs=0.002, alpha_nsd=0.02), seed-averaged
over 10 clean seeds with divergent replicates rejected. Per-dataset colour/marker
match the references (P1 orange/o, P2 green/s, P3 blue/^, P4 purple/D). The bottom
legend states, per column, whether the batch is the Tuning or the Validation set.

Only the states already shown in the reference figures are plotted:
  measured : Xv, Gal, Urd, Glc, Gln          (rows) x P1..P4 (cols)
  Asn      : asparagine                        2x2 over P1..P4
  NSDs     : UDP-Gal, UDP-Glc, UDP-GlcNAc     (rows) x P1..P4 (cols)

Usage:
    ./.venv/bin/python scripts/plot_maintext_cv.py
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
p = argparse.ArgumentParser(description="Main-text CV figures (P1-P4) with uncertainty bands")
p.add_argument("--nsd-run", default="results_multirun_nsd/fold_P4/agg/alpha_0.02.pkl",
               help="tuning (in-sample P4) agg pkl")
p.add_argument("--val-dir", default="results_multirun_validation/fold_P4/agg",
               help="dir holding heldout_<k>.pkl for the validation columns")
p.add_argument("--outdir", default="/Users/luxi.yu/Research/Soft_Sensing_Paper/Figs")
p.add_argument("--downsample", default=10, type=int)
p.add_argument("--dpi", default=300, type=int)
args = p.parse_args()

DOWN = max(int(args.downsample), 1)
OUTDIR = Path(args.outdir); OUTDIR.mkdir(parents=True, exist_ok=True)

COLS = ["P1", "P2", "P3", "P4"]
COLZ = {"P1": "tab:orange", "P2": "tab:green", "P3": "tab:blue", "P4": "#9C179E"}
MKR = {"P1": "o", "P2": "o", "P3": "o", "P4": "o"}
ROLE = {"P1": "Validation", "P2": "Validation", "P3": "Validation", "P4": "Tuning"}
MODEL_LW, ENKF_LW = 2.0, 2.0


def _resolve(pth):
    return Path(pth) if Path(pth).is_absolute() else cfg.PROJECT_ROOT / pth


# ── Load the seed-averaged run for each column ──────────────────────────────────
SRC = {"P4": _resolve(args.nsd_run)}
for k in ("P1", "P2", "P3"):
    SRC[k] = _resolve(args.val_dir) / f"heldout_{k}.pkl"

DATA, MEAS = {}, {}
for ds in COLS:
    with open(SRC[ds], "rb") as f:
        a = pickle.load(f)
    DATA[ds] = {
        "T": np.asarray(a["T"]),
        "model": np.asarray(a["model_trajectory"]),
        "enkf": np.asarray(a["avg_mean_trajectory"]),
        "std": np.asarray(a["avg_std_trajectory"]),
        "lo": np.asarray(a["band_2sigma_lo"]),
        "hi": np.asarray(a["band_2sigma_hi"]),
        "state_names": list(a["state_names"]),
        "n_seed": len(a.get("seeds", [])),
    }
    d = load_dataset(ds)
    MEAS[ds] = {
        "set_meas": np.asarray(d["set_meas"], dtype=float),          # (17, 9)
        "set_meas_err": np.asarray(d["set_meas_errorbar"], dtype=float),
        "NSD_meas": np.asarray(d["NSD_meas"], dtype=float),          # (17, 7)
        "NSD_meas_err": np.asarray(d["NSD_meas_errorbar"], dtype=float),
    }
    print(f"{ds} ({ROLE[ds]}): {SRC[ds].name}  [{DATA[ds]['n_seed']} clean seeds]")

T_meas = np.asarray(cfg.T_MEAS_FIXED, dtype=float)
STATE_NAMES = DATA["P4"]["state_names"]
LETTERS = "abcdefghijklmnopqrstuvwxyz"


def draw_panel(ax, ds, state_idx, ylabel=None, meas_vals=None, meas_err=None,
               letter=None, show_role=False, show_ylabel=True, tag_bottom=False):
    color, marker = COLZ[ds], MKR[ds]
    d, sl = DATA[ds], slice(None, None, DOWN)
    Td = d["T"][sl]
    lo = np.maximum(d["lo"][sl, state_idx], 0.0)
    hi = d["hi"][sl, state_idx]
    mean1 = d["enkf"][sl, state_idx]
    sig1 = d["std"][sl, state_idx]
    lo1 = np.maximum(mean1 - sig1, 0.0)
    hi1 = mean1 + sig1
    ax.fill_between(Td, lo, hi, color=color, alpha=0.16, lw=0, zorder=1)     # +/-2 sigma
    ax.fill_between(Td, lo1, hi1, color=color, alpha=0.22, lw=0, zorder=2)   # +/-1 sigma (darker)
    ax.plot(Td, d["model"][sl, state_idx], color=color, lw=MODEL_LW, ls="-", zorder=3)
    ax.plot(Td, d["enkf"][sl, state_idx], color=color, lw=ENKF_LW, ls="--", zorder=4)
    if meas_vals is not None:
        ax.errorbar(T_meas, meas_vals, yerr=meas_err, fmt=marker, color=color,
                    markersize=5.5, markeredgecolor="black", markeredgewidth=0.6,
                    capsize=2.2, elinewidth=1.0, ecolor="black", ls="none", zorder=5)
    if show_ylabel and ylabel:
        ax.set_ylabel(ylabel, fontsize=10.5, fontweight="bold")
    ax.set_xlim(-8, 296)
    ax.grid(alpha=0.13)
    ax.tick_params(axis="both", labelsize=9)
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_fontweight("bold")
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    if letter is not None:
        ax.text(0.035, 0.94, f"({letter})", transform=ax.transAxes, fontsize=12.5,
                fontweight="bold", va="top", ha="left")
    tag = f"{ds} ({ROLE[ds]})" if show_role else ds
    ty, tva = (0.06, "bottom") if tag_bottom else (0.83, "top")
    ax.text(0.035, ty, tag, transform=ax.transAxes, fontsize=10.5,
            fontweight="bold", va=tva, ha="left")


def cv_legend(fig, measurement_label=None, y=-0.015):
    """Single shared legend, drawn in black. Each panel already carries its own
    batch/role label (P1..P4, Tuning/Validation) and its own colour, so nothing is
    repeated per dataset here. Four entries only: the open-loop model prediction, the
    EnKF mean trajectory, the +/-2 sigma uncertainty band, and the experimental
    measurements. All panels use the same circle marker (the batch is already
    identified by panel colour and label), so one circle handle suffices."""
    handles = [
        Line2D([0], [0], color="black", lw=MODEL_LW, ls="-"),
        Line2D([0], [0], color="black", lw=ENKF_LW, ls="--"),
        Patch(facecolor="black", edgecolor="none", alpha=0.38),
        Patch(facecolor="black", edgecolor="none", alpha=0.16),
        Line2D([0], [0], color="black", marker="o", lw=0, markersize=7,
               markerfacecolor="black", markeredgecolor="black", markeredgewidth=0.6),
    ]
    labels = [
        "Model prediction",
        "EnKF mean trajectory",
        r"$\pm1\sigma$ uncertainty band",
        r"$\pm2\sigma$ uncertainty band",
        "Experimental Measurement",
    ]
    fig.legend(handles=handles, labels=labels, loc="lower center", ncol=5,
               fontsize=13, frameon=False, bbox_to_anchor=(0.5, y),
               columnspacing=2.2, handletextpad=0.6, handlelength=2.4)


# ── Figure 1: measured metabolites — 5 rows x 4 cols ────────────────────────────
MEAS_ROWS = [
    ("Xv",  "Viable Cell Density (cell $L^{-1}$)"),
    ("Gal", "Galactose Concentration (mM)"),
    ("Urd", "Uridine Concentration (mM)"),
    ("Glc", "Glucose Concentration (mM)"),
    ("Gln", "Glutamine Concentration (mM)"),
]
nr, nc = len(MEAS_ROWS), len(COLS)
fig, axes = plt.subplots(nr, nc, figsize=(4.7 * nc, 3.05 * nr), sharex=False)
li = 0
for r, (name, ylab) in enumerate(MEAS_ROWS):
    si = STATE_NAMES.index(name)
    for c, ds in enumerate(COLS):
        ax = axes[r, c]
        draw_panel(ax, ds, si, ylabel=ylab, meas_vals=MEAS[ds]["set_meas"][:, si],
                   meas_err=MEAS[ds]["set_meas_err"][:, si], letter=LETTERS[li],
                   show_role=(r == 0), show_ylabel=True)
        ax.set_xlabel("Time (hours)", fontsize=11, fontweight="bold")
        li += 1
cv_legend(fig, "Measurement Update", y=0.045)
plt.tight_layout(rect=[0, 0.085, 1, 1])
out = OUTDIR / "measured_metabolites_CV_bands.png"
fig.savefig(out, dpi=args.dpi, bbox_inches="tight"); plt.close(fig)
print(f"Saved: {out}")

# ── Figure 2: Asparagine — 1 x 4 over P1..P4 (single row, same format) ───────────
nc = len(COLS)
fig, axes = plt.subplots(1, nc, figsize=(4.7 * nc, 3.9), sharex=False)
si = STATE_NAMES.index("Asn")
for i, ds in enumerate(COLS):
    ax = axes[i]
    draw_panel(ax, ds, si, ylabel="Asparagine Concentration (mM)",
               meas_vals=MEAS[ds]["set_meas"][:, si], meas_err=MEAS[ds]["set_meas_err"][:, si],
               letter=LETTERS[i], show_role=True, show_ylabel=True, tag_bottom=True)
    ax.set_xlabel("Time (hours)", fontsize=11, fontweight="bold")
cv_legend(fig, "Measurement", y=-0.03)
plt.tight_layout(rect=[0, 0.11, 1, 1])
out = OUTDIR / "asn_CV_bands.png"
fig.savefig(out, dpi=args.dpi, bbox_inches="tight"); plt.close(fig)
print(f"Saved: {out}")

# ── Figure 3: reported NSDs — 3 rows x 4 cols ───────────────────────────────────
NSD_ROWS = [
    ("UDPGal",    0, "UDP-Gal Concentration (mM)"),
    ("UDPGlc",    2, "UDP-Glc Concentration (mM)"),
    ("UDPGlcNAc", 3, "UDP-GlcNAc Concentration (mM)"),
]
nr, nc = len(NSD_ROWS), len(COLS)
fig, axes = plt.subplots(nr, nc, figsize=(4.7 * nc, 3.4 * nr), sharex=False)
li = 0
for r, (name, jloc, ylab) in enumerate(NSD_ROWS):
    si = STATE_NAMES.index(name)
    for c, ds in enumerate(COLS):
        ax = axes[r, c]
        draw_panel(ax, ds, si, ylabel=ylab, meas_vals=MEAS[ds]["NSD_meas"][:, jloc],
                   meas_err=MEAS[ds]["NSD_meas_err"][:, jloc], letter=LETTERS[li],
                   show_role=(r == 0), show_ylabel=True)
        ax.set_xlabel("Time (hours)", fontsize=11, fontweight="bold")
        li += 1
cv_legend(fig, "Validation Measurement", y=0.05)
plt.tight_layout(rect=[0, 0.11, 1, 1])
out = OUTDIR / "main_nsd_CV_bands.png"
fig.savefig(out, dpi=args.dpi, bbox_inches="tight"); plt.close(fig)
print(f"Saved: {out}")

print("Done.")
