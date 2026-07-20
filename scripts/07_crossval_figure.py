"""
07_crossval_figure.py  —  cross-validation generalisation figure
=================================================================
Compact main-message figure for the cross-validation: for each evaluated
dataset it overlays the open-loop model against the EnKF result obtained under
every training fold, distinguishing the fold in which the dataset was the
tuning set (in-sample) from the folds in which it was held out (validation).

The two panels show the mean normalised RMSE over the eight measured
extracellular states and over the three reported nucleotide sugar donors
(UDP-Gal, UDP-Glc, UDP-GlcNAc). Reads from the seed-averaged agg pkls:
  in-sample  : results_multirun_nsd/fold_<D>/agg/alpha_0.02.pkl        (tuned on D, evaluated on D)
  held-out   : results_multirun_validation/fold_<f>/agg/heldout_<D>.pkl (tuned on f, evaluated on D)

Usage:
    ./.venv/bin/python scripts/07_crossval_figure.py
"""
import argparse, pickle, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.font_manager import FontProperties

import nsd_enkf.config as cfg
from nsd_enkf.data_loader import load_dataset

p = argparse.ArgumentParser(description="Cross-validation generalisation figure")
p.add_argument("--outdir", default="/Users/luxi.yu/Research/Soft_Sensing_Paper/Figs")
p.add_argument("--dpi", default=300, type=int)
args = p.parse_args()
OUTDIR = Path(args.outdir); OUTDIR.mkdir(parents=True, exist_ok=True)

FOLDS = ["P1", "P2", "P3", "P4"]
Tm = np.array(cfg.T_MEAS_FIXED)
# (name, state index, measurement column, source)
MEAS = [("Xv",0,0,"met"),("mAb",1,1,"met"),("Gal",2,2,"met"),("Urd",3,3,"met"),
        ("Glc",4,4,"met"),("Amm",5,5,"met"),("Gln",6,6,"met"),("Lac",7,7,"met")]
ASN  = [("Asn",8,8,"met")]
NSD  = [("UDP-Gal",10,0,"nsd"),("UDP-GalNAc",11,1,"nsd"),("UDP-Glc",12,2,"nsd"),
        ("UDP-GlcNAc",13,3,"nsd"),("GDP-Man",14,4,"nsd"),("GDP-Fuc",15,5,"nsd"),
        ("CMP-Neu5Ac",16,6,"nsd")]

# ── Bold, paper-figure fonts ──────────────────────────────────────────────────
LABEL_FS, TICK_FS, TITLE_FS, LEG_FS = 14, 12, 14, 12
LEG_FP = FontProperties(size=LEG_FS, weight="bold")
BLUE, GREY = "tab:blue", "0.5"


def _meas(D):
    d = load_dataset(D)
    sm = np.asarray(d["set_meas"], dtype=float)
    nsd = pd.DataFrame(d["NSD_meas"]).apply(pd.to_numeric, errors="coerce").to_numpy()
    return sm, nsd

MEASDATA = {D: _meas(D) for D in FOLDS}


def _nrmse(traj, T, idx, mv):
    pred = np.interp(Tm, T, traj[:, idx]); m = ~np.isnan(mv)
    if m.sum() == 0:
        return np.nan
    return np.sqrt(np.mean((mv[m] - pred[m]) ** 2)) / (np.mean(np.abs(mv[m])) or 1.0)


def _agg(f, D):
    q = (f"results_multirun_nsd/fold_{f}/agg/alpha_0.02.pkl" if D == f
         else f"results_multirun_validation/fold_{f}/agg/heldout_{D}.pkl")
    return pickle.load(open(cfg.PROJECT_ROOT / q, "rb"))


def _mean_nrmse(traj, T, states, D):
    sm, nsd = MEASDATA[D]
    return np.nanmean([_nrmse(traj, T, si, (sm[:, c] if s == "met" else nsd[:, c]))
                       for _, si, c, s in states])


def _style(ax, title, ylabel):
    ax.set_title(title, fontsize=TITLE_FS, fontweight="bold")
    ax.set_xlabel("Evaluated dataset", fontsize=LABEL_FS, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=LABEL_FS, fontweight="bold")
    ax.set_xticks(range(4)); ax.set_xticklabels(FOLDS)
    ax.tick_params(axis="both", labelsize=TICK_FS)
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_fontweight("bold")
    for sp in ax.spines.values():
        sp.set_linewidth(1.5)
    ax.grid(alpha=0.25); ax.set_xlim(-0.5, 3.5)


fig, axs = plt.subplots(1, 3, figsize=(16, 4.4))
for ax, (title, states) in zip(
        axs, [("(a) Measured extracellular states", MEAS),
              ("(b) Asparagine", ASN),
              ("(c) Nucleotide sugar donors", NSD)]):
    for xi, D in enumerate(FOLDS):
        a0 = _agg([f for f in FOLDS if f != D][0], D)
        mo = _mean_nrmse(np.asarray(a0["model_trajectory"]), np.asarray(a0["T"]), states, D)
        ax.plot(xi, mo, marker="s", ms=10, color=GREY, mec="black", mew=1.0, zorder=3)
        for k, f in enumerate(FOLDS):
            v = _mean_nrmse(np.asarray(_agg(f, D)["avg_mean_trajectory"]),
                            np.asarray(_agg(f, D)["T"]), states, D)
            xj = xi + (k - 1.5) * 0.13
            if f == D:
                ax.plot(xj, v, marker="o", ms=9, mfc=BLUE, mec="black", mew=1.2, zorder=4)
            else:
                ax.plot(xj, v, marker="o", ms=8, mfc="none", mec=BLUE, mew=1.6, zorder=4)
    _style(ax, title, "Mean normalised RMSE")

handles = [
    Line2D([], [], marker="s", ms=10, color=GREY, mec="black", ls="none", label="Open-loop model"),
    Line2D([], [], marker="o", ms=9, mfc=BLUE, mec="black", ls="none", label="EnKF, tuning (in-sample)"),
    Line2D([], [], marker="o", ms=8, mfc="none", mec=BLUE, mew=1.6, ls="none", label="EnKF, validation (held-out)"),
]
fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
           bbox_to_anchor=(0.5, -0.02), prop=LEG_FP)
plt.tight_layout(rect=[0, 0.07, 1, 1])
out = OUTDIR / "crossval_nrmse.png"
fig.savefig(out, dpi=args.dpi, bbox_inches="tight"); plt.close(fig)
print(f"Saved: {out}")

# ── Per-fold RMSE tables in the format of Table S8, one per training fold ──────
# (name, unit, state index, measurement column, source), in the row order of Table S8.
ORDER = [("Xv", r"cell $L^{-1}$",0,0,"met"), ("mAb", r"mg $L^{-1}$",1,1,"met"),
         ("Gal","mM",2,2,"met"), ("Urd","mM",3,3,"met"), ("Glc","mM",4,4,"met"),
         ("Amm","mM",5,5,"met"), ("Gln","mM",6,6,"met"), ("Lac","mM",7,7,"met"),
         ("Asn","mM",8,8,"met"),
         ("UDP-Gal","mM",10,0,"nsd"), ("UDP-Glc","mM",12,2,"nsd"), ("UDP-GlcNAc","mM",13,3,"nsd"),
         ("UDP-GalNAc","mM",11,1,"nsd"), ("GDP-Man","mM",14,4,"nsd"),
         ("GDP-Fuc","mM",15,5,"nsd"), ("CMP-Neu5Ac","mM",16,6,"nsd")]
NSD_START = 9   # index in ORDER at which the nucleotide-sugar-donor block begins


def _rmse(traj, T, idx, mv):
    pred = np.interp(Tm, T, traj[:, idx]); m = ~np.isnan(mv)
    if m.sum() == 0:
        return np.nan
    return float(np.sqrt(np.mean((mv[m] - pred[m]) ** 2)))


def _fmt(v):
    return f"\\num{{{v:.2e}}}" if v >= 1e4 else f"\\num{{{v:.2f}}}"


def _rmse_of(traj_key, f, D, si, col, src):
    a = _agg(f, D); sm, nsd = MEASDATA[D]
    return _rmse(np.asarray(a[traj_key]), np.asarray(a["T"]), si,
                 (sm[:, col] if src == "met" else nsd[:, col]))

def model_rmse(D, si, col, src):
    return _rmse_of("model_trajectory", [g for g in FOLDS if g != D][0], D, si, col, src)

def enkf_rmse(f, D, si, col, src):
    return _rmse_of("avg_mean_trajectory", f, D, si, col, src)


tables_path = Path("/Users/luxi.yu/Research/Soft_Sensing_Paper/Sections/SI_crossval_tables.tex")
lines = ["% Auto-generated by scripts/07_crossval_figure.py -- do not edit by hand.", ""]
for f in FOLDS:
    top = " ".join(r"& \multicolumn{2}{c}{\textbf{" + (D + r"$^{*}$" if D == f else D) + r"}}"
                   for D in FOLDS)
    cmid = " ".join(r"\cmidrule(lr){%d-%d}" % (2 + 2 * i, 3 + 2 * i) for i in range(4))
    sub = " ".join(r"& \textbf{Model} & \textbf{EnKF}" for _ in FOLDS)
    lines += [r"\begin{table}[htbp]", "{", r"\color{blue}",
              r"\renewcommand{\arraystretch}{1.4}", r"\setlength{\tabcolsep}{10pt}", "",
              r"\centering", r"\Large",
              (r"\caption{Cross-validation fold tuned on " + f + r": seed-averaged RMSE of the "
               r"open-loop model and the EnKF against the experimental measurements for each dataset. "
               r"The tuning set " + f + r", marked with an asterisk, is in-sample; the remaining "
               r"datasets are held out for validation.}"),
              (r"\label{tab:cv_fold_" + f + r"}"),
              r"\begin{adjustbox}{max width=\textwidth}",
              r"\begin{tabular}{lrrrrrrrr}", r"\toprule",
              r"\textbf{State (unit)} " + top + r" \\",
              cmid,
              sub + r" \\", r"\midrule"]
    for i, (nm, unit, si, col, src) in enumerate(ORDER):
        if i == NSD_START:
            lines.append(r"\midrule")
        parts = []
        for D in FOLDS:
            parts.append(_fmt(model_rmse(D, si, col, src)))
            parts.append(_fmt(enkf_rmse(f, D, si, col, src)))
        lines.append(f"{nm} ({unit}) & " + " & ".join(parts) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}", "}", r"\end{table}", ""]
tables_path.write_text("\n".join(lines))
print(f"Saved: {tables_path}")
