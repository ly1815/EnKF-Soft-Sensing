"""
06_observability_gramian.py  —  empirical observability Gramian figure
=======================================================================
Ports the dimensionless empirical observability Gramian "test" from the JPC
notebook (JPC_NSD_softsensing.ipynb, cells 44/54) into this repository so the
main-text observability figure can be regenerated from the tracked model/data.

For each state i it perturbs the (dimensionless) initial condition by +/-epsilon,
propagates the full 17-state mechanistic model with the dataset's feed schedule
and volume trajectory, and integrates the squared change in the *measured* outputs:

    W_obs,ii = 1/(4 eps^2) * sum_k || y+(t_k) - y-(t_k) ||^2 * dt      (dimensionless)

Measured outputs are the 8 routinely available extracellular states
(Xv, mAb, Gal, Urd, Glc, Amm, Gln, Lac); Asn/Glu and the NSDs are unmeasured.

Figure (one per dataset, 12.5x5in, dpi 600 — same layout/style as the reference):
  (a) extracellular Gramian (log10); Asn hatched as unmeasured
  (b) all-NSD Gramian (log10); all hatched (unmeasured)
Bars are drawn in a calm grey (not the original orange).

Usage:
    ./.venv/bin/python scripts/06_observability_gramian.py                 # P1 (the paper figure)
    ./.venv/bin/python scripts/06_observability_gramian.py --datasets P1 P2 P3 P4
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.font_manager import FontProperties
from tqdm import tqdm

import nsd_enkf.config as cfg
from nsd_enkf.data_loader import (
    select_datasets, load_dataset, get_initial_condition, build_schedule,
)
from nsd_enkf.model import compute_volume_results, model_step

# ── Calm grey (replaces the original orange) ──────────────────────────────────
GREY = "#7f7f7f"

# ── CLI ───────────────────────────────────────────────────────────────────────
p = argparse.ArgumentParser(description="Empirical observability Gramian figure")
p.add_argument("--datasets", nargs="+", default=["P1"],
               help="datasets to compute/plot (default: P1 — the figure used in the paper)")
p.add_argument("--outdir", default="/Users/luxi.yu/Research/Soft_Sensing_Paper/Figs")
p.add_argument("--epsilon", default=0.01, type=float, help="perturbation size (dimensionless)")
p.add_argument("--dpi", default=600, type=int)
args = p.parse_args()

OUTDIR = Path(args.outdir); OUTDIR.mkdir(parents=True, exist_ok=True)
STATE_NAMES = list(cfg.STATE_NAMES)
MEASURED = ["Xv", "mAb", "Gal", "Urd", "Glc", "Amm", "Gln", "Lac"]

# ── Fixed grids / volume (mirrors scripts/04_cross_validate.py) ────────────────
time_grid = np.arange(cfg.DT, cfg.T_END + cfg.DT, cfg.DT)
step_len = np.full(len(time_grid), cfg.DT)
N_model = int(cfg.T_END / cfg.DT)
T_model = np.linspace(0, cfg.T_END, N_model + 1)
volume_results = compute_volume_results(select_datasets(*args.datasets), cfg.INITIAL_VOLUMES,
                                         build_schedule, step_len)


# ── Gramian computation (ported from notebook cell 44) ────────────────────────
def compute_gramian(ds_name, epsilon=0.01, clip_value=1e-12):
    """Dimensionless empirical observability Gramian diagonal (all 17 states)."""
    name_to_idx = {n: i for i, n in enumerate(STATE_NAMES)}
    measured_indices = [name_to_idx[n] for n in MEASURED]
    n_states = len(STATE_NAMES)
    dt = float(cfg.DT)

    d = load_dataset(ds_name)
    _, x0_dim = get_initial_condition(d["met_df"], d["nsd_df"])
    x0_dim = np.asarray(x0_dim, dtype=float)

    # Dimensionless scaling: each state by max(|x0|, 1)
    state_scale = np.maximum(np.abs(x0_dim), 1.0)
    x0_dimless = x0_dim / state_scale
    output_scale = state_scale[measured_indices]

    Fin, Fout, Gal_feed, Urd_feed = build_schedule(ds_name)
    V_traj = np.asarray(volume_results[ds_name])[1:]
    n_steps = min(len(Fin), len(T_model) - 1)

    def simulate_dimless(x_dimless_init):
        x_dim = x_dimless_init * state_scale
        traj = [x_dim.copy()]
        for k in range(n_steps):
            controls_k = {"Fin": Fin[k], "Fout": Fout[k], "V": V_traj[k],
                          "Gal_feed": Gal_feed[k], "Urd_feed": Urd_feed[k]}
            x_dim = model_step(x_dim, T_model[k], controls_k, dt)
            x_dim = np.clip(x_dim, clip_value, None)
            traj.append(x_dim.copy())
        return np.array(traj)

    Wo = np.zeros(n_states)
    for i in tqdm(range(n_states), desc=f"Gramian {ds_name}", leave=False):
        xp = x0_dimless.copy(); xp[i] += epsilon
        xm = x0_dimless.copy(); xm[i] -= epsilon
        Yp = simulate_dimless(xp)[:, measured_indices] / output_scale
        Ym = simulate_dimless(xm)[:, measured_indices] / output_scale
        Wo[i] = np.sum((Yp - Ym) ** 2) * dt / (4 * epsilon ** 2)
    return Wo


# ── Figure (ported from notebook cell 54; bars in calm grey) ──────────────────
def plot_gramian(ds_name, Wo, log_offset=1e-20, nsd_hatch="//",
                 exclude_last_n_states=7, exclude_states_plot=("Glu",),
                 asn_name="Asn",
                 selected_nsds=("UDPGal", "UDPGalNAc", "UDPGlc", "UDPGlcNAc",
                                "GDPMan", "GDPFuc", "CMPNeu5Ac"),
                 selected_nsd_labels=("UDP-Gal", "UDP-GalNAc", "UDP-Glc", "UDP-GlcNAc",
                                      "GDP-Man", "GDP-Fuc", "CMP-Neu5Ac")):
    Wo = np.asarray(Wo, dtype=float)

    # panel (a): extracellular states (drop the NSD block + Glu)
    visible_n = len(STATE_NAMES) - int(exclude_last_n_states)
    excl = set(exclude_states_plot)
    idx_a = [i for i in range(visible_n) if STATE_NAMES[i] not in excl]
    labels_a = [STATE_NAMES[i] for i in idx_a]
    asn_pos = idx_a.index(STATE_NAMES.index(asn_name)) if asn_name in STATE_NAMES else None

    # panel (b): selected NSDs (all unmeasured)
    name_to_idx = {n: i for i, n in enumerate(STATE_NAMES)}
    idx_b = [name_to_idx[n] for n in selected_nsds]
    labels_b = list(selected_nsd_labels)

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(12.5, 5))

    # (a)
    ya = np.log10(Wo[idx_a] + log_offset)
    bars_a = ax_a.bar(np.arange(len(idx_a)), ya, color=GREY, edgecolor="black", linewidth=1.2)
    if asn_pos is not None:
        bars_a[asn_pos].set_hatch(nsd_hatch)
        bars_a[asn_pos].set_linewidth(2.2)
    ax_a.axhline(0, color="black", linestyle="--", linewidth=1)
    ax_a.set_title("(a)", fontsize=14, fontweight="bold", loc="left")
    ax_a.set_xticks(np.arange(len(idx_a)))
    ax_a.set_xticklabels(labels_a, rotation=90, fontsize=12, fontweight="bold")
    ax_a.set_ylabel("log10(Observability Score)", fontsize=14, fontweight="bold")
    ax_a.grid(alpha=0.12)
    ax_a.tick_params(axis="y", labelsize=12)
    for lab in ax_a.get_yticklabels():
        lab.set_fontweight("bold")
    for spine in ax_a.spines.values():
        spine.set_linewidth(1.5)

    # (b)
    yb = np.log10(Wo[idx_b] + log_offset)
    bars_b = ax_b.bar(np.arange(len(idx_b)), yb, color=GREY, edgecolor="black", linewidth=1.2)
    for bb in bars_b:
        bb.set_hatch(nsd_hatch)
        bb.set_linewidth(2.0)
    ax_b.axhline(0, color="black", linestyle="--", linewidth=1)
    ax_b.set_title("(b)", fontsize=14, fontweight="bold", loc="left")
    ax_b.set_xticks(np.arange(len(idx_b)))
    ax_b.set_xticklabels(labels_b, rotation=45, ha="right", fontsize=12, fontweight="bold")
    ax_b.set_ylabel("log10(Observability Score)", fontsize=14, fontweight="bold")
    ax_b.grid(alpha=0.12)
    ax_b.tick_params(axis="y", labelsize=12)
    for lab in ax_b.get_yticklabels():
        lab.set_fontweight("bold")
    for spine in ax_b.spines.values():
        spine.set_linewidth(1.5)

    # shared legend: measured (solid) vs unmeasured (hatched)
    handles = [
        Patch(facecolor=GREY, edgecolor="black", label="Measured"),
        Patch(facecolor=GREY, edgecolor="black", hatch=nsd_hatch, linewidth=2.0, label="Unmeasured"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=2,
               frameon=False, prop=FontProperties(size=12, weight="bold"))

    plt.tight_layout(rect=[0, 0.06, 1, 0.95])
    out = OUTDIR / f"Observability_Extracellular_vs_NSD_{ds_name}.png"
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


for ds in args.datasets:
    Wo = compute_gramian(ds, epsilon=args.epsilon)
    print(f"\n{ds} Gramian (log10):")
    for n, v in zip(STATE_NAMES, Wo):
        print(f"  {n:12s} {np.log10(v + 1e-20):+7.2f}")
    plot_gramian(ds, Wo)

print("Done.")
