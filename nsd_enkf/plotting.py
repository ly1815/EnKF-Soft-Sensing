"""
plotting.py
===========
Publication-quality plotting functions for EnKF soft sensing results.

Every function accepts an optional *save_path* (Path or str).
Pass None to display without saving.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — safe in scripts
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from pathlib import Path


def _savefig(fig, save_path, dpi=300):
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close(fig)


# ─── NSD state indices ──────────────────────────────────────────────────────

NSD_STATES_GLOBAL = {
    "UDPGal": 10, "UDPGalNAc": 11, "UDPGlc": 12, "UDPGlcNAc": 13,
    "GDPMan": 14, "GDPFuc": 15, "CMPNeu5Ac": 16,
}

NSD_STATES_LOCAL = {
    "UDPGal": 0, "UDPGalNAc": 1, "UDPGlc": 2, "UDPGlcNAc": 3,
    "GDPMan": 4, "GDPFuc": 5, "CMPNeu5Ac": 6,
}

NSD_ORDER = ["UDPGal", "UDPGalNAc", "UDPGlc", "UDPGlcNAc",
             "GDPMan", "GDPFuc", "CMPNeu5Ac"]


# ─── NSD grid plot (all NSDs x all datasets) ────────────────────────────────

def plot_nsd_grid(
    set_model_by_dataset, enkf_results_by_dataset,
    enkf_runs_by_dataset, load_dataset_fn,
    T_model, T_kf, T_meas_by_dataset, axis_name,
    dataset_list=("P1", "P2", "P3", "P4"),
    plot_individual_runs=True, max_runs_to_plot=None,
    save_path=None,
):
    """Plot all NSD states across all datasets in a grid."""
    n_rows = len(NSD_ORDER)
    n_cols = len(dataset_list)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4.8 * n_cols, 3.9 * n_rows),
        sharex=True,
    )
    if n_rows == 1:
        axes = np.expand_dims(axes, 0)
    if n_cols == 1:
        axes = np.expand_dims(axes, 1)

    run_lw, run_alpha, mean_lw = 1.0, 0.18, 2.4

    for r, state_key in enumerate(NSD_ORDER):
        i_global = NSD_STATES_GLOBAL[state_key]
        j_local = NSD_STATES_LOCAL[state_key]

        for c, name in enumerate(dataset_list):
            ax = axes[r, c]
            set_model = set_model_by_dataset[name]
            set_EnKF_mean = enkf_results_by_dataset[name]
            list_EnKF = enkf_runs_by_dataset.get(name, None)

            data = load_dataset_fn(name)
            NSD_meas = data["NSD_meas"]
            NSD_meas_errorbar = data["NSD_meas_errorbar"]
            T_meas = T_meas_by_dataset[name]

            ax.plot(T_model, set_model[:, i_global], color="red", lw=2.2)

            if plot_individual_runs and list_EnKF is not None:
                runs = list_EnKF if max_runs_to_plot is None else list_EnKF[:max_runs_to_plot]
                for run_arr in runs:
                    ax.plot(T_kf, run_arr[:, i_global], color="black",
                            lw=run_lw, alpha=run_alpha)

            ax.plot(T_kf, set_EnKF_mean[:, i_global], color="black",
                    ls="--", lw=mean_lw)

            ax.errorbar(
                T_meas, NSD_meas[:, j_local], yerr=NSD_meas_errorbar[:, j_local],
                fmt="o", color="orange", markersize=4.5, capsize=2,
                elinewidth=1, alpha=0.9,
            )

            if r == 0:
                ax.set_title(name, fontsize=13, fontweight="bold")
            if c == 0:
                ax.set_ylabel(axis_name[i_global], fontsize=11, fontweight="bold")
            if r == n_rows - 1:
                ax.set_xlabel("Time (hours)", fontsize=11, fontweight="bold")

            ax.grid(alpha=0.12)
            ax.tick_params(axis="both", labelsize=10)
            for lab in ax.get_xticklabels() + ax.get_yticklabels():
                lab.set_fontweight("bold")
            for spine in ax.spines.values():
                spine.set_linewidth(2)

    legend_elements = [
        Line2D([0], [0], color="red", lw=2.3, label="Mechanistic Model"),
        Line2D([0], [0], color="black", lw=run_lw, alpha=0.45,
               label="EnKF individual runs"),
        Line2D([0], [0], color="black", lw=mean_lw, ls="--", label="EnKF mean"),
        Line2D([0], [0], color="orange", marker="o", lw=0, markersize=6,
               label="Measurements"),
    ]
    fig.legend(
        handles=legend_elements, loc="lower center", ncol=4,
        fontsize=12, frameon=False, bbox_to_anchor=(0.5, -0.01),
    )
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    _savefig(fig, save_path)


# ─── Measured metabolites grid plot ──────────────────────────────────────────

def plot_measured_metabolites_grid(
    set_model_by_dataset, enkf_results_by_dataset,
    enkf_runs_by_dataset, load_dataset_fn,
    T_model, T_kf, T_meas_by_dataset,
    state_name, axis_name, meas_num,
    dataset_list=("P1", "P2", "P3", "P4"),
    dataset_colours=None, dataset_markers=None,
    plot_individual_runs=True, save_path=None,
):
    """Plot measured extracellular metabolites across all datasets."""
    n_rows = meas_num
    n_cols = len(dataset_list)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4.8 * n_cols, 3.2 * n_rows),
        sharex=True,
    )
    if n_rows == 1:
        axes = np.expand_dims(axes, 0)
    if n_cols == 1:
        axes = np.expand_dims(axes, 1)

    run_lw, run_alpha, mean_lw = 1.0, 0.18, 2.4

    for r in range(n_rows):
        for c, name in enumerate(dataset_list):
            ax = axes[r, c]
            colour = dataset_colours.get(name, "tab:blue") if dataset_colours else "tab:blue"

            set_model = set_model_by_dataset[name]
            set_EnKF_mean = enkf_results_by_dataset[name]
            list_EnKF = enkf_runs_by_dataset.get(name, None)
            data = load_dataset_fn(name)
            T_meas = T_meas_by_dataset[name]
            set_meas = data["set_meas"].astype(float)
            set_meas_err = data["set_meas_errorbar"].astype(float)

            ax.plot(T_model, set_model[:, r], color="red", lw=2.2)

            if plot_individual_runs and list_EnKF is not None:
                for run_arr in list_EnKF:
                    ax.plot(T_kf, run_arr[:, r], color="black",
                            lw=run_lw, alpha=run_alpha)

            ax.plot(T_kf, set_EnKF_mean[:, r], color="black", ls="--", lw=mean_lw)

            if r < set_meas.shape[1]:
                ax.errorbar(
                    T_meas, set_meas[:, r], yerr=set_meas_err[:, r],
                    fmt="o", color=colour, markersize=4.5, capsize=2,
                    elinewidth=1, alpha=0.9,
                )

            if r == 0:
                ax.set_title(name, fontsize=13, fontweight="bold")
            if c == 0:
                ax.set_ylabel(axis_name[r], fontsize=11, fontweight="bold")
            if r == n_rows - 1:
                ax.set_xlabel("Time (hours)", fontsize=11, fontweight="bold")

            ax.grid(alpha=0.12)
            ax.tick_params(axis="both", labelsize=10)
            for lab in ax.get_xticklabels() + ax.get_yticklabels():
                lab.set_fontweight("bold")
            for spine in ax.spines.values():
                spine.set_linewidth(2)

    plt.tight_layout()
    _savefig(fig, save_path)


# ─── Asparagine soft sensing plot ────────────────────────────────────────────

def plot_asn_soft_sensing(
    set_model_by_dataset, enkf_results_by_dataset,
    enkf_runs_by_dataset, load_dataset_fn,
    T_model, T_kf, T_meas_by_dataset,
    axis_name, asn_state_idx=8,
    dataset_list=("P1", "P2", "P3", "P4"),
    save_path=None,
):
    """Plot Asparagine (unmeasured) soft sensing results."""
    sns.set(style="white", context="talk")
    n = len(dataset_list)
    nrow, ncol = (2, 2) if n == 4 else (1, n)

    fig, axes = plt.subplots(nrow, ncol, figsize=(16, 9.2), sharex=False)
    axes = axes.flatten() if n > 1 else [axes]

    run_lw, run_alpha, mean_lw = 1.0, 0.18, 2.4

    for idx, name in enumerate(dataset_list):
        ax = axes[idx]
        set_model = set_model_by_dataset[name]
        set_EnKF_mean = enkf_results_by_dataset[name]
        list_EnKF = enkf_runs_by_dataset.get(name, None)
        data = load_dataset_fn(name)
        T_meas = T_meas_by_dataset[name]
        set_meas = data["set_meas"].astype(float)
        set_meas_err = data["set_meas_errorbar"].astype(float)

        ax.plot(T_model, set_model[:, asn_state_idx], color="red", lw=2.2)

        if list_EnKF is not None:
            for run_arr in list_EnKF:
                ax.plot(T_kf, run_arr[:, asn_state_idx], color="black",
                        lw=run_lw, alpha=run_alpha)

        ax.plot(T_kf, set_EnKF_mean[:, asn_state_idx], color="black",
                ls="--", lw=mean_lw)

        # Asn is the last column in set_meas (index -1)
        asn_col = set_meas.shape[1] - 1
        ax.errorbar(
            T_meas, set_meas[:, asn_col], yerr=set_meas_err[:, asn_col],
            fmt="o", color="orange", markersize=4.5, capsize=2,
            elinewidth=1, alpha=0.9,
        )

        ax.set_title(name, fontsize=13, fontweight="bold")
        ax.set_ylabel(axis_name[asn_state_idx], fontsize=11, fontweight="bold")
        ax.set_xlabel("Time (hours)", fontsize=11, fontweight="bold")
        ax.grid(alpha=0.12)

    plt.tight_layout()
    _savefig(fig, save_path)


# ─── Helper: ensemble statistics from runs ───────────────────────────────────

def _ensemble_stats(enkf_runs_by_dataset, name, state_idx, T_kf, downsample=10):
    """
    Compute mean, ±1 std, and ±2 std envelopes from the independent EnKF runs.

    Parameters
    ----------
    downsample : int
        Take every Nth point to reduce plotting overhead on 28 k-point arrays.

    Returns
    -------
    t, mean, lo1, hi1, lo2, hi2 : np.ndarray
    """
    runs = np.array(enkf_runs_by_dataset[name])  # (n_runs, n_t, n_states)
    vals = runs[:, ::downsample, state_idx]       # (n_runs, n_t_ds)
    t = T_kf[::downsample]
    mean = np.mean(vals, axis=0)
    std = np.std(vals, axis=0)
    return t, mean, mean - std, mean + std, mean - 2 * std, mean + 2 * std


def _style_ax(ax):
    """Apply consistent styling to an axis."""
    ax.grid(alpha=0.15)
    ax.tick_params(axis="both", labelsize=10)
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_fontweight("bold")
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)


# ─── Measured metabolites with uncertainty bands ─────────────────────────────

def plot_measured_metabolites_uncertainty(
    set_model_by_dataset, enkf_results_by_dataset,
    enkf_runs_by_dataset, load_dataset_fn,
    T_model, T_kf, T_meas_by_dataset,
    state_name, axis_name, meas_num,
    dataset_list=("P1", "P2", "P3", "P4"),
    dataset_colours=None,
    downsample=10, save_path=None,
):
    """
    Plot measured extracellular metabolites with shaded ±1 std and ±2 std
    uncertainty bands from the ensemble of EnKF runs.

    Addresses Reviewer 2 #3, Reviewer 1 #2, Reviewer 3 #1.
    """
    n_rows = meas_num
    n_cols = len(dataset_list)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(5.0 * n_cols, 3.0 * n_rows),
        sharex=True,
    )
    if n_rows == 1:
        axes = np.expand_dims(axes, 0)
    if n_cols == 1:
        axes = np.expand_dims(axes, 1)

    for r in range(n_rows):
        for c, name in enumerate(dataset_list):
            ax = axes[r, c]
            colour = (dataset_colours or {}).get(name, "tab:blue")

            # Model trajectory
            ax.plot(T_model[::downsample],
                    set_model_by_dataset[name][::downsample, r],
                    color="red", lw=1.8, label="Model" if (r == 0 and c == 0) else None)

            # EnKF uncertainty bands
            t, mean, lo1, hi1, lo2, hi2 = _ensemble_stats(
                enkf_runs_by_dataset, name, r, T_kf, downsample)

            ax.fill_between(t, lo2, hi2, color="steelblue", alpha=0.15, label=None)
            ax.fill_between(t, lo1, hi1, color="steelblue", alpha=0.30, label=None)
            ax.plot(t, mean, color="steelblue", lw=2.0, label=None)

            # Measurements with error bars
            data = load_dataset_fn(name)
            T_meas = T_meas_by_dataset[name]
            set_meas = data["set_meas"].astype(float)
            set_meas_err = data["set_meas_errorbar"].astype(float)
            if r < set_meas.shape[1]:
                ax.errorbar(
                    T_meas, set_meas[:, r], yerr=set_meas_err[:, r],
                    fmt="o", color=colour, markersize=4.0, capsize=2,
                    elinewidth=1, alpha=0.9, zorder=5,
                )

            if r == 0:
                ax.set_title(name, fontsize=13, fontweight="bold")
            if c == 0:
                ax.set_ylabel(axis_name[r], fontsize=10, fontweight="bold")
            if r == n_rows - 1:
                ax.set_xlabel("Time (hours)", fontsize=11, fontweight="bold")
            _style_ax(ax)

    legend_elements = [
        Line2D([0], [0], color="red", lw=1.8, label="Mechanistic model"),
        Line2D([0], [0], color="steelblue", lw=2.0, label="EnKF mean"),
        Patch(facecolor="steelblue", alpha=0.30, label=r"EnKF $\pm 1\sigma$"),
        Patch(facecolor="steelblue", alpha=0.15, label=r"EnKF $\pm 2\sigma$"),
        Line2D([0], [0], color="grey", marker="o", lw=0, markersize=5,
               label="Measurements"),
    ]
    fig.legend(
        handles=legend_elements, loc="lower center", ncol=5,
        fontsize=11, frameon=False, bbox_to_anchor=(0.5, -0.01),
    )
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    _savefig(fig, save_path)


# ─── NSD states with uncertainty bands ───────────────────────────────────────

def plot_nsd_uncertainty(
    set_model_by_dataset, enkf_results_by_dataset,
    enkf_runs_by_dataset, load_dataset_fn,
    T_model, T_kf, T_meas_by_dataset, axis_name,
    dataset_list=("P1", "P2", "P3", "P4"),
    downsample=10, save_path=None,
):
    """
    Plot NSD (unmeasured intracellular) states with shaded ±1 std and ±2 std
    uncertainty bands.

    Addresses Reviewer 3 #1, #3 — uncertainty on structurally unobservable states.
    """
    n_rows = len(NSD_ORDER)
    n_cols = len(dataset_list)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(5.0 * n_cols, 3.2 * n_rows),
        sharex=True,
    )
    if n_rows == 1:
        axes = np.expand_dims(axes, 0)
    if n_cols == 1:
        axes = np.expand_dims(axes, 1)

    for r, state_key in enumerate(NSD_ORDER):
        i_global = NSD_STATES_GLOBAL[state_key]
        j_local = NSD_STATES_LOCAL[state_key]

        for c, name in enumerate(dataset_list):
            ax = axes[r, c]

            # Model
            ax.plot(T_model[::downsample],
                    set_model_by_dataset[name][::downsample, i_global],
                    color="red", lw=1.8)

            # EnKF bands
            t, mean, lo1, hi1, lo2, hi2 = _ensemble_stats(
                enkf_runs_by_dataset, name, i_global, T_kf, downsample)

            ax.fill_between(t, lo2, hi2, color="steelblue", alpha=0.15)
            ax.fill_between(t, lo1, hi1, color="steelblue", alpha=0.30)
            ax.plot(t, mean, color="steelblue", lw=2.0)

            # NSD measurements
            data = load_dataset_fn(name)
            NSD_meas = data["NSD_meas"]
            NSD_meas_err = data["NSD_meas_errorbar"]
            T_meas = T_meas_by_dataset[name]

            ax.errorbar(
                T_meas, NSD_meas[:, j_local], yerr=NSD_meas_err[:, j_local],
                fmt="o", color="darkorange", markersize=4.0, capsize=2,
                elinewidth=1, alpha=0.9, zorder=5,
            )

            if r == 0:
                ax.set_title(name, fontsize=13, fontweight="bold")
            if c == 0:
                ax.set_ylabel(axis_name[i_global], fontsize=10, fontweight="bold")
            if r == n_rows - 1:
                ax.set_xlabel("Time (hours)", fontsize=11, fontweight="bold")
            _style_ax(ax)

    legend_elements = [
        Line2D([0], [0], color="red", lw=1.8, label="Mechanistic model"),
        Line2D([0], [0], color="steelblue", lw=2.0, label="EnKF mean"),
        Patch(facecolor="steelblue", alpha=0.30, label=r"EnKF $\pm 1\sigma$"),
        Patch(facecolor="steelblue", alpha=0.15, label=r"EnKF $\pm 2\sigma$"),
        Line2D([0], [0], color="darkorange", marker="o", lw=0, markersize=5,
               label="Measurements"),
    ]
    fig.legend(
        handles=legend_elements, loc="lower center", ncol=5,
        fontsize=11, frameon=False, bbox_to_anchor=(0.5, -0.01),
    )
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    _savefig(fig, save_path)


# ─── Asparagine + Glutamate with uncertainty bands ───────────────────────────

def plot_unmeasured_extracellular_uncertainty(
    set_model_by_dataset, enkf_results_by_dataset,
    enkf_runs_by_dataset, load_dataset_fn,
    T_model, T_kf, T_meas_by_dataset, axis_name,
    state_indices=(8, 9), state_labels=("Asn", "Glu"),
    dataset_list=("P1", "P2", "P3", "P4"),
    downsample=10, save_path=None,
):
    """
    Plot unmeasured extracellular states (Asn, Glu) with uncertainty bands.

    These states are not assimilated in the EnKF update step but are
    estimated via model dynamics — making uncertainty analysis crucial.
    """
    n_rows = len(state_indices)
    n_cols = len(dataset_list)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(5.0 * n_cols, 3.5 * n_rows),
        sharex=True,
    )
    if n_rows == 1:
        axes = np.expand_dims(axes, 0)
    if n_cols == 1:
        axes = np.expand_dims(axes, 1)

    for r, (si, sl) in enumerate(zip(state_indices, state_labels)):
        for c, name in enumerate(dataset_list):
            ax = axes[r, c]

            # Model
            ax.plot(T_model[::downsample],
                    set_model_by_dataset[name][::downsample, si],
                    color="red", lw=1.8)

            # EnKF bands
            t, mean, lo1, hi1, lo2, hi2 = _ensemble_stats(
                enkf_runs_by_dataset, name, si, T_kf, downsample)

            ax.fill_between(t, lo2, hi2, color="steelblue", alpha=0.15)
            ax.fill_between(t, lo1, hi1, color="steelblue", alpha=0.30)
            ax.plot(t, mean, color="steelblue", lw=2.0)

            # Measurements (Asn is last col in set_meas; Glu has no direct meas)
            data = load_dataset_fn(name)
            T_meas = T_meas_by_dataset[name]
            set_meas = data["set_meas"].astype(float)
            set_meas_err = data["set_meas_errorbar"].astype(float)

            if sl == "Asn":
                asn_col = set_meas.shape[1] - 1
                ax.errorbar(
                    T_meas, set_meas[:, asn_col], yerr=set_meas_err[:, asn_col],
                    fmt="o", color="darkorange", markersize=4.0, capsize=2,
                    elinewidth=1, alpha=0.9, zorder=5,
                )

            if r == 0:
                ax.set_title(name, fontsize=13, fontweight="bold")
            if c == 0:
                ax.set_ylabel(axis_name[si], fontsize=10, fontweight="bold")
            if r == n_rows - 1:
                ax.set_xlabel("Time (hours)", fontsize=11, fontweight="bold")
            _style_ax(ax)

    legend_elements = [
        Line2D([0], [0], color="red", lw=1.8, label="Mechanistic model"),
        Line2D([0], [0], color="steelblue", lw=2.0, label="EnKF mean"),
        Patch(facecolor="steelblue", alpha=0.30, label=r"EnKF $\pm 1\sigma$"),
        Patch(facecolor="steelblue", alpha=0.15, label=r"EnKF $\pm 2\sigma$"),
        Line2D([0], [0], color="darkorange", marker="o", lw=0, markersize=5,
               label="Measurements (validation)"),
    ]
    fig.legend(
        handles=legend_elements, loc="lower center", ncol=5,
        fontsize=11, frameon=False, bbox_to_anchor=(0.5, -0.01),
    )
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    _savefig(fig, save_path)


# ─── Ensemble spread evolution ───────────────────────────────────────────────

def plot_ensemble_spread_evolution(
    enkf_runs_by_dataset, T_kf, state_name, axis_name,
    state_indices=None,
    dataset_list=("P1", "P2", "P3", "P4"),
    downsample=50, save_path=None,
):
    """
    Plot evolution of ensemble standard deviation over time for selected states.

    Directly addresses Reviewer 3 #4 — whether the ensemble maintains
    meaningful spread or collapses during filtering.
    """
    if state_indices is None:
        # Default: Xv, Glc, Asn, UDPGlcNAc
        state_indices = [0, 4, 8, 13]

    n_rows = len(state_indices)
    n_cols = len(dataset_list)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(5.0 * n_cols, 2.8 * n_rows),
        sharex=True,
    )
    if n_rows == 1:
        axes = np.expand_dims(axes, 0)
    if n_cols == 1:
        axes = np.expand_dims(axes, 1)

    for r, si in enumerate(state_indices):
        for c, name in enumerate(dataset_list):
            ax = axes[r, c]
            runs = np.array(enkf_runs_by_dataset[name])
            vals = runs[:, ::downsample, si]
            t = T_kf[::downsample]
            std = np.std(vals, axis=0)

            ax.plot(t, std, color="steelblue", lw=1.8)
            ax.fill_between(t, 0, std, color="steelblue", alpha=0.2)

            if r == 0:
                ax.set_title(name, fontsize=13, fontweight="bold")
            if c == 0:
                ax.set_ylabel(f"Std({state_name[si]})", fontsize=10, fontweight="bold")
            if r == n_rows - 1:
                ax.set_xlabel("Time (hours)", fontsize=11, fontweight="bold")
            _style_ax(ax)

    plt.tight_layout()
    _savefig(fig, save_path)


# ─── Within-run ensemble spread (single-run diagnostics) ─────────────────────

def plot_ensemble_std_trajectory(
    std_trajectory, T_kf, state_name, axis_name,
    T_meas=None,
    state_indices=None,
    dataset_name="",
    downsample=10, save_path=None,
):
    """
    Plot the within-run ensemble standard deviation over time.

    Shows how the N=100 ensemble members' spread evolves — whether it
    collapses (bad) or maintains meaningful spread throughout filtering.

    Addresses Reviewer 3 #4.
    """
    if state_indices is None:
        state_indices = list(range(len(state_name)))

    n_states = len(state_indices)
    ncol = 4
    nrow = int(np.ceil(n_states / ncol))

    fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 3.0 * nrow), sharex=True)
    axes = axes.flatten()

    t = T_kf[:len(std_trajectory)][::downsample]
    for idx, si in enumerate(state_indices):
        ax = axes[idx]
        std_ds = std_trajectory[::downsample, si]

        ax.plot(t, std_ds, color="steelblue", lw=1.5)
        ax.fill_between(t, 0, std_ds, color="steelblue", alpha=0.2)

        # Mark measurement update times
        if T_meas is not None:
            for tm in T_meas:
                ax.axvline(tm, color="darkorange", lw=0.6, alpha=0.5, ls="--")

        ax.set_title(f"{state_name[si]}", fontsize=11, fontweight="bold")
        if idx % ncol == 0:
            ax.set_ylabel(r"Ensemble $\sigma$", fontsize=10, fontweight="bold")
        if idx >= (nrow - 1) * ncol:
            ax.set_xlabel("Time (hours)", fontsize=10, fontweight="bold")
        _style_ax(ax)

    # Hide unused axes
    for idx in range(n_states, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(
        f"Within-run ensemble spread (N=100) — {dataset_name}",
        fontsize=14, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    _savefig(fig, save_path)


def plot_ensemble_snapshots(
    ensemble_at_updates, state_name, axis_name,
    state_indices=None,
    snapshot_times=None,
    dataset_name="",
    save_path=None,
):
    """
    Plot violin/box plots of the ensemble distribution at selected
    measurement update times (forecast vs analysis).

    Shows whether the 100 ensemble members maintain diversity or collapse.

    Addresses Reviewer 1 #2, Reviewer 3 #1 and #4.
    """
    if state_indices is None:
        state_indices = [0, 4, 8, 13]  # Xv, Glc, Asn, UDPGlcNAc

    # Filter to analysis snapshots only (post-update)
    analysis_snaps = [s for s in ensemble_at_updates if "analysis" in s["label"]]
    if not analysis_snaps:
        analysis_snaps = ensemble_at_updates[1:]  # fallback

    if snapshot_times is not None:
        analysis_snaps = [s for s in analysis_snaps if s["time"] in snapshot_times]

    # Also include initial
    initial = ensemble_at_updates[0]

    # Select a manageable number of snapshots (every 2nd or 3rd)
    if len(analysis_snaps) > 8:
        step = max(1, len(analysis_snaps) // 8)
        analysis_snaps = analysis_snaps[::step]

    all_snaps = [initial] + analysis_snaps
    n_snaps = len(all_snaps)
    n_states = len(state_indices)

    fig, axes = plt.subplots(
        n_states, 1,
        figsize=(max(12, 1.4 * n_snaps), 3.2 * n_states),
        sharex=True,
    )
    if n_states == 1:
        axes = [axes]

    time_labels = []
    for s in all_snaps:
        t = s["time"]
        if t == 0:
            time_labels.append("t=0")
        else:
            time_labels.append(f"{t:.0f}h")

    positions = np.arange(n_snaps)

    for row, si in enumerate(state_indices):
        ax = axes[row]
        data_for_violin = []

        for s in all_snaps:
            ens = s["ensemble"][:, si]
            data_for_violin.append(ens)

        parts = ax.violinplot(
            data_for_violin, positions=positions,
            showmeans=True, showmedians=False, showextrema=False,
        )

        for pc in parts['bodies']:
            pc.set_facecolor("steelblue")
            pc.set_alpha(0.4)
        parts['cmeans'].set_color("steelblue")
        parts['cmeans'].set_linewidth(2)

        # Overlay box plots for quartiles
        bp = ax.boxplot(
            data_for_violin, positions=positions,
            widths=0.3, patch_artist=True,
            showfliers=False, zorder=3,
        )
        for patch in bp['boxes']:
            patch.set_facecolor("steelblue")
            patch.set_alpha(0.3)
        for element in ['whiskers', 'caps', 'medians']:
            for line in bp[element]:
                line.set_color("steelblue")
                line.set_linewidth(1.2)

        ax.set_ylabel(axis_name[si], fontsize=10, fontweight="bold")
        _style_ax(ax)

    axes[-1].set_xticks(positions)
    axes[-1].set_xticklabels(time_labels, rotation=45, ha="right", fontsize=9)
    axes[-1].set_xlabel("Measurement update time", fontsize=11, fontweight="bold")

    fig.suptitle(
        f"Ensemble distribution at measurement updates (N=100) — {dataset_name}",
        fontsize=13, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    _savefig(fig, save_path)


def plot_forecast_vs_analysis_spread(
    ensemble_at_updates, state_name, axis_name,
    state_indices=None,
    dataset_name="",
    save_path=None,
):
    """
    Plot ensemble std before (forecast) and after (analysis) each measurement
    update, showing how the update step reduces spread for observed states
    and how it propagates to unobserved states.
    """
    if state_indices is None:
        state_indices = [0, 4, 8, 13]  # Xv, Glc, Asn, UDPGlcNAc

    # Pair up forecast/analysis at the same time
    pairs = []
    i = 1  # skip initial
    while i < len(ensemble_at_updates) - 1:
        s1 = ensemble_at_updates[i]
        s2 = ensemble_at_updates[i + 1]
        if "forecast" in s1["label"] and "analysis" in s2["label"]:
            pairs.append((s1, s2))
            i += 2
        else:
            i += 1

    times = [p[0]["time"] for p in pairs]
    n_states = len(state_indices)

    fig, axes = plt.subplots(
        n_states, 1,
        figsize=(12, 3.0 * n_states),
        sharex=True,
    )
    if n_states == 1:
        axes = [axes]

    for row, si in enumerate(state_indices):
        ax = axes[row]
        fc_std = [np.std(p[0]["ensemble"][:, si]) for p in pairs]
        an_std = [np.std(p[1]["ensemble"][:, si]) for p in pairs]

        x = np.arange(len(times))
        w = 0.35
        ax.bar(x - w / 2, fc_std, w, color="salmon", alpha=0.8, label="Forecast")
        ax.bar(x + w / 2, an_std, w, color="steelblue", alpha=0.8, label="Analysis")

        ax.set_ylabel(f"Std({state_name[si]})", fontsize=10, fontweight="bold")
        if row == 0:
            ax.legend(fontsize=10, frameon=False)
        _style_ax(ax)

    axes[-1].set_xticks(np.arange(len(times)))
    axes[-1].set_xticklabels([f"{t:.0f}h" for t in times], rotation=45, ha="right", fontsize=9)
    axes[-1].set_xlabel("Measurement update time", fontsize=11, fontweight="bold")

    fig.suptitle(
        f"Forecast vs analysis ensemble spread — {dataset_name}",
        fontsize=13, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    _savefig(fig, save_path)


# ─── Within-run ensemble uncertainty on concentration profiles ───────────────

def _ensemble_profile_stats(diag_entry, state_idx, T_kf, downsample=10):
    """
    Mean ± 1σ/2σ from within-run ensemble diagnostics.

    diag_entry can be:
      - a single dict with "mean_trajectory" and "std_trajectory"
      - a list of such dicts (one per run) — aggregates across runs
    """
    if isinstance(diag_entry, list):
        # Multiple runs: average the mean, RMS-average the std
        means = np.array([d["mean_trajectory"][::downsample, state_idx] for d in diag_entry])
        stds = np.array([d["std_trajectory"][::downsample, state_idx] for d in diag_entry])
        m = np.mean(means, axis=0)
        s = np.sqrt(np.mean(stds ** 2, axis=0))
        t = T_kf[:len(diag_entry[0]["mean_trajectory"])][::downsample]
    else:
        t = T_kf[:len(diag_entry["mean_trajectory"])][::downsample]
        m = diag_entry["mean_trajectory"][::downsample, state_idx]
        s = diag_entry["std_trajectory"][::downsample, state_idx]
    return t, m, m - s, m + s, m - 2 * s, m + 2 * s


def plot_metabolites_with_ensemble_bands(
    set_model_by_dataset, diagnostics_by_dataset,
    load_dataset_fn, T_model, T_kf, T_meas_by_dataset,
    state_name, axis_name, meas_num,
    dataset_list=("P1", "P2", "P3", "P4"),
    dataset_colours=None,
    downsample=10, save_path=None,
):
    """
    Measured extracellular metabolites with within-run ensemble ±1σ/±2σ bands.

    Uses the posterior ensemble spread (N=100 members from a single EnKF run)
    — the filter's own uncertainty estimate — rather than across-run variability.
    """
    n_rows = meas_num
    n_cols = len(dataset_list)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(5.0 * n_cols, 3.0 * n_rows),
        sharex=True,
    )
    if n_rows == 1:
        axes = np.expand_dims(axes, 0)
    if n_cols == 1:
        axes = np.expand_dims(axes, 1)

    for r in range(n_rows):
        for c, name in enumerate(dataset_list):
            ax = axes[r, c]
            colour = (dataset_colours or {}).get(name, "tab:blue")
            diag = diagnostics_by_dataset[name]

            # Model
            ax.plot(T_model[::downsample],
                    set_model_by_dataset[name][::downsample, r],
                    color="red", lw=1.8)

            # EnKF posterior bands (within-run ensemble)
            t, m, lo1, hi1, lo2, hi2 = _ensemble_profile_stats(
                diag, r, T_kf, downsample)
            ax.fill_between(t, lo2, hi2, color="steelblue", alpha=0.15)
            ax.fill_between(t, lo1, hi1, color="steelblue", alpha=0.30)
            ax.plot(t, m, color="steelblue", lw=2.0)

            # Measurements
            data = load_dataset_fn(name)
            T_meas = T_meas_by_dataset[name]
            set_meas = data["set_meas"].astype(float)
            set_meas_err = data["set_meas_errorbar"].astype(float)
            if r < set_meas.shape[1]:
                ax.errorbar(
                    T_meas, set_meas[:, r], yerr=set_meas_err[:, r],
                    fmt="o", color=colour, markersize=4.0, capsize=2,
                    elinewidth=1, alpha=0.9, zorder=5,
                )

            if r == 0:
                ax.set_title(name, fontsize=13, fontweight="bold")
            if c == 0:
                ax.set_ylabel(axis_name[r], fontsize=10, fontweight="bold")
            if r == n_rows - 1:
                ax.set_xlabel("Time (hours)", fontsize=11, fontweight="bold")
            _style_ax(ax)

    legend_elements = [
        Line2D([0], [0], color="red", lw=1.8, label="Mechanistic model"),
        Line2D([0], [0], color="steelblue", lw=2.0, label="EnKF posterior mean"),
        Patch(facecolor="steelblue", alpha=0.30, label=r"Posterior $\pm 1\sigma$"),
        Patch(facecolor="steelblue", alpha=0.15, label=r"Posterior $\pm 2\sigma$"),
        Line2D([0], [0], color="grey", marker="o", lw=0, markersize=5,
               label="Measurements"),
    ]
    fig.legend(
        handles=legend_elements, loc="lower center", ncol=5,
        fontsize=11, frameon=False, bbox_to_anchor=(0.5, -0.01),
    )
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    _savefig(fig, save_path)


def plot_nsd_with_ensemble_bands(
    set_model_by_dataset, diagnostics_by_dataset,
    load_dataset_fn, T_model, T_kf, T_meas_by_dataset, axis_name,
    dataset_list=("P1", "P2", "P3", "P4"),
    downsample=10, save_path=None,
):
    """NSD states with within-run ensemble ±1σ/±2σ bands."""
    n_rows = len(NSD_ORDER)
    n_cols = len(dataset_list)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(5.0 * n_cols, 3.2 * n_rows),
        sharex=True,
    )
    if n_rows == 1:
        axes = np.expand_dims(axes, 0)
    if n_cols == 1:
        axes = np.expand_dims(axes, 1)

    for r, state_key in enumerate(NSD_ORDER):
        i_global = NSD_STATES_GLOBAL[state_key]
        j_local = NSD_STATES_LOCAL[state_key]

        for c, name in enumerate(dataset_list):
            ax = axes[r, c]
            diag = diagnostics_by_dataset[name]

            ax.plot(T_model[::downsample],
                    set_model_by_dataset[name][::downsample, i_global],
                    color="red", lw=1.8)

            t, m, lo1, hi1, lo2, hi2 = _ensemble_profile_stats(
                diag,
                i_global, T_kf, downsample)
            ax.fill_between(t, lo2, hi2, color="steelblue", alpha=0.15)
            ax.fill_between(t, lo1, hi1, color="steelblue", alpha=0.30)
            ax.plot(t, m, color="steelblue", lw=2.0)

            data = load_dataset_fn(name)
            NSD_meas = data["NSD_meas"]
            NSD_meas_err = data["NSD_meas_errorbar"]
            T_meas = T_meas_by_dataset[name]
            ax.errorbar(
                T_meas, NSD_meas[:, j_local], yerr=NSD_meas_err[:, j_local],
                fmt="o", color="darkorange", markersize=4.0, capsize=2,
                elinewidth=1, alpha=0.9, zorder=5,
            )

            if r == 0:
                ax.set_title(name, fontsize=13, fontweight="bold")
            if c == 0:
                ax.set_ylabel(axis_name[i_global], fontsize=10, fontweight="bold")
            if r == n_rows - 1:
                ax.set_xlabel("Time (hours)", fontsize=11, fontweight="bold")
            _style_ax(ax)

    legend_elements = [
        Line2D([0], [0], color="red", lw=1.8, label="Mechanistic model"),
        Line2D([0], [0], color="steelblue", lw=2.0, label="EnKF posterior mean"),
        Patch(facecolor="steelblue", alpha=0.30, label=r"Posterior $\pm 1\sigma$"),
        Patch(facecolor="steelblue", alpha=0.15, label=r"Posterior $\pm 2\sigma$"),
        Line2D([0], [0], color="darkorange", marker="o", lw=0, markersize=5,
               label="Measurements"),
    ]
    fig.legend(
        handles=legend_elements, loc="lower center", ncol=5,
        fontsize=11, frameon=False, bbox_to_anchor=(0.5, -0.01),
    )
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    _savefig(fig, save_path)


def plot_asn_glu_with_ensemble_bands(
    set_model_by_dataset, diagnostics_by_dataset,
    load_dataset_fn, T_model, T_kf, T_meas_by_dataset, axis_name,
    state_indices=(8, 9), state_labels=("Asn", "Glu"),
    dataset_list=("P1", "P2", "P3", "P4"),
    downsample=10, save_path=None,
):
    """Unmeasured extracellular states (Asn, Glu) with within-run ensemble bands."""
    n_rows = len(state_indices)
    n_cols = len(dataset_list)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(5.0 * n_cols, 3.5 * n_rows),
        sharex=True,
    )
    if n_rows == 1:
        axes = np.expand_dims(axes, 0)
    if n_cols == 1:
        axes = np.expand_dims(axes, 1)

    for r, (si, sl) in enumerate(zip(state_indices, state_labels)):
        for c, name in enumerate(dataset_list):
            ax = axes[r, c]
            diag = diagnostics_by_dataset[name]

            ax.plot(T_model[::downsample],
                    set_model_by_dataset[name][::downsample, si],
                    color="red", lw=1.8)

            t, m, lo1, hi1, lo2, hi2 = _ensemble_profile_stats(
                diag,
                si, T_kf, downsample)
            ax.fill_between(t, lo2, hi2, color="steelblue", alpha=0.15)
            ax.fill_between(t, lo1, hi1, color="steelblue", alpha=0.30)
            ax.plot(t, m, color="steelblue", lw=2.0)

            data = load_dataset_fn(name)
            T_meas = T_meas_by_dataset[name]
            set_meas = data["set_meas"].astype(float)
            set_meas_err = data["set_meas_errorbar"].astype(float)
            if sl == "Asn":
                asn_col = set_meas.shape[1] - 1
                ax.errorbar(
                    T_meas, set_meas[:, asn_col], yerr=set_meas_err[:, asn_col],
                    fmt="o", color="darkorange", markersize=4.0, capsize=2,
                    elinewidth=1, alpha=0.9, zorder=5,
                )

            if r == 0:
                ax.set_title(name, fontsize=13, fontweight="bold")
            if c == 0:
                ax.set_ylabel(axis_name[si], fontsize=10, fontweight="bold")
            if r == n_rows - 1:
                ax.set_xlabel("Time (hours)", fontsize=11, fontweight="bold")
            _style_ax(ax)

    legend_elements = [
        Line2D([0], [0], color="red", lw=1.8, label="Mechanistic model"),
        Line2D([0], [0], color="steelblue", lw=2.0, label="EnKF posterior mean"),
        Patch(facecolor="steelblue", alpha=0.30, label=r"Posterior $\pm 1\sigma$"),
        Patch(facecolor="steelblue", alpha=0.15, label=r"Posterior $\pm 2\sigma$"),
        Line2D([0], [0], color="darkorange", marker="o", lw=0, markersize=5,
               label="Measurements (validation)"),
    ]
    fig.legend(
        handles=legend_elements, loc="lower center", ncol=5,
        fontsize=11, frameon=False, bbox_to_anchor=(0.5, -0.01),
    )
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    _savefig(fig, save_path)


# ─── Gramian heatmap plots ──────────────────────────────────────────────────

def plot_gramian_heatmap(
    Wo_by_dataset, state_name,
    dataset_list=("P1", "P2", "P3", "P4"),
    dataset_colours=None, exclude_last_n_states=7,
    save_path=None,
):
    """Plot observability Gramian diagonal for extracellular metabolites."""
    n_states = len(state_name) - exclude_last_n_states
    labels = state_name[:n_states]
    n_ds = len(dataset_list)

    fig, axes = plt.subplots(1, n_ds, figsize=(5 * n_ds, 4), sharey=True)
    if n_ds == 1:
        axes = [axes]

    for idx, name in enumerate(dataset_list):
        ax = axes[idx]
        Wo = Wo_by_dataset[name]
        diag = np.diag(Wo)[:n_states]
        colour = dataset_colours.get(name, "steelblue") if dataset_colours else "steelblue"

        ax.barh(range(n_states), diag, color=colour, alpha=0.8)
        ax.set_yticks(range(n_states))
        ax.set_yticklabels(labels)
        ax.set_title(name, fontweight="bold")
        ax.set_xlabel("Gramian diagonal", fontweight="bold")
        ax.invert_yaxis()

    plt.tight_layout()
    _savefig(fig, save_path)


def plot_gramian_nsd(
    Wo_by_dataset, state_name,
    dataset_list=("P1", "P2", "P3", "P4"),
    dataset_colours=None, nsd_last_n_states=7,
    save_path=None,
):
    """Plot observability Gramian diagonal for NSD states only."""
    n_total = len(state_name)
    start = n_total - nsd_last_n_states
    labels = state_name[start:]
    n_nsd = len(labels)
    n_ds = len(dataset_list)

    fig, axes = plt.subplots(1, n_ds, figsize=(5 * n_ds, 4), sharey=True)
    if n_ds == 1:
        axes = [axes]

    for idx, name in enumerate(dataset_list):
        ax = axes[idx]
        Wo = Wo_by_dataset[name]
        diag = np.diag(Wo)[start:]
        colour = dataset_colours.get(name, "steelblue") if dataset_colours else "steelblue"

        ax.barh(range(n_nsd), diag, color=colour, alpha=0.8)
        ax.set_yticks(range(n_nsd))
        ax.set_yticklabels(labels)
        ax.set_title(name, fontweight="bold")
        ax.set_xlabel("Gramian diagonal", fontweight="bold")
        ax.invert_yaxis()

    plt.tight_layout()
    _savefig(fig, save_path)


def plot_selected_nsds_per_dataset(
    Wo_by_dataset, state_name,
    dataset_list=("P1", "P2", "P3", "P4"),
    dataset_colours=None,
    selected_nsds=("UDPGal", "UDPGlc", "UDPGlcNAc"),
    save_path=None,
):
    """Plot Gramian diagonal for selected NSD states per dataset."""
    nsd_indices = {s: state_name.index(s) for s in selected_nsds}
    n_ds = len(dataset_list)

    fig, axes = plt.subplots(1, n_ds, figsize=(5 * n_ds, 3), sharey=True)
    if n_ds == 1:
        axes = [axes]

    for idx, name in enumerate(dataset_list):
        ax = axes[idx]
        Wo = Wo_by_dataset[name]
        diag = np.diag(Wo)
        vals = [diag[nsd_indices[s]] for s in selected_nsds]
        colour = dataset_colours.get(name, "steelblue") if dataset_colours else "steelblue"

        ax.barh(range(len(selected_nsds)), vals, color=colour, alpha=0.8)
        ax.set_yticks(range(len(selected_nsds)))
        ax.set_yticklabels(list(selected_nsds))
        ax.set_title(name, fontweight="bold")
        ax.set_xlabel("Gramian diagonal", fontweight="bold")
        ax.invert_yaxis()

    plt.tight_layout()
    _savefig(fig, save_path)
