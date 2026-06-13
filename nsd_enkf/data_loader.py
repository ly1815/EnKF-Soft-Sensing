"""
data_loader.py
==============
Load CHO cell culture datasets from Excel files and construct feeding schedules.
"""

import numpy as np
import pandas as pd
from pathlib import Path

from nsd_enkf.config import (
    DATASETS_ALL, STATE_NAMES, DT, T_END,
    FIN_PULSES, FOUT_PULSES, BOLUS_DOSES,
)


def select_datasets(*names, datasets_all=None):
    """
    Select a subset of datasets by name.

    Parameters
    ----------
    *names : str
        Dataset identifiers (e.g. 'P1', 'P3').
        If no names provided, all datasets are returned.
    """
    if datasets_all is None:
        datasets_all = DATASETS_ALL

    if len(names) == 0:
        return dict(datasets_all)

    names = [str(n).upper() for n in names]
    missing = [n for n in names if n not in datasets_all]
    if missing:
        raise KeyError(
            f"Unknown dataset(s): {missing}. "
            f"Available: {list(datasets_all.keys())}"
        )
    return {n: datasets_all[n] for n in names}


def load_dataset(name, datasets_cfg=None):
    """
    Load measurement data for a single dataset.

    Returns
    -------
    dict with keys: set_meas, NSD_meas, set_meas_errorbar,
                    NSD_meas_errorbar, met_df, nsd_df
    """
    if datasets_cfg is None:
        datasets_cfg = DATASETS_ALL

    name = str(name).upper()
    cfg = datasets_cfg[name]
    file_path = cfg["file"]

    if not file_path.exists():
        raise FileNotFoundError(
            f"Data file not found: {file_path}\n"
            "Ensure the Excel files are placed in data/raw/."
        )

    met_df = pd.read_excel(file_path, sheet_name=cfg["sheets"]["met"])
    nsd_df = pd.read_excel(file_path, sheet_name=cfg["sheets"]["nsd"])

    # Extracellular metabolites: value columns at 3::2, error bars at 4::2
    set_meas = met_df.iloc[2:, 3::2].to_numpy()
    set_meas_errorbar = met_df.iloc[2:, 4::2].to_numpy()

    # Intracellular NSDs: value columns at 1::2, error bars at 2::2
    NSD_meas = nsd_df.iloc[2:, 1::2].to_numpy()
    NSD_meas_errorbar = nsd_df.iloc[2:, 2::2].to_numpy()

    return {
        "set_meas": set_meas,
        "NSD_meas": NSD_meas,
        "set_meas_errorbar": set_meas_errorbar,
        "NSD_meas_errorbar": NSD_meas_errorbar,
        "met_df": met_df,
        "nsd_df": nsd_df,
    }


def get_initial_condition(met_df, nsd_df, state_names=None, row_idx=2):
    """
    Construct initial conditions from experimental data.

    Returns
    -------
    initial_condition : dict
    state_init : np.ndarray
    """
    if state_names is None:
        state_names = STATE_NAMES

    initial_condition = {
        "Xv": met_df["Xv"][row_idx],
        "mAb": met_df["mAb"][row_idx],
        "Gal": met_df["Gal"][row_idx],
        "Urd": met_df["Urd"][row_idx],
        "Glc": met_df["Glc"][row_idx],
        "Amm": met_df["Amm"][row_idx],
        "Gln": met_df["Gln"][row_idx],
        "Lac": met_df["Lac"][row_idx],
        "Asn": met_df["Asn"][row_idx],
        "Glu": 2.125,
        "UDPGal": nsd_df["UDPGal"][row_idx],
        "UDPGalNAc": nsd_df["UDPGalNAc"][row_idx],
        "UDPGlc": nsd_df["UDPGlc"][row_idx],
        "UDPGlcNAc": nsd_df["UDPGlcNAc"][row_idx],
        "GDPMan": nsd_df["GDPMan"][row_idx],
        "GDPFuc": nsd_df["GDPFuc"][row_idx],
        "CMPNeu5Ac": nsd_df["CMPNeu5Ac"][row_idx],
    }

    state_init = np.array(
        [initial_condition[k] for k in state_names],
        dtype=float,
    )
    return initial_condition, state_init


def hr_to_idx(t_hr: float, dt: float = None) -> int:
    """Convert culture time in hours to array index on the DT grid."""
    if dt is None:
        dt = DT
    return int(round(t_hr / dt))


def build_schedule(dataset_name: str, n_steps: int = None, dt: float = None):
    """
    Construct Fin, Fout, Gal_feed, Urd_feed arrays for a given dataset.

    Returns
    -------
    Fin, Fout, Gal_feed, Urd_feed : np.ndarray, each of length n_steps
    """
    if dt is None:
        dt = DT
    if n_steps is None:
        n_steps = int(T_END / dt)

    Fin = np.zeros(n_steps)
    Fout = np.zeros(n_steps)
    Gal_feed = np.zeros(n_steps)
    Urd_feed = np.zeros(n_steps)

    for t in FIN_PULSES:
        Fin[hr_to_idx(t, dt)] = 1.0

    for t, val in FOUT_PULSES.items():
        Fout[hr_to_idx(t, dt)] = val

    doses = BOLUS_DOSES[dataset_name]
    for t, (gal_val, urd_val) in doses.items():
        idx = hr_to_idx(t, dt)
        Gal_feed[idx] = gal_val
        Urd_feed[idx] = urd_val

    return Fin, Fout, Gal_feed, Urd_feed
