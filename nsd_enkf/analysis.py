"""
analysis.py
===========
Post-processing: RMSE computation, measurement ensemble generation,
and observability Gramian computation.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

from nsd_enkf.model import model_step


def generate_measurement_ensembles(datasets_cfg, load_dataset_fn, meas_num,
                                   ensemble_size, var_meas):
    """
    Generate noisy measurement ensembles for all datasets.

    Returns
    -------
    dict  {name: np.ndarray of shape (N_meas_time, ensemble_size, meas_num)}
    """
    set_meas_ens_by_dataset = {}
    meas_std = np.sqrt(var_meas[:meas_num])

    for name in tqdm(datasets_cfg.keys(), desc="Generating measurement ensembles"):
        data = load_dataset_fn(name)
        set_meas_full = data["set_meas"]
        N_meas_time = set_meas_full.shape[0]

        set_meas_kf = set_meas_full[:, :meas_num]
        set_meas_ens = np.zeros((N_meas_time, ensemble_size, meas_num))

        for i in range(N_meas_time):
            noise_samples = np.random.multivariate_normal(
                mean=np.zeros(meas_num),
                cov=np.diag(var_meas[:meas_num]),
                size=ensemble_size,
            )
            # 3-sigma capping
            for j in range(meas_num):
                noise_samples[:, j] = np.clip(
                    noise_samples[:, j], -3.0 * meas_std[j], 3.0 * meas_std[j]
                )

            set_meas_ens[i] = np.clip(
                set_meas_kf[i] + noise_samples, a_min=1e-12, a_max=None
            )

        set_meas_ens_by_dataset[name] = set_meas_ens

    return set_meas_ens_by_dataset


def compute_rmse_table(datasets_cfg, load_dataset_fn,
                       set_model_by_dataset, enkf_results_by_dataset,
                       T_model, T_kf, T_meas_by_dataset,
                       axis_name, state_num):
    """
    Compute RMSE for model and EnKF predictions vs measurements.

    Returns
    -------
    pd.DataFrame with columns: Dataset, State, RMSE_Model, RMSE_EnKF
    """
    rmse_records = []

    for name in datasets_cfg.keys():
        data = load_dataset_fn(name)
        set_meas = data["set_meas"].astype(float)
        n_met = set_meas.shape[1]

        NSD_meas = (
            pd.DataFrame(data["NSD_meas"])
            .apply(pd.to_numeric, errors="coerce")
            .to_numpy()
        )
        n_nsd = NSD_meas.shape[1]

        T_meas = T_meas_by_dataset[name]
        set_model = set_model_by_dataset[name]
        set_EnKF = enkf_results_by_dataset[name]

        # Metabolites
        for i in range(n_met):
            measured = set_meas[:, i].astype(float)
            model_pred = np.interp(T_meas, T_model, set_model[:, i])
            enkf_pred = np.interp(T_meas, T_kf, set_EnKF[:, i])

            mask = ~np.isnan(measured)
            if np.sum(mask) == 0:
                continue

            rmse_records.append({
                "Dataset": name,
                "State": axis_name[i],
                "RMSE_Model": np.sqrt(mean_squared_error(measured[mask], model_pred[mask])),
                "RMSE_EnKF": np.sqrt(mean_squared_error(measured[mask], enkf_pred[mask])),
            })

        # NSDs
        start_nsd = state_num - n_nsd
        for j in range(n_nsd):
            i = start_nsd + j
            measured = NSD_meas[:, j].astype(float)
            model_pred = np.interp(T_meas, T_model, set_model[:, i])
            enkf_pred = np.interp(T_meas, T_kf, set_EnKF[:, i])

            mask = ~np.isnan(measured)
            if np.sum(mask) == 0:
                continue

            rmse_records.append({
                "Dataset": name,
                "State": axis_name[i],
                "RMSE_Model": np.sqrt(mean_squared_error(measured[mask], model_pred[mask])),
                "RMSE_EnKF": np.sqrt(mean_squared_error(measured[mask], enkf_pred[mask])),
            })

    return pd.DataFrame(rmse_records).round(3)


def compute_dimensionless_gramian(
    datasets_cfg, state_name, T_model, dt_model,
    load_dataset_fn, build_schedule_fn,
    state_init_by_dataset, volume_results,
    measured_state_names, epsilon=0.01,
):
    """
    Compute the dimensionless observability Gramian for all datasets.

    Returns
    -------
    Wo_by_dataset : dict  {name: np.ndarray of shape (state_num, state_num)}
    measured_indices : list of int
    """
    state_num = len(state_name)
    measured_indices = [state_name.index(s) for s in measured_state_names]
    n_meas = len(measured_indices)
    n_steps = len(T_model) - 1

    Wo_by_dataset = {}

    for name in tqdm(datasets_cfg.keys(), desc="Computing Gramians"):
        x0 = state_init_by_dataset[name].copy()
        Fin, Fout, Gal_feed, Urd_feed = build_schedule_fn(name)
        V_traj = volume_results[name][1:]
        step_len_arr = np.full(n_steps, dt_model)

        # Nominal trajectory
        traj_nom = [x0.copy()]
        state = x0.copy()
        for k in range(n_steps):
            controls_k = {
                "Fin": Fin[k], "Fout": Fout[k], "V": V_traj[k],
                "Gal_feed": Gal_feed[k], "Urd_feed": Urd_feed[k],
            }
            state = model_step(state, 0.0, controls_k, step_len_arr[k])
            traj_nom.append(state.copy())
        traj_nom = np.array(traj_nom)

        # Compute Gramian via finite differences
        Wo = np.zeros((state_num, state_num))

        for i in range(state_num):
            x0_pert = x0.copy()
            delta = epsilon * max(abs(x0[i]), 1e-12)
            x0_pert[i] += delta

            traj_pert = [x0_pert.copy()]
            state = x0_pert.copy()
            for k in range(n_steps):
                controls_k = {
                    "Fin": Fin[k], "Fout": Fout[k], "V": V_traj[k],
                    "Gal_feed": Gal_feed[k], "Urd_feed": Urd_feed[k],
                }
                state = model_step(state, 0.0, controls_k, step_len_arr[k])
                traj_pert.append(state.copy())
            traj_pert = np.array(traj_pert)

            # Sensitivity of measured outputs to state i
            dy = (traj_pert[:, measured_indices] - traj_nom[:, measured_indices]) / delta

            # Normalize by nominal output scale
            y_nom = traj_nom[:, measured_indices]
            scale = np.maximum(np.abs(y_nom).max(axis=0), 1e-12)
            dy_norm = dy / scale

            # Accumulate Gramian
            for j in range(n_steps + 1):
                Wo[i, :] += dy_norm[j] @ dy_norm[j].reshape(n_meas, 1).T @ np.eye(state_num)[i]

        Wo_by_dataset[name] = Wo * dt_model

    return Wo_by_dataset, measured_indices
