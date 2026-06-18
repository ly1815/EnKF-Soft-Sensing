"""
enkf.py
=======
Ensemble Kalman Filter for state estimation with soft sensing.

Classes
-------
EnsembleKalmanFilter  — state estimation via EnKF with partial observations

Runners
-------
run_enkf_multi_dataset  — run EnKF across all datasets with multiple runs
"""

import warnings
import numpy as np
from scipy.linalg import solve, LinAlgWarning
from tqdm import tqdm

warnings.filterwarnings("ignore", category=LinAlgWarning)

from nsd_enkf.model import model_step


class EnsembleKalmanFilter:
    """
    State estimation via the Ensemble Kalman Filter.

    The filter propagates an ensemble of state vectors through the mechanistic
    model and updates them when measurements are available. Only the measured
    states (extracellular metabolites) are used in the update step; unmeasured
    states (NSDs, Asn) are estimated via the model dynamics.

    Supports two noise modes:
      - additive: noise_i ~ N(0, Q_ii)  (fixed variance, default)
      - multiplicative: noise_i ~ N(0, (cv_i * x_i)^2)  (state-proportional)
    Per-state noise mode is controlled by process_noise_cv (dict).
    """

    def __init__(self, num_x, num_z):
        self.x = None       # state mean
        self.z = None        # observations
        self.Q = None        # process noise covariance (additive states)
        self.R = None        # measurement noise covariance
        self.fx = None       # model step function
        self.H = None        # observation matrix
        self.dt = None       # time step
        self.P = None        # state covariance

        self.X = None        # ensemble (N, num_x)
        self.num_x = num_x   # number of states
        self.num_X = None     # ensemble size
        self.num_z = num_z    # number of measured states

        # Multiplicative noise: dict {state_index: cv} for states using
        # state-proportional noise. States not in this dict use additive Q.
        self.process_noise_cv = {}

        # Localization: set of state indices excluded from the Kalman update.
        # These states evolve only through the model (no measurement correction).
        # Used for structurally unobservable states where cross-covariance
        # corrections introduce spurious jumps.
        self.no_update_indices = set()

    def create_ensemble(self, N, Cov):
        """Draw initial ensemble from multivariate normal, capped at 3-sigma."""
        self.num_X = N
        self.num_x = len(self.x)

        self.X = np.random.multivariate_normal(self.x, Cov, N)

        # 3-sigma capping per state
        for i in range(self.num_x):
            sd = np.sqrt(Cov[i, i])
            self.X[:, i] = np.clip(
                self.X[:, i], self.x[i] - 3.0 * sd, self.x[i] + 3.0 * sd
            )

        self.X = np.clip(self.X, a_min=1e-12, a_max=None)
        self.x = np.mean(self.X, axis=0)

    def predict(self, controls):
        """Forecast step: propagate each ensemble member through the model."""
        X_new = []
        for x in self.X:
            X_new.append(self.fx(x, 0.0, controls, self.dt))
        X_new = np.array(X_new)

        # Build noise per ensemble member
        noise = np.zeros_like(X_new)

        # Additive noise for states NOT in process_noise_cv
        additive_indices = [i for i in range(self.num_x)
                           if i not in self.process_noise_cv]
        if additive_indices:
            Q_add = np.diag(np.diag(self.Q)[additive_indices])
            n_add = np.random.multivariate_normal(
                np.zeros(len(additive_indices)), Q_add, size=self.num_X
            )
            for j_out, i_state in enumerate(additive_indices):
                sd = np.sqrt(self.Q[i_state, i_state])
                noise[:, i_state] = np.clip(n_add[:, j_out], -3.0 * sd, 3.0 * sd)

        # Multiplicative noise for states in process_noise_cv
        for i_state, cv in self.process_noise_cv.items():
            state_vals = np.maximum(X_new[:, i_state], 1e-12)
            sd = cv * state_vals
            n_mult = np.random.randn(self.num_X) * sd
            noise[:, i_state] = np.clip(n_mult, -3.0 * sd, 3.0 * sd)

        self.X = X_new + noise
        self.X = np.clip(self.X, a_min=1e-12, a_max=None)
        self.x = np.mean(self.X, axis=0)

    def update(self, z_ensemble):
        """
        Analysis step: assimilate an ensemble of observations.

        Parameters
        ----------
        z_ensemble : np.ndarray, shape (N, num_z)
            Noisy observation ensemble for this time step.
        """
        N = self.num_X

        # State anomalies
        x_mean = np.mean(self.X, axis=0)
        E_x = self.X - x_mean

        # Predicted measurements and anomalies
        Z = np.array([self.H @ x for x in self.X])
        z_mean = np.mean(Z, axis=0)
        E_z = Z - z_mean

        # Covariances
        P_xz = (E_x.T @ E_z) / (N - 1)
        P_zz = (E_z.T @ E_z) / (N - 1) + self.R

        # Kalman gain: K = P_xz @ P_zz^{-1}, solved as P_zz^T @ K^T = P_xz^T
        K = solve(P_zz.T, P_xz.T, assume_a='pos').T

        # Localization: zero out Kalman gain for unobservable states
        if self.no_update_indices:
            for i in self.no_update_indices:
                K[i, :] = 0.0

        # Update ensemble
        self.X += (K @ (z_ensemble - Z).T).T
        self.X = np.clip(self.X, a_min=1e-12, a_max=None)
        self.x = np.mean(self.X, axis=0)


# ─── Runner: multi-dataset EnKF ─────────────────────────────────────────────

def run_enkf_multi_dataset(
    datasets_cfg, load_dataset_fn, build_schedule_fn,
    state_init_by_dataset, volume_results,
    set_meas_ens_by_dataset, T_meas_by_dataset,
    state_num, meas_num, ensemble_size, n_runs,
    Q, R, H, dt_kf, N_kf,
    P0=None,
    process_noise_cv=None,
    no_update_indices=None,
    decimal_places=2,
    save_fn=None,
):
    """
    Run the EnKF across all datasets with multiple independent runs.

    Parameters
    ----------
    process_noise_cv : dict or None
        {state_index: cv} for states using multiplicative noise.
    no_update_indices : set or None
        State indices excluded from Kalman update (structurally unobservable).

    Returns
    -------
    enkf_results_by_dataset : dict  {name: mean trajectory (N_kf+1, state_num)}
    """
    enkf_results_by_dataset = {}

    for name in tqdm(datasets_cfg.keys(), desc="Running EnKF over datasets"):
        print(f"\nRunning EnKF for {name}")

        data = load_dataset_fn(name)
        set_meas_full = data["set_meas"]
        set_meas = set_meas_full[:, :meas_num].copy()
        set_meas_ens = set_meas_ens_by_dataset[name]
        T_meas = T_meas_by_dataset[name]

        Fin, Fout, Gal_feed, Urd_feed = build_schedule_fn(name)
        V_traj = volume_results[name][1:]

        state_init = state_init_by_dataset[name].copy()

        time_steps_A = [round(i * dt_kf, decimal_places) for i in range(N_kf)]
        time_steps_B = [round(t, decimal_places) for t in T_meas.tolist()]
        meas_time_to_index = {t: i for i, t in enumerate(time_steps_B)}

        # Running sum for mean across runs
        mean_sum = None

        for run_i in range(n_runs):
            enkf = EnsembleKalmanFilter(state_num, meas_num)
            enkf.x = state_init.copy()
            enkf.z = set_meas.copy()
            enkf.Q = Q.copy()
            enkf.R = R.copy()
            enkf.H = H.copy()
            enkf.fx = model_step
            enkf.dt = dt_kf
            if process_noise_cv is not None:
                enkf.process_noise_cv = dict(process_noise_cv)
            if no_update_indices is not None:
                enkf.no_update_indices = set(no_update_indices)

            init_cov = P0 if P0 is not None else Q
            enkf.create_ensemble(ensemble_size, init_cov)

            set_EnKF = [state_init.copy()]
            mean_traj = [enkf.x.copy()]
            std_traj = [np.std(enkf.X, axis=0)]

            # Full ensemble snapshots only on first run (large data)
            is_first = (run_i == 0)
            if is_first:
                ensemble_at_updates = [{
                    "time": 0.0, "label": "t=0 (initial)",
                    "ensemble": enkf.X.copy(), "mean": enkf.x.copy(),
                }]

            for idx_A, step_A in enumerate(
                tqdm(time_steps_A, desc=f"{name} run {run_i+1}/{n_runs}", leave=False)
            ):
                controls_k = {
                    "Fin": Fin[idx_A],
                    "Fout": Fout[idx_A],
                    "V": V_traj[idx_A],
                    "Gal_feed": Gal_feed[idx_A],
                    "Urd_feed": Urd_feed[idx_A],
                }
                enkf.predict(controls_k)

                if step_A in meas_time_to_index:
                    idx_B = meas_time_to_index[step_A]
                    z_ens = set_meas_ens[idx_B]

                    if is_first:
                        ensemble_at_updates.append({
                            "time": step_A, "label": f"t={step_A}h forecast",
                            "ensemble": enkf.X.copy(), "mean": enkf.x.copy(),
                        })

                    enkf.update(z_ens)

                    if is_first:
                        ensemble_at_updates.append({
                            "time": step_A, "label": f"t={step_A}h analysis",
                            "ensemble": enkf.X.copy(), "mean": enkf.x.copy(),
                        })

                set_EnKF.append(enkf.x.copy())
                mean_traj.append(enkf.x.copy())
                std_traj.append(np.std(enkf.X, axis=0))

            run_traj = np.array(set_EnKF)

            # Accumulate running sum (memory: one array, not n_runs arrays)
            if mean_sum is None:
                mean_sum = run_traj.copy()
            else:
                mean_sum += run_traj

            # Build per-run diagnostics and save immediately
            run_diag = {
                "mean_trajectory": np.array(mean_traj),
                "std_trajectory": np.array(std_traj),
            }
            if is_first:
                run_diag["ensemble_at_updates"] = ensemble_at_updates

            if save_fn is not None:
                save_fn(run_diag, f"diagnostics_{name}_run{run_i}.pkl")
                save_fn(run_traj, f"enkf_traj_{name}_run{run_i}.pkl")

            # Free memory immediately
            del run_traj, run_diag, mean_traj, std_traj, set_EnKF, enkf
            if is_first:
                del ensemble_at_updates

        set_EnKF_mean = mean_sum / n_runs
        enkf_results_by_dataset[name] = set_EnKF_mean

        if save_fn is not None:
            save_fn(set_EnKF_mean, f"enkf_results_{name}.pkl")

    return enkf_results_by_dataset


# ─── Diagnostic runner: record full ensemble at measurement updates ──────────

def run_enkf_single_with_ensemble_diagnostics(
    dataset_name, load_dataset_fn, build_schedule_fn,
    state_init, volume_results,
    set_meas_ens, T_meas,
    state_num, meas_num, ensemble_size,
    Q, R, H, dt_kf, N_kf,
    P0=None,
    process_noise_cv=None,
    no_update_indices=None,
    decimal_places=2,
):
    """
    Run a single EnKF pass and record the full ensemble (N, state_num)
    at every measurement update time, plus the ensemble std at every timestep.

    Returns
    -------
    ensemble_at_updates : list of dict
        Each entry: {"time": float, "ensemble": (N, state_num), "mean": (state_num,)}
    std_trajectory : np.ndarray, shape (N_kf+1, state_num)
        Ensemble standard deviation at every timestep.
    mean_trajectory : np.ndarray, shape (N_kf+1, state_num)
        Ensemble mean at every timestep.
    """
    Fin, Fout, Gal_feed, Urd_feed = build_schedule_fn(dataset_name)
    V_traj = volume_results[dataset_name][1:]

    time_steps_A = [round(i * dt_kf, decimal_places) for i in range(N_kf)]
    time_steps_B = [round(t, decimal_places) for t in T_meas.tolist()]
    meas_time_to_index = {t: i for i, t in enumerate(time_steps_B)}

    enkf = EnsembleKalmanFilter(state_num, meas_num)
    enkf.x = state_init.copy()
    enkf.Q = Q.copy()
    enkf.R = R.copy()
    enkf.H = H.copy()
    enkf.fx = model_step
    enkf.dt = dt_kf
    if process_noise_cv is not None:
        enkf.process_noise_cv = dict(process_noise_cv)
    if no_update_indices is not None:
        enkf.no_update_indices = set(no_update_indices)

    init_cov = P0 if P0 is not None else Q
    enkf.create_ensemble(ensemble_size, init_cov)

    ensemble_at_updates = [{
        "time": 0.0,
        "label": "t=0 (initial)",
        "ensemble": enkf.X.copy(),
        "mean": enkf.x.copy(),
    }]

    mean_traj = [enkf.x.copy()]
    std_traj = [np.std(enkf.X, axis=0)]

    for idx_A, step_A in enumerate(
        tqdm(time_steps_A, desc=f"EnKF diagnostics {dataset_name}", leave=True)
    ):
        controls_k = {
            "Fin": Fin[idx_A],
            "Fout": Fout[idx_A],
            "V": V_traj[idx_A],
            "Gal_feed": Gal_feed[idx_A],
            "Urd_feed": Urd_feed[idx_A],
        }
        enkf.predict(controls_k)

        if step_A in meas_time_to_index:
            idx_B = meas_time_to_index[step_A]
            z_ens = set_meas_ens[idx_B]

            # Record pre-update (forecast) ensemble
            ensemble_at_updates.append({
                "time": step_A,
                "label": f"t={step_A}h forecast",
                "ensemble": enkf.X.copy(),
                "mean": enkf.x.copy(),
            })

            enkf.update(z_ens)

            # Record post-update (analysis) ensemble
            ensemble_at_updates.append({
                "time": step_A,
                "label": f"t={step_A}h analysis",
                "ensemble": enkf.X.copy(),
                "mean": enkf.x.copy(),
            })

        mean_traj.append(enkf.x.copy())
        std_traj.append(np.std(enkf.X, axis=0))

    return (
        ensemble_at_updates,
        np.array(std_traj),
        np.array(mean_traj),
    )
