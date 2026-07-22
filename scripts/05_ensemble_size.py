"""
05_ensemble_size.py  —  Stage 5: ensemble-size sensitivity / verification
=========================================================================
Ensemble-size sensitivity + calibration diagnostics on P4, using the CURRENT
production filter configuration (self-contained — no dependency on prior runs).

Directly targets Reviewer 2.1 (is N=100 justified? sensitivity, instability) and
Reviewer 3.4 (does the ensemble keep meaningful spread / stay calibrated, not just
accurate in the mean?).

For each ensemble size N, TARGET_GOOD independent EnKF passes (distinct seeds, seed =
--seed-offset + i) are run on P4 with the exact production settings pulled from config.
Divergent replicates are rejected (pool-relative peak-sigma outlier rule, per size, C=3x
the across-run median — identical to the tuning/validation sweeps) and resampled to
TARGET_GOOD clean runs (--no-reject to disable). The per-size divergence count is itself a
reported result: small N is where blow-ups happen, which is the core R2.1 stability point.
Production settings:
  - measured states  -> multiplicative CV noise (PROCESS_NOISE_CV)
  - unmeasured states -> additive two-stage-alpha noise (PROCESS_NOISE_VAR)
  - IQR clipping on CLIP_STATES, localization on NO_UPDATE_STATES
  - P0: measured = measurement variance, unmeasured = process-noise variance

Metrics per size (mean +/- std across runs):
  measured : normalised RMSE, NIS = mean(d^2/S) [ideal 1], 2-sigma coverage %
  NSD (7)  : normalised RMSE, 2-sigma coverage %, spread-skill = std/RMSE [ideal 1]
  Asn      : normalised RMSE
  cost     : wall-clock seconds per pass

Crash-safe: results are saved after every pass; --resume continues from where it left off
(per-size pkl is a {seed: run} cache holding every pass drawn, including rejected ones so
they are never recomputed), so a kill costs <=1 pass. Used/rejected seeds per size are
written to seed_selection.json.

Output (default --out results_multirun_ensemble_size/):
    ensemble_N<N>.pkl                {seed: run} cache — EVERY pass drawn (incl. rejected),
                                     each run carries its downsampled mean+std trajectory
    all_trajectories.pkl             combined archive: mean + spread of every clean run of
                                     every size + filter-config provenance (P4 tuning set)
    ensemble_sensitivity_summary.pkl per-size mean/std metrics + divergence counts
    seed_selection.json              used / rejected seeds per size
    ensemble_size_sensitivity.png    6-panel sensitivity + calibration figure

Usage (macOS venv):
    # Default sweep — N in {25,50,100,150,200}, 10 clean seeds each, divergence rejection on:
    caffeinate -i ./.venv/bin/python scripts/05_ensemble_size.py --n-runs 10 --resume

    # Resume a killed/interrupted run (each pass is saved as it finishes; kill costs <=1 pass):
    #   just re-run the same command — --resume skips passes already on disk.

    # Full-resolution trajectories instead of x20 downsampled: --traj-down 1

Runtime scales ~linearly with N and with n_runs. Uses the CURRENT config.py — the adopted
automated CVs (cap CV_MAX=0.006) plus the two-stage additive alpha (0.01 for the 7 NSDs,
0.002 for Asn/Glu). Pulled live from config, so this always reflects the tuned filter.

Trajectories: each run also stores its mean/std trajectory (downsampled by --traj-down,
default x20) plus the raw innovations and S at the measurement-update times, so any
trajectory-level statistic can be recomputed later without re-running. Pass
--traj-down 0 for metrics only.
"""

import argparse
import json
import pickle
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy.linalg import LinAlgWarning
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=LinAlgWarning)

import nsd_enkf.config as cfg
from nsd_enkf.data_loader import (
    select_datasets, load_dataset, get_initial_condition, build_schedule,
)
from nsd_enkf.model import compute_volume_results, model_step
from nsd_enkf.enkf import EnsembleKalmanFilter

# ── CLI ──────────────────────────────────────────────────────────────────────
p = argparse.ArgumentParser(description="Ensemble-size sensitivity + calibration on P4")
p.add_argument("--run", default="ensemble_sens", help="label used in console output only")
p.add_argument("--out", default="results_multirun_ensemble_size",
               help="output folder (under project root unless absolute)")
p.add_argument("--dataset", default="P4",
               help="tuning set: config.py holds the P4-fold calibration (CVs + alphas)")
p.add_argument("--sizes", default="25,50,100,150,200")
p.add_argument("--n-runs", default=10, type=int)
p.add_argument("--seed-offset", default=42, type=int)
p.add_argument("--resume", action="store_true", help="skip passes already on disk (per-seed cache)")
p.add_argument("--traj-down", default=20, type=int,
               help="also save per-run mean/std trajectories downsampled by this factor "
                    "(0 = metrics only, no trajectories)")
p.add_argument("--no-reject", dest="auto_reject", action="store_false", default=True,
               help="disable divergent-replicate rejection (default: on, matching the tuning/"
                    "validation sweeps)")
p.add_argument("--reject-mult", default=3.0, type=float,
               help="reject a pass if any unmeasured state's peak sigma exceeds this multiple of "
                    "the across-run median peak, per ensemble size (default 3.0)")
p.add_argument("--target-good", default=None, type=int,
               help="clean replicates required per size (default --n-runs)")
p.add_argument("--max-seeds", default=40, type=int,
               help="cap on seeds drawn per size while resampling to --target-good clean runs")
args = p.parse_args()

DS = args.dataset
SIZES = [int(x) for x in args.sizes.split(",")]
N_RUNS = args.n_runs
TARGET_GOOD = args.target_good if args.target_good is not None else N_RUNS

OUT_DIR = Path(args.out) if Path(args.out).is_absolute() else cfg.PROJECT_ROOT / args.out
OUT_DIR.mkdir(parents=True, exist_ok=True)

def save_pkl(obj, name):
    tmp = OUT_DIR / (name + ".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(obj, f)
    tmp.replace(OUT_DIR / name)

print("=" * 64)
print(f"Ensemble-size sensitivity  [run={args.run}, dataset={DS}]")
print(f"  sizes = {SIZES} | clean runs/size = {TARGET_GOOD} | seeds from {args.seed_offset}")
print(f"  divergence rejection: " + (f"ON (peak sigma > {args.reject_mult}x per-size median, "
                                     f"resample <= {args.max_seeds} seeds)"
                                     if args.auto_reject else "OFF (--no-reject)"))
print(f"  trajectories: " + (f"saved, downsampled x{args.traj_down}"
                             if args.traj_down > 0 else "not saved (metrics only)"))
print("=" * 64)

# ── Fixed config (current production filter) ─────────────────────────────────
time_grid = np.arange(cfg.DT, cfg.T_END + cfg.DT, cfg.DT)
step_len = np.full(len(time_grid), cfg.DT)
N_kf = int(cfg.T_END / cfg.DT)
T_kf = np.linspace(0, cfg.T_END, N_kf + 1)
dt_kf = cfg.DT

var_model = np.array(list(cfg.PROCESS_NOISE_VAR.values()))   # two-stage alpha already baked in
var_meas = np.array(list(cfg.MEASUREMENT_NOISE_VAR.values()))
Q = np.diag(var_model)
R = np.diag(var_meas[:cfg.MEAS_NUM])
H = np.hstack((np.eye(cfg.MEAS_NUM), np.zeros((cfg.MEAS_NUM, cfg.STATE_NUM - cfg.MEAS_NUM))))
meas_std = np.sqrt(np.diag(R))

process_noise_cv = {cfg.STATE_NAMES.index(s): cv for s, cv in cfg.PROCESS_NOISE_CV.items()}
no_update_indices = {cfg.STATE_NAMES.index(s) for s in getattr(cfg, "NO_UPDATE_STATES", [])}
clip_indices = {cfg.STATE_NAMES.index(s) for s in getattr(cfg, "CLIP_STATES", [])}

# P0: measured -> measurement variance, unmeasured -> process-noise variance
P0_diag = var_model.copy()
P0_diag[:cfg.MEAS_NUM] = var_meas[:cfg.MEAS_NUM]
P0 = np.diag(P0_diag)

# ── Shared data for the dataset (computed inline; no prior-run dependency) ────
volume_results = compute_volume_results(select_datasets(DS), cfg.INITIAL_VOLUMES,
                                        build_schedule, step_len)
data = load_dataset(DS)
_, state_init = get_initial_condition(data["met_df"], data["nsd_df"])
Fin, Fout, Gal_feed, Urd_feed = build_schedule(DS)
V_traj = volume_results[DS][1:]

set_meas = data["set_meas"][:, :cfg.MEAS_NUM].astype(float)
asn_full = data["set_meas"][:, 8].astype(float) if data["set_meas"].shape[1] > 8 else None
nsd_vals = pd.DataFrame(data["NSD_meas"]).apply(pd.to_numeric, errors="coerce").to_numpy()

T_meas = np.array(cfg.T_MEAS_FIXED)
time_steps_A = [round(i * dt_kf, 2) for i in range(N_kf)]
meas_time_to_index = {round(t, 2): i for i, t in enumerate(T_meas.tolist())}
meas_grid_idx = [min(int(round(t / dt_kf)) + 1, N_kf) for t in T_meas]  # post-update index

n_nsd = nsd_vals.shape[1]
nsd_state_idx = list(range(cfg.STATE_NUM - n_nsd, cfg.STATE_NUM))
nsd_names = [cfg.STATE_NAMES[i] for i in nsd_state_idx]

# Reported NSDs — the three reliably-measured nucleotide sugars used for the NSD
# calibration summary throughout the paper (UDP-Gal, UDP-Glc, UDP-GlcNAc). The
# figure's NSD panels and the console NSD columns are restricted to these.
REPORTED_NSD = [n for n in ["UDPGal", "UDPGlc", "UDPGlcNAc"] if n in nsd_names]
REPORTED_LABEL = {"UDPGal": "UDP-Gal", "UDPGlc": "UDP-Glc", "UDPGlcNAc": "UDP-GlcNAc"}
ASN_IDX = cfg.STATE_NAMES.index("Asn")

# Divergence gate operates on the unmeasured states (observable-unmeasured Asn/Glu + the 7 NSDs) —
# identical rule to the tuning/validation sweeps: pool-relative peak-sigma outlier, per ensemble size.
obs_idx = [cfg.STATE_NAMES.index(s) for s in getattr(cfg, "ALPHA_OBS_STATES", ["Asn", "Glu"])]
unmeas_idx = obs_idx + nsd_state_idx

# Normalisation scales = median |measurement| per state
def med_scale(v):
    v = v[~np.isnan(v)]
    return max(np.median(np.abs(v)), 1e-12) if v.size else 1.0
meas_scales = np.array([med_scale(set_meas[:, j]) for j in range(cfg.MEAS_NUM)])
nsd_scales = np.array([med_scale(nsd_vals[:, j]) for j in range(n_nsd)])
asn_scale = med_scale(asn_full) if asn_full is not None else 1.0


def run_pass(N, seed):
    """One production-config EnKF pass; return per-metric diagnostics."""
    np.random.seed(seed)
    # perturbed measurement ensemble
    N_meas_time = set_meas.shape[0]
    set_meas_ens = np.zeros((N_meas_time, N, cfg.MEAS_NUM))
    for i in range(N_meas_time):
        noise = np.random.multivariate_normal(np.zeros(cfg.MEAS_NUM), R, size=N)
        for j in range(cfg.MEAS_NUM):
            noise[:, j] = np.clip(noise[:, j], -3 * meas_std[j], 3 * meas_std[j])
        set_meas_ens[i] = np.clip(set_meas[i] + noise, a_min=1e-12, a_max=None)

    enkf = EnsembleKalmanFilter(cfg.STATE_NUM, cfg.MEAS_NUM)
    enkf.x = state_init.copy()
    enkf.Q = Q.copy(); enkf.R = R.copy(); enkf.H = H.copy()
    enkf.fx = model_step; enkf.dt = dt_kf
    enkf.process_noise_cv = dict(process_noise_cv)
    enkf.no_update_indices = set(no_update_indices)
    enkf.clip_indices = set(clip_indices)          # <-- production clipping (was missing)
    enkf.create_ensemble(N, P0)

    mean_traj = [enkf.x.copy()]
    std_traj = [np.std(enkf.X, axis=0)]
    innovations, innov_covs, meas_at_updates = [], [], []

    t0 = time.time()
    for idx_A, step_A in enumerate(time_steps_A):
        enkf.predict({"Fin": Fin[idx_A], "Fout": Fout[idx_A], "V": V_traj[idx_A],
                      "Gal_feed": Gal_feed[idx_A], "Urd_feed": Urd_feed[idx_A]})
        if step_A in meas_time_to_index:
            b = meas_time_to_index[step_A]
            Z = enkf.X @ enkf.H.T
            z_mean = Z.mean(axis=0)
            Ez = Z - z_mean
            S = (Ez.T @ Ez) / (N - 1) + R
            innovations.append(set_meas[b] - z_mean)
            innov_covs.append(np.diag(S).copy())
            meas_at_updates.append(set_meas[b])
            enkf.update(set_meas_ens[b])
        mean_traj.append(enkf.x.copy())
        std_traj.append(np.std(enkf.X, axis=0))
    wall = time.time() - t0

    mean_traj = np.array(mean_traj); std_traj = np.array(std_traj)
    innovations = np.array(innovations); innov_covs = np.array(innov_covs)
    n_up = len(innovations)

    # measured: NRMSE, NIS = mean(d^2/S), 2-sigma coverage
    m_nrmse = np.zeros(cfg.MEAS_NUM); m_nis = np.zeros(cfg.MEAS_NUM); m_cov = np.zeros(cfg.MEAS_NUM)
    for j in range(cfg.MEAS_NUM):
        pred = np.interp(T_meas, T_kf, mean_traj[:, j])
        mask = ~np.isnan(set_meas[:, j])
        m_nrmse[j] = np.sqrt(np.mean((set_meas[mask, j] - pred[mask]) ** 2)) / meas_scales[j]
        m_nis[j] = np.mean(innovations[:, j] ** 2 / innov_covs[:, j])
        m = mean_traj[meas_grid_idx, j]; s = std_traj[meas_grid_idx, j]
        mm = ~np.isnan(set_meas[:, j]) & (s > 0)
        m_cov[j] = 100.0 * np.mean(np.abs(set_meas[mm, j] - m[mm]) <= 2 * s[mm]) if mm.any() else np.nan

    # NSD (all 7): NRMSE, coverage, spread-skill
    nsd_nrmse = {}; nsd_cov = {}; nsd_ss = {}
    for col, si in enumerate(nsd_state_idx):
        meas = nsd_vals[:, col]
        m = mean_traj[meas_grid_idx, si]; s = std_traj[meas_grid_idx, si]
        valid = ~np.isnan(meas) & (s > 0)
        if valid.sum() == 0:
            nsd_nrmse[nsd_names[col]] = nsd_cov[nsd_names[col]] = nsd_ss[nsd_names[col]] = np.nan
            continue
        err = meas[valid] - m[valid]
        rmse = np.sqrt(np.mean(err ** 2))
        nsd_nrmse[nsd_names[col]] = rmse / nsd_scales[col]
        nsd_cov[nsd_names[col]] = 100.0 * np.mean(np.abs(err) <= 2 * s[valid])
        nsd_ss[nsd_names[col]] = np.mean(s[valid]) / rmse if rmse > 0 else np.nan

    # Asn
    if asn_full is not None:
        pred = np.interp(T_meas, T_kf, mean_traj[:, ASN_IDX])
        mask = ~np.isnan(asn_full)
        asn_nrmse = np.sqrt(np.mean((asn_full[mask] - pred[mask]) ** 2)) / asn_scale
    else:
        asn_nrmse = np.nan

    out = {
        "seed": int(seed),
        # peak (over time) ensemble std per unmeasured state — input to the divergence gate
        "peak_sigma": {i: float(std_traj[:, i].max()) for i in unmeas_idx},
        "wall_time_s": wall,
        "meas_nrmse_mean": np.mean(m_nrmse), "meas_nis_mean": np.mean(m_nis),
        "meas_cov_mean": np.nanmean(m_cov),
        "nsd_nrmse": nsd_nrmse, "nsd_cov": nsd_cov, "nsd_ss": nsd_ss,
        "nsd_nrmse_mean": np.nanmean(list(nsd_nrmse.values())),
        "nsd_ss_median": np.nanmedian(list(nsd_ss.values())),
        "asn_nrmse": asn_nrmse,
    }

    # Also persist the (downsampled) mean/std trajectories + the raw innovations at
    # update times, so any trajectory-level statistic can be recomputed later without
    # re-running. Full ensemble is NOT stored (too large); mean+std+innovations
    # reconstruct every diagnostic this script reports.
    d = int(args.traj_down)
    if d > 0:
        out["traj"] = {
            "down": d,
            "state_names": list(cfg.STATE_NAMES),
            "T": T_kf[::d].copy(),              # (M,)      downsampled time grid
            "mean": mean_traj[::d].copy(),      # (M, 17)   ensemble mean
            "std": std_traj[::d].copy(),        # (M, 17)   ensemble std (uncertainty band)
        }
        out["innov"] = {                        # at the 17 measurement-update times
            "T_meas": T_meas.copy(),
            "meas_names": list(cfg.MEASURED_STATES),
            "d": innovations.copy(),            # (n_up, 8) innovation z - forecast mean
            "S_diag": innov_covs.copy(),        # (n_up, 8) diag(P_zz + R)
            "z": np.array(meas_at_updates),     # (n_up, 8) actual measurements
        }
    return out


# ── Sweep (per-seed cache; divergence gate + resample; resumable at pass granularity) ─
def load_pool(pkl_path):
    """Return {seed: run_dict} of already-computed passes for a size (resume)."""
    if not (args.resume and pkl_path.exists()):
        return {}
    with open(pkl_path, "rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, dict):                       # current format: seed -> run
        return {int(k): v for k, v in obj.items()}
    return {args.seed_offset + i: r for i, r in enumerate(obj)}   # legacy list -> sequential seeds


def gate(pool):
    """Pool-relative peak-sigma outlier rule, per size (identical to tuning/validation)."""
    med = {i: float(np.median([pool[s]["peak_sigma"][i] for s in pool])) for i in unmeas_idx}
    good, rej = [], []
    for s in sorted(pool):
        ps = pool[s]["peak_sigma"]
        bad = any(ps[i] > args.reject_mult * med[i] and med[i] > 0 for i in unmeas_idx)
        (rej if bad else good).append(s)
    return good, rej


all_results = {}                                    # N -> list of the TARGET_GOOD clean runs
selection = {}                                      # N -> {"used": [...], "rejected": [...]}
for N in SIZES:
    pkl_name = f"ensemble_N{N}.pkl"
    pkl_path = OUT_DIR / pkl_name
    pool = load_pool(pkl_path)
    print(f"\n  N={N}:" + (f" resuming — {len(pool)} pass(es) cached." if pool else ""), flush=True)

    def add(seed):
        res = run_pass(N, seed)
        pool[seed] = res
        save_pkl(pool, pkl_name)                    # crash-safe: persist after EVERY pass
        print(f"    seed {seed}: {res['wall_time_s']:.0f}s  "
              f"NIS={res['meas_nis_mean']:.2f} NSD-ss={res['nsd_ss_median']:.2f}", flush=True)

    if not args.auto_reject:
        for seed in range(args.seed_offset, args.seed_offset + N_RUNS):
            if seed not in pool:
                add(seed)
        used = list(range(args.seed_offset, args.seed_offset + N_RUNS))
        rej = []
    else:
        cand = args.seed_offset
        while cand in pool:
            cand += 1
        cap = args.seed_offset + args.max_seeds
        while True:                                 # draw until TARGET_GOOD clean, then verify gate
            while len(pool) < TARGET_GOOD and cand < cap:
                add(cand); cand += 1
            good, rej = gate(pool)
            if len(good) >= TARGET_GOOD or cand >= cap:
                break
            add(cand); cand += 1
        used = good[:TARGET_GOOD]
        if rej:
            print(f"    N={N}: REJECTED divergent seed(s) {rej} "
                  f"({len(rej)}/{len(pool)} drawn); using {used}", flush=True)
        if len(used) < TARGET_GOOD:
            print(f"    N={N}: WARNING only {len(used)}/{TARGET_GOOD} clean seeds within "
                  f"--max-seeds={args.max_seeds}", flush=True)

    all_results[N] = [pool[s] for s in used]
    selection[N] = {"used": used, "rejected": rej, "n_drawn": len(pool)}

# ── Divergence / seed-selection manifest ─────────────────────────────────────
save_pkl({str(N): selection[N] for N in SIZES}, "seed_selection.pkl")
with open(OUT_DIR / "seed_selection.json", "w") as f:
    json.dump({"reject_mult": args.reject_mult if args.auto_reject else None,
               "target_good": TARGET_GOOD, "auto_reject": bool(args.auto_reject),
               "sizes": {str(N): selection[N] for N in SIZES}}, f, indent=2)

# ── Combined trajectory archive: mean + spread of EVERY (clean) run, all sizes ──
# One self-contained pkl carrying the seed-averaging inputs for the whole sweep. Rejected
# passes are NOT dropped from disk — they remain in each ensemble_N<N>.pkl {seed: run} cache.
_MET_KEYS = ["wall_time_s", "meas_nrmse_mean", "meas_nis_mean", "meas_cov_mean",
             "nsd_nrmse_mean", "nsd_ss_median", "asn_nrmse"]
combined = {
    "dataset": DS,
    "sizes": SIZES,
    "state_names": list(cfg.STATE_NAMES),
    "reject_mult": args.reject_mult if args.auto_reject else None,
    "target_good": TARGET_GOOD,
    "provenance": {                                  # exact filter config these runs used
        "tuning_set": "P4-fold calibration (adopted production config, nsd_enkf/config.py)",
        "process_noise_cv": dict(cfg.PROCESS_NOISE_CV),
        "alpha_obs": float(cfg.PROCESS_NOISE_ALPHA_OBS),
        "alpha_nsd": float(cfg.PROCESS_NOISE_ALPHA),
        "ensemble_dt": cfg.DT, "T_end": cfg.T_END,
    },
    "runs": {},                                      # N -> {used_seeds, rejected_seeds, runs:[...]}
}
for N in SIZES:
    entries = []
    for r in all_results[N]:
        e = {"seed": r["seed"],
             "metrics": {k: r[k] for k in _MET_KEYS if k in r},
             "nsd_nrmse": r.get("nsd_nrmse"), "nsd_cov": r.get("nsd_cov"),
             "nsd_ss": r.get("nsd_ss")}
        if "traj" in r:                              # mean trajectory + spread (std), downsampled
            e["T"] = r["traj"]["T"]
            e["mean"] = r["traj"]["mean"]            # (M, 17) ensemble mean
            e["std"] = r["traj"]["std"]              # (M, 17) ensemble std = uncertainty band
        entries.append(e)
    combined["runs"][N] = {"used_seeds": selection[N]["used"],
                           "rejected_seeds": selection[N]["rejected"], "runs": entries}
save_pkl(combined, "all_trajectories.pkl")
n_traj = sum(len(combined["runs"][N]["runs"]) for N in SIZES)
print(f"\nSaved combined trajectory archive: {OUT_DIR / 'all_trajectories.pkl'} "
      f"({n_traj} runs across {len(SIZES)} sizes)"
      + ("" if args.traj_down > 0 else "  [NOTE: --traj-down 0 -> no trajectories stored]"))

# ── Aggregate ────────────────────────────────────────────────────────────────
def agg(N, key):
    return np.array([r[key] for r in all_results[N]])

summary = []
for N in SIZES:
    row = {"N": N, "n_used": len(all_results[N]), "n_rejected": len(selection[N]["rejected"]),
           "n_drawn": selection[N]["n_drawn"], "rejected_seeds": selection[N]["rejected"]}
    for key in ["wall_time_s", "meas_nrmse_mean", "meas_nis_mean", "meas_cov_mean",
                "nsd_nrmse_mean", "nsd_ss_median", "asn_nrmse"]:
        v = agg(N, key)
        row[key + "_mean"] = np.nanmean(v); row[key + "_std"] = np.nanstd(v)
    for name in nsd_names:
        v = np.array([r["nsd_nrmse"][name] for r in all_results[N]])
        row[f"nsd_{name}_mean"] = np.nanmean(v); row[f"nsd_{name}_std"] = np.nanstd(v)
    # Reported-NSD aggregates: mean over the 3 reported deposits, per run, then across runs
    rep_nrmse = np.array([np.nanmean([r["nsd_nrmse"][n] for n in REPORTED_NSD])
                          for r in all_results[N]])
    rep_ss = np.array([np.nanmean([r["nsd_ss"][n] for n in REPORTED_NSD])
                       for r in all_results[N]])
    row["nsd_nrmse_reported_mean"] = np.nanmean(rep_nrmse)
    row["nsd_nrmse_reported_std"] = np.nanstd(rep_nrmse)
    row["nsd_ss_reported_mean"] = np.nanmean(rep_ss)
    row["nsd_ss_reported_std"] = np.nanstd(rep_ss)
    summary.append(row)
save_pkl(summary, "ensemble_sensitivity_summary.pkl")

print("\n" + "=" * 100)
print(f"{'N':>5s} | {'used/div':>9s} | {'Time(s)':>9s} | {'NIS':>9s} | {'MetNRMSE':>10s} | "
      f"{'MetCov%':>9s} | {'rNSD_NRMSE':>10s} | {'rNSD_ss':>8s}")
print("-" * 100)
for s in summary:
    print(f"{s['N']:>5d} | {s['n_used']:>3d}/{s['n_rejected']:<5d} | "
          f"{s['wall_time_s_mean']:>4.0f}±{s['wall_time_s_std']:>3.0f} | "
          f"{s['meas_nis_mean_mean']:>4.2f}±{s['meas_nis_mean_std']:>3.2f} | "
          f"{s['meas_nrmse_mean_mean']:>10.4f} | "
          f"{s['meas_cov_mean_mean']:>4.0f}±{s['meas_cov_mean_std']:>3.0f} | "
          f"{s['nsd_nrmse_reported_mean']:>10.4f} | "
          f"{s['nsd_ss_reported_mean']:>4.2f}±{s['nsd_ss_reported_std']:>3.2f}")
if args.auto_reject:
    print(f"\n  divergence rejection ON (peak sigma > {args.reject_mult}x per-size median);"
          f" 'used/div' = clean runs kept / divergent seeds rejected.")
else:
    print("\n  divergence rejection OFF (--no-reject): all seeds aggregated.")

# ── Figure (bold paper-figure style: LABEL 14 / TICK 12 / TITLE 14, spine 1.5) ─
LABEL_FS, TICK_FS, TITLE_FS = 14, 12, 14
Ns = [s["N"] for s in summary]
def eb(ax, key, color, marker, ylabel, title):
    ax.errorbar(Ns, [s[key + "_mean"] for s in summary], [s[key + "_std"] for s in summary],
                fmt=f"{marker}-", color=color, lw=2, ms=8, capsize=4,
                markeredgecolor="black", markeredgewidth=0.8)
    ax.set_xlabel("Ensemble size $N$", fontsize=LABEL_FS, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=LABEL_FS, fontweight="bold")
    ax.set_title(title, fontsize=TITLE_FS, fontweight="bold", loc="left")
    ax.tick_params(axis="both", labelsize=TICK_FS)
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_fontweight("bold")
    for sp in ax.spines.values():
        sp.set_linewidth(1.5)
    ax.grid(alpha=0.25)

fig, ax = plt.subplots(2, 3, figsize=(16, 9))
eb(ax[0, 0], "meas_nrmse_mean", "tab:red", "o", "Normalised RMSE", "(a) Measured — NRMSE")
eb(ax[0, 1], "meas_nis_mean", "tab:blue", "s", "Mean NIS", "(b) Measured — consistency (NIS)")
eb(ax[0, 2], "meas_cov_mean", "tab:green", "^", "2σ coverage (%)", "(c) Measured — 2σ coverage")
eb(ax[1, 0], "nsd_nrmse_mean", "tab:purple", "o", "Normalised RMSE", "(d) NSDs — NRMSE")
eb(ax[1, 1], "nsd_ss_reported", "tab:orange", "D", "Spread-to-error",
   "(e) NSDs — spread-to-error")
ax[1, 1].axhline(1.0, ls="--", color="0.5", lw=1.5, zorder=1, label="Ideal (= 1)")
ax[1, 1].legend(frameon=False, prop={"weight": "bold", "size": TICK_FS}, loc="best")
eb(ax[1, 2], "wall_time_s", "tab:gray", "o", "Wall time (s)", "(f) Cost per pass")
plt.tight_layout()
out = OUT_DIR / "ensemble_size_sensitivity.png"
plt.savefig(out, dpi=300, bbox_inches="tight")
paper_fig = Path("/Users/luxi.yu/Research/Soft_Sensing_Paper/Figs/ensemble_size_sensitivity.png")
if paper_fig.parent.exists():
    plt.savefig(paper_fig, dpi=300, bbox_inches="tight")
    print(f"Saved figure: {paper_fig}")
plt.close()
print(f"\nSaved figure: {out}")
print("Done.")
