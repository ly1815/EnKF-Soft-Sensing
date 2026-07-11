# `results/` — default output root (starts empty)

This is the **default output directory** for the pipeline scripts. A fresh clone starts with
this folder empty (only this README); it is populated when you run the analysis.

Each script writes to `results/<run>/`, where `<run>` is either `config.RUN_NAME`
(currently `tuned_v1`) or the script's `--run` argument:

| Script | Default output |
|---|---|
| `scripts/run_enkf.py`, `scripts/plot_results.py` | `results/<RUN_NAME>/` |
| `scripts/01_tune_cv.py` | `results/cv_tuning/` |
| `scripts/02_tune_alpha_asn.py` | `results/alpha_asn/` |
| `scripts/03_tune_alpha_nsd.py` | `results/alpha_nsd/` |
| `scripts/05_ensemble_size.py` | `results/ensemble_sens/` |

## What is tracked vs regenerable

Per the repo `.gitignore`, only lightweight, provenance artifacts are committed here
(`.png` figures, `.json`), while the heavy regenerable EnKF trajectories (`*.pkl`) stay on
disk and are **not** committed — any figure or statistic can be re-derived from the pkls by
re-running the plotting scripts.

## The other `results_*` folders

The curated study outputs live in dedicated top-level folders, each with its own README:

- `results_single_sweep/` — per-fold cross-validation tuning (calibrated CVs + α_obs / α_nsd
  sweeps + `picks.json`); the main systematic-tuning result.
- `results_multirun_nsd/` — multi-seed α_nsd sweep (seed-averaged calibration).
- `results_asn_alpha0.01/` — α_obs downstream-inertness test (α_obs 0.01 vs 0.002 on the NSDs).
- `results_legacy/` — archive of superseded exploratory runs and the older single-seed
  cross-validation.
