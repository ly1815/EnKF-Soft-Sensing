# Soft Sensing of Intracellular States for CHO Cell Bioprocessing with Ensemble Kalman Filters

[![bioRxiv](https://img.shields.io/badge/bioRxiv-2026.05.28.728559-b31b1b.svg)](https://doi.org/10.64898/2026.05.28.728559) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This repository contains the code accompanying the preprint:

> **Soft Sensing of Intracellular States for CHO Cell Bioprocessing with Ensemble Kalman Filters**
> Luxi Yu, Antonio del Rio Chanona, Cleo Kontoravdi
> *bioRxiv* (2026). doi: [10.64898/2026.05.28.728559](https://doi.org/10.64898/2026.05.28.728559)

## Overview

We present an **Ensemble Kalman Filter (EnKF)** framework for soft sensing of unmeasured intracellular states in Chinese Hamster Ovary (CHO) cell culture bioprocesses. An imperfect kinetic process model is combined with noisy extracellular measurements, explicitly accounting for process variability and measurement uncertainty through ensemble-based propagation and updates.

Intracellular nucleotide sugar donors (NSDs) directly determine glycosylation outcomes but are rarely measured due to analytical complexity and process disruption. The EnKF dynamically infers NSD concentrations from routinely available extracellular measurements, enabling earlier and quality-relevant insight into glycosylation-critical intracellular dynamics.

The framework uses four independent fed-batch experiments (P1–P4) with distinct galactose and uridine feeding strategies. All covariance parameters are calibrated **from data** — there are no hand-tuned frozen intermediates — and the filter is stress-tested by seed-averaged, divergence-screened cross-validation.

## Repository structure

```
EnKF-Soft-Sensing/
├── nsd_enkf/                         # Core library
│   ├── config.py                     #   constants, model & noise parameters (R, P0 from data)
│   ├── data_loader.py                #   Excel data loading, feed schedules
│   ├── model.py                      #   17-state ODE model, volume integration
│   ├── enkf.py                       #   EnsembleKalmanFilter class & runners
│   ├── analysis.py                   #   RMSE, measurement ensembles, Gramian
│   ├── plotting.py                   #   publication-quality plotting functions
│   └── io_utils.py                   #   pickle I/O helpers
├── scripts/                          # Pipeline that produces the paper's results (see below)
│   ├── 04_cross_validate.py          #   calibrate CVs + sweep/pick α per fold → results_single_sweep/
│   ├── sweep_alpha_nsd_multiseed.py  #   seed-averaged α_nsd sweep    → results_multirun_nsd/
│   ├── validate_multiseed.py         #   seed-averaged held-out CV    → results_multirun_validation/
│   ├── 05_ensemble_size.py           #   ensemble-size sensitivity    → results_multirun_ensemble_size/
│   ├── plot_maintext_cv.py           #   main-text figures, P1–P4 grid (tuning P4 vs validation P1–P3)
│   └── plot_p4_maintext.py           #   main-text figures, P4 only
├── results_single_sweep/             # per-fold calibrated CVs (cv_final.json) + α picks (picks.json)
├── results_multirun_nsd/             # seed-averaged α_nsd sweep (P4 = adopted tuning fold)
├── results_multirun_validation/      # seed-averaged held-out validation (P1–P3 under the P4-tuned filter)
├── results_multirun_ensemble_size/   # ensemble-size sensitivity / calibration diagnostics
├── docs/                             # tuning_strategy.md, tuning_log.md, final_parameters.md
├── data/raw/                         # experimental data (P1–P4.xlsx)
├── pyproject.toml / poetry.lock
└── LICENSE
```

Heavy EnKF trajectory pickles (`*.pkl`, regenerable) are **not** tracked; each results
folder keeps its figures (`.png`) and small JSON provenance (`cv_final.json`, `picks.json`,
`seed_selection.json`, `summary.json`) in git. Re-running the scripts regenerates the pickles.

## Quick start

```bash
pip install poetry
poetry install
```

## Reproducing the results

The three published results folders are produced by a two-part pipeline: a data-driven
**calibration** step, then three **multi-seed** production runs. Every EnKF pass uses a
distinct seed and archives the full 17-state ensemble mean **and** standard-deviation
(uncertainty) trajectory; the multi-seed scripts report the seed-averaged posterior with
divergent replicates rejected (pool-relative peak-σ rule, identical across all three).

Measurement noise `R` and the initial covariance `P0` are set from data in `config.py`;
they are never fit to filter residuals.

### 1. Calibration → `results_single_sweep/`

`04_cross_validate.py` calibrates, per fold, the 8 measured-state multiplicative CVs by the
fixed-point rule that drives each normalised innovation variance to 1 (NIV → 1, cap
`CV_MAX = 0.006`), and sweeps the additive process-noise scalars α_obs (Asn/Glu tier) and
α_nsd (7 NSDs). It writes `fold_<X>/cv/cv_final.json` — the calibrated CVs that the
multi-seed scripts consume — plus per-alpha sweep figures.

```bash
# sweep stage: per-fold CV calibration + α sweeps (resumable; long — use --train for one fold)
poetry run python scripts/04_cross_validate.py --stage sweep
poetry run python scripts/04_cross_validate.py --stage sweep --train P4 --resume

# after inspecting the sweeps, record the chosen α per fold in results_single_sweep/picks.json, then:
poetry run python scripts/04_cross_validate.py --stage validate --picks results_single_sweep/picks.json
```

`picks.json` format:

```json
{ "P1": {"alpha_obs": 0.002, "alpha_nsd": 0.02},
  "P2": {"alpha_obs": 0.002, "alpha_nsd": 0.02},
  "P3": {"alpha_obs": 0.002, "alpha_nsd": 0.02},
  "P4": {"alpha_obs": 0.002, "alpha_nsd": 0.02} }
```

### 2. `results_multirun_nsd/` — seed-averaged α_nsd sweep

Repeats the α_nsd sweep over N seeds per fold using each fold's calibrated CVs (from step 1)
and the adopted α_obs = 0.002, reporting calibration on the seed-averaged posterior. **P4 is
the adopted tuning fold** for the paper's main text.

```bash
caffeinate -i poetry run python scripts/sweep_alpha_nsd_multiseed.py --n-runs 10 --datasets P4
```

### 3. `results_multirun_validation/` — seed-averaged held-out validation

Applies each training fold's calibrated CVs + picked α (from step 1) to the three held-out
batches it never saw, over N seeds, rejecting divergent replicates. Nothing is tuned here —
the honest generalisation test.

```bash
caffeinate -i poetry run python scripts/validate_multiseed.py --folds P4
```

### 4. `results_multirun_ensemble_size/` — ensemble-size sensitivity

Self-contained (reads the adopted production filter from `config.py`, no dependency on
steps 1–3). Runs N ∈ {25, 50, 100, 150, 200}, 10 seeds each, on P4; rejects divergent
replicates and reports NRMSE / NIS / 2σ coverage / spread-skill / cost vs N — the
justification for N = 100.

```bash
caffeinate -i poetry run python scripts/05_ensemble_size.py --n-runs 10 --resume
```

### 5. Figures

```bash
poetry run python scripts/plot_maintext_cv.py   # P1–P4 grid: P4 (tuning) + P1–P3 (validation), 2σ bands
poetry run python scripts/plot_p4_maintext.py    # P4-only main-text figures
```

Each multi-seed script also writes its own per-run figures into its results folder. All runs
are crash-safe and resumable (per-pass cache); a kill costs at most one pass.

The tuning method (criteria, dependencies, reasoning) is documented in
[`docs/tuning_strategy.md`](docs/tuning_strategy.md); the decision history is in
[`docs/tuning_log.md`](docs/tuning_log.md).

## Citation

If you use this code, please cite:

```bibtex
@article{yu2026softsensing,
  title   = {Soft Sensing of Intracellular States for {CHO} Cell Bioprocessing with Ensemble Kalman Filters},
  author  = {Yu, Luxi and del Rio Chanona, Antonio and Kontoravdi, Cleo},
  journal = {bioRxiv},
  year    = {2026},
  doi     = {10.64898/2026.05.28.728559}
}
```

## Contact

Luxi Yu — luxiyu611@gmail.com

## License

This project is licensed under the [MIT License](LICENSE).
