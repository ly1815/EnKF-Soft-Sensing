# Soft Sensing of Intracellular States for CHO Cell Bioprocessing with Ensemble Kalman Filters

[![bioRxiv](https://img.shields.io/badge/bioRxiv-2026.05.28.728559-b31b1b.svg)](https://doi.org/10.64898/2026.05.28.728559) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This repository contains the code accompanying the preprint:

> **Soft Sensing of Intracellular States for CHO Cell Bioprocessing with Ensemble Kalman Filters**
> Luxi Yu, Antonio del Rio Chanona, Cleo Kontoravdi
> *bioRxiv* (2026). doi: [10.64898/2026.05.28.728559](https://doi.org/10.64898/2026.05.28.728559)

## Overview

We present an **Ensemble Kalman Filter (EnKF)** framework for soft sensing of unmeasured intracellular states in Chinese Hamster Ovary (CHO) cell culture bioprocesses. An imperfect kinetic process model is combined with noisy extracellular measurements, explicitly accounting for process variability and measurement uncertainty through ensemble-based propagation and updates.

Intracellular nucleotide sugar donors (NSDs) directly determine glycosylation outcomes but are rarely measured due to analytical complexity and process disruption. The EnKF dynamically infers NSD concentrations from routinely available extracellular measurements, enabling earlier and quality-relevant insight into glycosylation-critical intracellular dynamics.

The framework is validated using four independent fed-batch experiments (P1-P4) with distinct galactose and uridine feeding strategies that are not used for model calibration.

## Repository Structure

```
EnKF-Soft-Sensing/
├── nsd_enkf/                # Core library
│   ├── config.py            #   Constants, model & noise parameters
│   ├── data_loader.py       #   Excel data loading, feed schedules
│   ├── model.py             #   17-state ODE model, volume integration
│   ├── enkf.py              #   EnsembleKalmanFilter class & runners
│   ├── analysis.py          #   RMSE, measurement ensembles, Gramian
│   ├── plotting.py          #   Publication-quality plotting functions
│   └── io_utils.py          #   Pickle I/O helpers
├── scripts/                 # Numbered systematic-tuning pipeline (run in order)
│   ├── 01_tune_cv.py        #   Stage 3  measured-state CVs -> NIV=1 (cap 0.006)
│   ├── 02_tune_alpha_asn.py #   Stage 4a observable-tier alpha (Asn & Glu)
│   ├── 03_tune_alpha_nsd.py #   Stage 4b NSD alpha (band inspection)
│   ├── 04_cross_validate.py #   full-fold cross-validation of the tuning procedure
│   ├── 05_ensemble_size.py  #   Stage 5  ensemble-size sensitivity (justify N=100)
│   ├── run_enkf.py          #   utility: run the production EnKF pipeline
│   ├── plot_results.py      #   utility: generate figures from a run
│   └── legacy/              #   superseded scripts (kept for provenance)
├── docs/
│   ├── tuning_strategy.md   #   the systematic tuning method (manuscript-ready)
│   └── tuning_log.md        #   chronological tuning decision log
├── data/raw/                # Experimental data (P1-P4, not in git)
├── results/                 # Generated outputs (pkls gitignored, figures tracked)
├── pyproject.toml
├── poetry.lock
└── LICENSE
```

## Quick Start

### Install dependencies

```bash
pip install poetry
poetry install
```

### Run the EnKF pipeline

```bash
# Run with default parameters (all datasets, 10 runs, N=100)
poetry run python scripts/run_enkf.py

# Override parameters
poetry run python scripts/run_enkf.py --run my_experiment --kq 0.5 --n-runs 5 --ensemble-size 200
```

### Generate figures

```bash
# All figures
poetry run python scripts/plot_results.py --run my_experiment

# Only specific figure groups
poetry run python scripts/plot_results.py --run my_experiment --only uncertainty
poetry run python scripts/plot_results.py --run my_experiment --only diagnostics
```

## Systematic covariance tuning

The EnKF noise parameters are calibrated by an ordered, reproducible-from-data procedure.
The full method (criteria, reasoning, dependencies) is in
[`docs/tuning_strategy.md`](docs/tuning_strategy.md); the decision history is in
[`docs/tuning_log.md`](docs/tuning_log.md). The `scripts/` are numbered to match the steps:

| Step | Script | What it tunes | Metric |
|------|--------|---------------|--------|
| 3  | `01_tune_cv.py`        | measured-state per-step CVs (multiplicative noise) | NIV → 1 (filter consistency), cap `CV_MAX=0.006` |
| 4a | `02_tune_alpha_asn.py` | observable-tier additive α (Asn & Glu, shared)     | Asn NRMSE + coverage |
| 4b | `03_tune_alpha_nsd.py` | NSD additive α (7 intracellular states)            | NSD NRMSE + band inspection |
| —  | `04_cross_validate.py` | full-fold CV of the whole procedure across P1–P4   | held-out NSD/Asn NRMSE + coverage |
| 5  | `05_ensemble_size.py`  | verify ensemble size N                             | NIS / coverage / spread-skill vs N |

Measurement noise `R` and the initial covariance `P0` are set from data in `config.py`
(Stages 0–2), not by these scripts. Each script tunes on **P4** and validates on **P1–P3**;
`04_cross_validate.py` rotates that split across all four batches. **Every run saves the
full 17-state ensemble mean *and* standard-deviation (uncertainty) trajectories** to
`results/<run>/pkl/` (bands = mean ± k·std), plus figures to `results/<run>/figures/`.

### Cross-validation (`04_cross_validate.py`) — modes A and B

`04` re-tunes on each fold and evaluates on the held-out batches. It is **fully automated**
— no manual choices per fold. Two re-tune modes, selected with `--retune`:

- **Mode A — `--retune cv`** *(default, ~6.6 h)*: per fold, re-calibrate only the measured
  CVs (automated NIV=1); hold α_obs/α_nsd at the adopted config constants. Tests whether the
  *automated calibration* generalizes.
- **Mode B — `--retune all`** *(~14 h)*: additionally auto-select α_nsd and α_obs (argmin
  NRMSE) per fold. Fully data-driven; heavier.

Run them on separate days (each resumable); the A-vs-B comparison is assembled from
whatever is already on disk. To keep each run short, add `--train <dataset>` to do **one
fold at a time** (~1.6 h), accumulating into the same summary:

```bash
# Mode A — whole CV, or one fold at a time:
caffeinate -i ./.venv/bin/python scripts/04_cross_validate.py --retune cv
caffeinate -i ./.venv/bin/python scripts/04_cross_validate.py --retune cv --train P4   # just the P4-trained fold

# Mode B (run after A to also emit the A-vs-B comparison):
caffeinate -i ./.venv/bin/python scripts/04_cross_validate.py --retune all
caffeinate -i ./.venv/bin/python scripts/04_cross_validate.py --retune all --train P4

# --scheme loo for leave-one-out (train on 3, hold out 1); --resume to continue; --retune both for one 20h shot
```

```bash
# Steps 3 -> 4a -> 4b, then cross-validation, then N verification:
caffeinate -i ./.venv/bin/python scripts/01_tune_cv.py --dataset P4
caffeinate -i ./.venv/bin/python scripts/02_tune_alpha_asn.py
caffeinate -i ./.venv/bin/python scripts/03_tune_alpha_nsd.py
caffeinate -i ./.venv/bin/python scripts/04_cross_validate.py --retune cv   # mode A (~6.6h)
caffeinate -i ./.venv/bin/python scripts/04_cross_validate.py --retune all  # mode B (~14h); emits A-vs-B
# or break it into one short fold per run (accumulates into the same summary):
caffeinate -i ./.venv/bin/python scripts/04_cross_validate.py --retune cv --train P4   # ~1.6h; then --train P1/P2/P3
caffeinate -i ./.venv/bin/python scripts/05_ensemble_size.py
```

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
