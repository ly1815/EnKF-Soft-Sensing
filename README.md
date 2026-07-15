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
