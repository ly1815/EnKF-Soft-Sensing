# Soft Sensing of Intracellular States for CHO Cell Bioprocessing with Ensemble Kalman Filters

[![bioRxiv](https://img.shields.io/badge/bioRxiv-2026.05.28.728559-b31b1b.svg)](https://doi.org/10.64898/2026.05.28.728559) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This repository contains the code accompanying the preprint:

> **Soft Sensing of Intracellular States for CHO Cell Bioprocessing with Ensemble Kalman Filters**
> Luxi Yu, Antonio del Rio Chanona, Cleo Kontoravdi
> *bioRxiv* (2026). doi: [10.64898/2026.05.28.728559](https://doi.org/10.64898/2026.05.28.728559)

## Overview

We present an **Ensemble Kalman Filter (EnKF)** framework for soft sensing of unmeasured intracellular states in Chinese Hamster Ovary (CHO) cell culture bioprocesses. An imperfect kinetic process model is combined with noisy extracellular measurements, explicitly accounting for process variability and measurement uncertainty through ensemble-based propagation and updates.

Intracellular nucleotide sugar donors (NSDs) directly determine glycosylation outcomes but are rarely measured due to analytical complexity and process disruption. The EnKF dynamically infers NSD concentrations from routinely available extracellular measurements, enabling earlier and quality-relevant insight into glycosylation-critical intracellular dynamics.

The framework is validated using four independent fed-batch experiments (P1–P4) with distinct galactose and uridine feeding strategies that are not used for model calibration.

## Repository Structure

```
EnKF-Soft-Sensing/
├── JPC_NSD_softsensing.ipynb   # Main notebook: EnKF implementation and analysis
├── data/
│   └── raw/                    # Experimental data (P1–P4 fed-batch experiments)
├── results/                    # Generated outputs (gitignored)
├── pyproject.toml              # Poetry project metadata
├── poetry.lock
└── LICENSE
```

## Datasets

| Exp. | Galactose (mM) |  |  |  | Uridine (mM) |  |  |  |
|------|------|------|------|--------|------|------|------|--------|
|      | Day 4 | Day 6 | Day 8 | Day 10 | Day 4 | Day 6 | Day 8 | Day 10 |
| P1   | 79.4 | 15.4 | 11.0 | 248.3 | 15.9 | 3.1 | 2.2 | 49.7 |
| P2   | 4.3 | 168.3 | 37.7 | 11.4 | 0.9 | 33.7 | 7.5 | 2.3 |
| P3   | 5.2 | 3.1 | 235.3 | 249.9 | 1.0 | 0.6 | 47.1 | 50.0 |
| P4   | 21.9 | 6.4 | 233.5 | 4.0 | 4.4 | 1.3 | 46.7 | 0.8 |

P4 used for EnKF covariance tuning; P1–P3 used for validation. All experiments use the IgG-producing CHO-T127 cell line in 500 mL shake flasks with CD CHO medium at 36.5 °C.

## Quick Start

### Install dependencies

```bash
pip install poetry
poetry install
```

### Run the analysis

```bash
poetry run jupyter notebook JPC_NSD_softsensing.ipynb
```

The notebook covers:
- Mechanistic model definition (extracellular cell culture + intracellular NSD metabolism)
- Empirical observability analysis via the observability Gramian
- EnKF implementation and state estimation
- Validation against four independent fed-batch experiments
- Visualization of estimated extracellular metabolites, asparagine, and intracellular NSDs

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
