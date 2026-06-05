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
