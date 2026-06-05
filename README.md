# Soft Sensing of Intracellular States for CHO Cell Bioprocessing with Ensemble Kalman Filters

This repository contains the code accompanying the preprint:

> **Soft Sensing of Intracellular States for CHO Cell Bioprocessing with Ensemble Kalman Filters**
> Luxi Yu, Antonio del Rio Chanona, Cleo Kontoravdi
> *bioRxiv*, 2026. DOI: [10.64898/2026.05.28.728559](https://doi.org/10.64898/2026.05.28.728559)

## Abstract

In biotherapeutic manufacturing, product quality such as glycosylation profile is typically assessed only after harvest, limiting opportunities for corrective action during cell culture operation. Intracellular nucleotide sugar donors (NSD) directly determine glycosylation outcomes but are rarely measured, even offline, due to analytical complexity and process disruption. This work introduces a model-based soft sensing framework to infer NSD concentrations from readily available extracellular measurements. A Bayesian state estimation approach based on the Ensemble Kalman Filter (EnKF) is developed to reconstruct unmeasured intracellular states during CHO cell culture. An imperfect kinetic process model is combined with noisy extracellular measurements, explicitly accounting for process variability and measurement uncertainty through ensemble-based propagation and updates. The framework is validated using four independent experiments with distinct feeding perturbations that are not used for model calibration. Building on corrected extracellular dynamics, the EnKF demonstrated robust estimation of a growth-determining amino acid, asparagine, from correlated extracellular states. Based on the improved extracellular and amino acid estimates, the framework further enabled reliable inference of intracellular NSDs across all experiments.

**Keywords:** Bioprocess Modelling, State Estimation, Soft Sensing, Ensemble Kalman Filter, Chinese hamster ovary cells, antibody production

## Repository Structure

```
EnKF-Soft-Sensing/
├── JPC_NSD_softsensing.ipynb   # Main notebook: EnKF implementation and analysis
├── data/
│   └── raw/                    # Experimental data (P1–P4 fed-batch experiments)
├── results/                    # Generated figures and outputs
├── pyproject.toml              # Python dependencies (managed by Poetry)
├── poetry.lock
└── LICENSE
```

## Installation

### Prerequisites
- Python 3.14+
- [Poetry](https://python-poetry.org/) for dependency management

### Setup
```bash
git clone https://github.com/ly1815/EnKF-Soft-Sensing.git
cd EnKF-Soft-Sensing
poetry install
poetry shell
```

## Usage

Launch the Jupyter notebook and run cells sequentially to reproduce the analysis:

```bash
jupyter notebook JPC_NSD_softsensing.ipynb
```

The notebook covers:
- Mechanistic model definition (extracellular cell culture + intracellular NSD metabolism)
- Empirical observability analysis via the observability Gramian
- Ensemble Kalman Filter implementation and state estimation
- Validation against four independent fed-batch experiments (P1–P4)
- Visualization of estimated extracellular metabolites, asparagine, and intracellular NSDs

## Data

Experimental data are from fed-batch CHO-T127 cell cultures with distinct galactose and uridine feeding strategies. Five datasets were used for model calibration; four additional independent experiments (P1–P4) are used for EnKF validation. Raw data files are located in `data/raw/`.

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@article{yu2026softsensing,
  title={Soft Sensing of Intracellular States for {CHO} Cell Bioprocessing with Ensemble Kalman Filters},
  author={Yu, Luxi and del Rio Chanona, Antonio and Kontoravdi, Cleo},
  journal={bioRxiv},
  year={2026},
  doi={10.64898/2026.05.28.728559}
}
```

## Contact

Luxi Yu — Department of Chemical Engineering, Imperial College London

Contact: luxiyu611@gmail.com
