# `results_legacy/` — archive of superseded runs

Kept for provenance and reproducibility of the decision record (`docs/tuning_log.md`); **not**
part of the current adopted pipeline. Live outputs go to `results/` (new runs) and the curated
`results_single_sweep/`, `results_multirun_nsd/`, `results_asn_alpha0.01/` folders.

## Contents

- **`cross_validation/`** — the *older* single-seed cross-validation run (`cv/` = train-1,
  `all/` = train-3 schemes) with held-out grids and comparison. Superseded by the per-fold
  independent design in `results_single_sweep/`.
- **`ensemble_sens/`** — an early ensemble-size sensitivity study (Stage 5). A fresh run of
  `scripts/05_ensemble_size.py` writes to `results/ensemble_sens/`; this copy is the archived one.
- **`tests/`** — superseded exploratory sweeps and calibrations kept as a trail:
  `alpha_asn/`, `alpha_check/`, `alpha_nsd/`, `cv_tuning/`, `cv_tuning_glcclip/` (Glc-clip
  trial), `ob_wide/`, `quick_test/`, and an earlier `ensemble_sens/`.

Why these were superseded (R → auto-CV → re-swept α, NRMSE→calibration α selection, the
odeint memory fix, the CV cap ratcheting to 0.006, etc.) is documented in
[`docs/tuning_log.md`](../docs/tuning_log.md). Heavy `*.pkl` are git-ignored; figures + JSON
are tracked.
