# `results_multirun_validation/` — multi-seed held-out cross-validation

Output of `scripts/validate_multiseed.py`. The seed-averaged, divergent-rejected analogue of the
final validation: for each cross-validation fold (training set P_k), the filter uses **that
fold's** calibrated measured CVs (`results_single_sweep/fold_<k>/cv/cv_final.json`) and **its**
picked alphas (`results_single_sweep/picks.json`: α_obs=0.002, α_nsd=0.02), and is applied to the
**three held-out batches it never saw** (rotate scheme). Nothing is tuned here — CVs/alphas are
fixed inputs; this is the honest generalisation test.

Each held-out run is repeated over **10 seeds**; every run is archived; divergent replicates are
rejected (pool-relative peak-σ outlier rule, C=3× the across-run median, identical to the tuning
sweep) and resampled to 10 clean runs; calibration is reported on the **seed-averaged held-out
posterior**.

## Layout

```
fold_<k>/                              (k = training fold)
  pkl/heldout_<name>_seed_<s>.pkl      every run: mean_traj + std_traj + metrics (float32)
  agg/heldout_<name>.pkl               10 seeds stacked + seed-averaged mean/std + between-seed
                                       spread + 2σ bands + per-seed metrics + metrics-on-average
  figures/heldout_<name>.png           all-17-state grid on the seed-averaged held-out posterior
  seed_selection.json                  used/rejected seeds per held-out set
summary.json                           cross-fold seed-averaged held-out metrics
```

## Run (per training fold, resumable)

```bash
caffeinate -i ./.venv/bin/python scripts/validate_multiseed.py --folds P4
# then P3, P2, P1   (or drop --folds for all four in one long run)
```

Reported states (used for the NSD calibration summary) are UDP-Gal, UDP-Glc, UDP-GlcNAc, flagged
`*` in the figures. Heavy `*.pkl` are git-ignored; figures + JSON are tracked.
