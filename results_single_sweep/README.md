# `results_single_sweep/` — per-fold cross-validation tuning (single-seed sweep)

Output of `scripts/04_cross_validate.py --stage sweep`. This is the **main systematic-tuning
result**: for each dataset **P1–P4 taken in turn as the training set** (a "fold"), the filter
is tuned *entirely* on that fold and everything needed to pick its parameters is saved.
Single stochastic realization (seed 42); the seed-averaged re-run lives in
`results_multirun_nsd/`.

## Layout

```
fold_<P>/
  cv/          measured-state CV calibration to NIV=1: cv_final.json (calibrated CVs +
               per-iteration NIV history + capped/floored flags), cv_trajectory.pkl,
               figures/niv_convergence.png
  alpha_obs/   α_obs sweep (Asn/Glu), α_nsd held at reference:
               pkl/alpha_<a>.pkl (full 17-state mean/std + 2σ bands + metrics), figures/
  alpha_nsd/   α_nsd sweep (7 NSDs), α_obs held at reference: pkl/… + figures/
  validation/  (created by --stage validate) held-out all-state bands + metrics + grids
picks.json     adopted (α_obs, α_nsd) per fold + the selection RULES that produced them
```

## Status of the picks (see `picks.json` + `docs/tuning_strategy.md`)

- **α_obs = 0.002** (all folds) — settled. Rule: largest α within 25 % of the minimum Asn
  NRMSE; confirmed downstream-inert on the NSDs (`results_asn_alpha0.01/`).
- **α_nsd** — pending; being re-derived on the seed-averaged sweep (`results_multirun_nsd/`),
  restricted to the reliably-measured NSDs (UDPGal, UDPGlc, UDPGlcNAc).

Heavy `*.pkl` trajectories are git-ignored (regenerable); figures + JSON are tracked.
