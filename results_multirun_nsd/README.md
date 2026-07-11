# `results_multirun_nsd/` — multi-seed α_nsd sweep (seed-averaged calibration)

Output of `scripts/sweep_alpha_nsd_multiseed.py`. The EnKF is stochastic, so selecting α_nsd
from one realization is fragile. This re-runs the α_nsd sweep **N times with distinct seeds**
per (fold, α), records **every** run, and reports calibration on the **seed-averaged
posterior** — the legitimate stochastic estimate.

- Grid: `0.005, 0.0075, 0.01, 0.02, 0.03, 0.04, 0.05` (0.05 added vs the single-seed sweep).
- Fixed: each fold's calibrated CVs (from `results_single_sweep/fold_*/cv/cv_final.json`) and
  **α_obs = 0.002**. Seeds default to 42…51, so seed 42 reproduces the single-seed sweep
  bit-for-bit (verified).
- Selection focuses on the reliably-measured NSDs: **UDPGal, UDPGlc, UDPGlcNAc** (flagged `*`
  in the figures). The others have unreliable measurements and are not used to pick α_nsd.

## Layout

```
fold_<P>/
  pkl/alpha_<a>_seed_<s>.pkl   every single run: mean_traj + std_traj + metrics (float32)
  agg/alpha_<a>.pkl            all seeds stacked + seed-averaged mean/std + between-seed
                               spread + per-seed metrics + metrics-on-the-average
  figures/nsd_alpha_<a>.png    paper-style 7-NSD grid on the seed-averaged posterior
summary.pkl                    seed-averaged calibration (coverage / spread-skill / NRMSE)
                               for all folds × alphas, over the reported NSDs
```

Run (per fold, resumable): `caffeinate -i ./.venv/bin/python scripts/sweep_alpha_nsd_multiseed.py --n-runs 10 --datasets P4`

Heavy `*.pkl` are git-ignored (regenerable); figures are tracked.
