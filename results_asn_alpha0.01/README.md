# `results_asn_alpha0.01/` — α_obs downstream-inertness test

Output of `scripts/test_asn_alpha_on_nsd.py`. A controlled test of the hypothesis that a
**looser Asn/Glu (larger α_obs)** — being *upstream* of the NSD pathway — might improve the
downstream NSD estimates.

Runs the EnKF at **α_obs = 0.01** (vs the adopted 0.002) on P1–P4, holding each fold's
calibrated CVs and **α_nsd = 0.02** fixed, and compares NSD metrics against the matching
baseline pkl in `results_single_sweep/fold_*/alpha_nsd/pkl/alpha_0.02.pkl` (identical except
α_obs = 0.002). So the only thing that differs between test and baseline is α_obs.

## Result

NSD accuracy is **flat** (mean NSD NRMSE unchanged) and the coverage change is small and
**inconsistent in sign** across folds → Asn/Glu process noise does **not** propagate materially
into the NSD estimates. The two α tiers are **empirically independent**, so α_obs is chosen on
the Asn band alone (see `docs/tuning_strategy.md` §4).

## Layout

```
pkl/P<n>.pkl            full 17-state mean/std trajectory + 2σ bands + metrics (α_obs=0.01)
figures/nsd_P<n>.png    paper-style 7-NSD grid at α_obs=0.01
comparison_metrics.pkl  test (0.01) vs baseline (0.002) metrics, all folds
```

Heavy `*.pkl` are git-ignored (regenerable); figures are tracked.
