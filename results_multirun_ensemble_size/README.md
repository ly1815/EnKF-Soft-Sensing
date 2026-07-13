# `results_multirun_ensemble_size/` — multi-seed ensemble-size sensitivity

Output of `scripts/05_ensemble_size.py`. Directly answers **Reviewer 2.1** (is `N = 100`
justified? sensitivity, instability at small `N`?) and **Reviewer 3.4** (does the ensemble keep
meaningful, calibrated spread — not just an accurate mean?).

For each ensemble size `N ∈ {25, 50, 100, 150, 200}`, **10 independent EnKF passes** (distinct
seeds, `seed = 42 + i`) are run on **P4** using the **adopted production filter** — the P4-fold
calibration in `nsd_enkf/config.py` (per-state multiplicative CVs, `α_obs = 0.002`,
`α_nsd = 0.02`). Nothing is tuned here; the filter config is a fixed input.

Divergent replicates are rejected with the **same pool-relative peak-σ rule as the tuning and
validation sweeps** (a pass is rejected if any unmeasured state's peak σ exceeds `C = 3×` the
across-run median peak, evaluated **per size**), then resampled to 10 clean runs. The per-size
divergence count is a reported result, not hidden — small `N` is where blow-ups cluster, which is
itself the R2.1 stability point.

## Layout

```
ensemble_N<N>.pkl                {seed: run} cache — EVERY pass drawn (incl. rejected);
                                 each run carries its downsampled (×20) mean + std trajectory,
                                 innovations + S at update times, and all metrics
all_trajectories.pkl             combined archive — mean + spread (std) of every clean run of
                                 every size (shape (M, 17) per run), + filter-config provenance
ensemble_sensitivity_summary.pkl per-size mean ± std metrics + used/rejected/drawn counts
seed_selection.json              used / rejected seeds per size (+ reject_mult)
ensemble_size_sensitivity.png    6 panels: (a) measured NRMSE, (b) NIS→1, (c) 2σ coverage,
                                 (d) per-NSD NRMSE, (e) NSD spread-skill→1, (f) wall-clock/pass
```

Heavy `*.pkl` are git-ignored (regenerable); `seed_selection.json` + the figure are tracked.

## Metrics per size (mean ± std across the clean runs)

- **measured** — normalised RMSE, NIS `= ⟨d²/S⟩` (ideal 1), 2σ coverage %
- **NSD (7)** — normalised RMSE, 2σ coverage %, spread-skill `= ⟨σ⟩/RMSE` (ideal 1)
- **Asn** — normalised RMSE
- **cost** — wall-clock seconds per pass

## Run (resumable; per-pass cache, kill costs ≤ 1 pass)

```bash
caffeinate -i ./.venv/bin/python scripts/05_ensemble_size.py --n-runs 10 --resume
```

`--no-reject` aggregates all seeds without the divergence gate; `--traj-down 1` stores
full-resolution trajectories instead of ×20 downsampled. Runtime ≈ 7 h for the full 5-size,
10-seed sweep (~linear in `N`; small-`N` sizes may draw a few extra seeds while resampling).
