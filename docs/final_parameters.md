# Finalised EnKF noise parameters — per fold

Canonical record of every tuned noise value, per cross-validation fold. Each fold is tuned
**independently** on its own training set; what is shared is the **selection rule**, not the
value (see [`tuning_strategy.md`](tuning_strategy.md)). The complete parameter set for a fold is:
its 8 measured-state CVs **+** `alpha_obs` **+** `alpha_nsd`. `R` is a pooled instrument constant
(same for all folds).

Sources: `results_single_sweep/fold_*/cv/cv_final.json` (CVs, NIV),
`results_single_sweep/picks.json` (alphas), `nsd_enkf/config.py` (R, scales, adopted set).

---

## 1. Measurement-noise variance `R` (pooled over P1–P4, same for all folds)

Biological-triplicate variance; **from data, never fitted** (`R_jj = σ²_z,j`, diagonal).

| state | Xv | mAb | Gal | Urd | Glc | Amm | Gln | Lac |
|---|---|---|---|---|---|---|---|---|
| `R_jj` | 6.928e16 | 573.3 | 1.739 | 5.103e-3 | 1.256 | 4.5e-2 | 7.35e-3 | 0.2971 |

---

## 2. Measured-state CVs per fold (multiplicative noise, NIV = 1 fixed point)

Calibrated per fold to filter consistency (NIV → 1) with cap `CV_MAX = 6.0e-3`, floor `1e-4`.
Values are `×10⁻³`; **`*` = pinned at the cap** (structural bias — NIV cannot reach 1 within the
physical ceiling, left under-dispersed by design). NIV shown in parentheses.

| state | P1 | P2 | P3 | P4 | adopted (=P4) |
|---|---|---|---|---|---|
| Xv  | 6.00\* (1.26) | 6.00\* (1.01) | 6.00\* (1.06) | 5.55 (1.00) | 5.6e-3 |
| mAb | 6.00\* (1.09) | 6.00\* (1.08) | 5.70 (1.00) | 5.64 (1.00) | 5.6e-3 |
| Gal | 6.00\* (2.32) | 6.00\* (1.03) | 5.60 (1.00) | 5.16 (1.00) | 5.1e-3 |
| Urd | 6.00\* (3.02) | 6.00\* (1.26) | 6.00\* (4.77) | 6.00\* (2.30) | 6.0e-3\* |
| Glc | 6.00\* (7.92) | 6.00\* (8.08) | 6.00\* (8.37) | 6.00\* (8.21) | 6.0e-3\* |
| Amm | 6.00\* (1.02) | 6.00\* (1.42) | 6.00\* (1.30) | 6.00\* (1.15) | 6.0e-3\* |
| Gln | 2.79 (1.00) | 4.07 (1.01) | 3.84 (1.01) | 2.83 (1.00) | 2.8e-3 |
| Lac | 6.00\* (2.66) | 6.00\* (3.35) | 6.00\* (3.33) | 6.00\* (3.17) | 6.0e-3\* |

**Structural (capped in every fold): Glc (NIV≈8, severe glucose-model bias), Lac & Urd (NIV≈2–5),
Amm (mild).** Xv/mAb/Gal reach NIV=1 below the cap in the well-behaved folds; Gln always reaches
NIV=1 at a small CV. These are disclosed model-bias limitations, not tuning errors.

---

## 3. Additive-noise scalars per fold (unmeasured states)

`Q_ii = (α · scale_i)²`. Both α's are set by a shared rule and land on the **same value in every
fold** (the rule is shared; the coincidence is a result):

| fold | `alpha_obs` (Asn, Glu) | `alpha_nsd` (7 NSDs) |
|---|---|---|
| P1 | 0.002 | 0.02 |
| P2 | 0.002 | 0.02 |
| P3 | 0.002 | 0.02 |
| P4 | 0.002 | 0.02 |

- **`alpha_obs = 0.002`** — largest α whose Asn NRMSE stays within 25 % of its minimum (accuracy-
  guarded width; 0.002 ≈ 1.13× min, 0.004 ≈ 1.44×). Confirmed downstream-inert on the NSDs.
- **`alpha_nsd = 0.02`** — smallest α where the reported-NSD (UDP-Gal, UDP-Glc, UDP-GlcNAc) mean
  spread-skill reaches ≈1 (≥0.95) on the seed-averaged posterior (10 clean replicates, divergent
  runs rejected). NRMSE knee at 0.03 (0.76→1.09) guards against over-widening.

### Characteristic scales `s_i` (median level, mM) — fixed, shared

| Asn | Glu | UDP-Gal | UDP-GalNAc | UDP-Glc | UDP-GlcNAc | GDP-Man | GDP-Fuc | CMP-Neu5Ac |
|---|---|---|---|---|---|---|---|---|
| 5.13 | 3.735 | 0.5198 | 0.1615 | 0.5224 | 0.9022 | 0.5799 | 0.0543 | 1.112 |

---

## 4. Fixed scalars (all folds)

| quantity | value |
|---|---|
| ensemble size `n` | 100 |
| time step `Δt` | 0.01 h |
| horizon `T` | 288 h (28 800 steps) |
| inter-measurement interval | ≈ 2400 steps |
| CV bounds `[c_min, CV_MAX]` | [1e-4, 6e-3] |
| NIV tolerance | 0.15 |
| NSD IQR clip | `[1e-12, Q3 + 5·IQR]` (7 NSDs; Glc not clipped) |
| divergence-rejection threshold | peak σ > 3× across-run median (per unmeasured state) |

---

## 5. Adopted production config

`nsd_enkf/config.py` holds the **P4-fold** calibration as the canonical single-filter values
(`PROCESS_NOISE_CV`, `PROCESS_NOISE_ALPHA_OBS = 0.002`, `PROCESS_NOISE_ALPHA = 0.02`). The other
folds exist to demonstrate the rule generalises (full-fold cross-validation); no held-out batch
influences the filter applied to it.
