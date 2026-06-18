# EnKF Covariance Tuning Log

This document records all tuning decisions for the EnKF noise parameters, including
the reasoning, data, and diagnostics behind each choice. It serves as a reference
for the manuscript methods section and for future re-tuning.

**Final configuration:** tuned_v6 (git commit: `1adfc95`)
**Tuning dataset:** P4 (used exclusively for covariance calibration)
**Validation datasets:** P1, P2, P3 (not used during tuning)

**Diagnostic criteria:**
- Normalised Innovation Variance (NIV): target ~1.0 per measured state
- 2-sigma coverage: target ~95% (fraction of measurements within mean +/- 2*std)
- RMSE: should improve over open-loop model

---

## Tuning Procedure Summary

The following 6-step procedure was applied:

1. **R from data:** Measurement noise variance set from biological triplicate
   error bar variance (mean across P1-P4).

2. **P0 separated from Q:** Initial ensemble covariance set from measurement
   error bar variance (measured states) or process noise variance (unmeasured).

3. **Noise model selection:** Multiplicative (state-proportional) noise for
   measured extracellular states; additive noise for unmeasured states.

4. **Per-state CV tuning on P4:** Each measured state's CV tuned independently
   targeting NIV ~1.0 and 2-sigma coverage ~95%.

5. **IQR clipping:** Hard constraint on structurally unobservable states to
   prevent outlier ensemble divergence through nonlinear model propagation.

6. **Selective localization:** Only UDP-Gal excluded from Kalman update
   (near-zero concentration instability). All other NSDs receive cross-covariance
   corrections from measured states.

---

## 1. Measurement Noise R

### Decision
R set from **biological triplicate error bar variance**, averaged across P1-P4.

### Reasoning
The experimental error bars represent observed variability across three independent
biological replicate flasks. These capture genuine batch-to-batch variability plus
measurement noise, providing a conservative upper bound on measurement uncertainty.
The original R values were 5-870x too small (hand-picked without reference to data).

### Values

| State | R (variance) | sqrt(R) (std) | Old R | Ratio new/old |
|-------|-------------|---------------|-------|---------------|
| Xv | 6.928e+16 | 2.632e+08 | 2.2e+14 | 315x |
| mAb | 573.3 | 23.94 | 10 | 57x |
| Gal | 1.739 | 1.319 | 0.002 | 870x |
| Urd | 5.103e-3 | 0.0714 | 0.001 | 5x |
| Glc | 1.256 | 1.121 | 0.08 | 16x |
| Amm | 4.500e-2 | 0.2121 | 0.005 | 9x |
| Gln | 7.350e-3 | 0.0857 | 1e-4 | 74x |
| Lac | 0.2971 | 0.5451 | 0.4 | 0.74x |

---

## 2. Initial Ensemble Covariance P0

### Decision
P0 set **separately from per-step process noise Q**:
- Measured states: P0 = R (error bar variance)
- Unmeasured states: P0 = PROCESS_NOISE_VAR (additive noise variance)

### Reasoning
The initial ensemble should reflect uncertainty in the initial condition, not
per-step model error. Since x0 is taken from the first measurement, the
uncertainty is approximately the measurement error.

---

## 3. Process Noise Model

### Decision
- **Measured extracellular states** (Xv, mAb, Gal, Urd, Glc, Amm, Gln, Lac):
  Multiplicative noise `noise_i ~ N(0, (cv_i * x_i)^2)`
- **Unmeasured states** (Asn, Glu, all NSDs):
  Additive noise `noise_i ~ N(0, Q_ii)`

### Reasoning: why multiplicative for measured states
Fixed additive noise is physically wrong for concentrations near zero. When
Gal = 0 mM (before feeding), additive noise creates artificial zig-zag
oscillations and pushes ensemble members negative. Multiplicative noise
naturally gives zero perturbation when the state is zero and scales
uncertainty with concentration level.

### Reasoning: why additive for unmeasured states
Multiplicative noise on unmeasured states was tested (UDP-Gal, CV=0.001-0.004)
but caused ensemble blow-up because there is no measurement correction to
stabilise the ensemble. Additive noise with small fixed variance is more
stable for these states.

---

## 4. Per-State CV Tuning (Measured States)

### Method
CVs tuned iteratively on P4 over 4 rounds. Each round adjusted 1-3 states
based on NIV and 2-sigma coverage diagnostics.

CV values are per-step (dt=0.01h). They compound as `cv * sqrt(N_steps)`
between measurement updates (~2400 steps = 24h apart).

### Tuning iterations

**Round 0 (v2 baseline):**
```
Xv=0.004, mAb=0.005, Gal=0.003, Urd=0.006, Glc=0.008, Amm=0.005, Gln=0.003, Lac=0.007
```

| State | NIV | Cov 2σ | Action |
|-------|-----|--------|--------|
| Xv | 0.82 | 75% | Increase CV |
| mAb | 0.60 | 69% | Increase CV |
| Gal | 1.37 | 81% | Slight adjustment |
| Urd | 2.22 | 94% | Keep (coverage good) |
| Glc | 5.80 | 75% | Structural bias, accept |
| Amm | 1.39 | 75% | Increase CV |
| Gln | 0.81 | 94% | Keep |
| Lac | 1.61 | 100% | Keep |

**Round 1:** Xv 0.004→0.005, mAb 0.005→0.007, Amm 0.005→0.006
- Amm: NIV 1.39→1.09, Cov 75%→88% (improved)
- mAb: Cov 69%→88% (improved, NIV=0.36 over-dispersed)

**Round 2:** Xv 0.005→0.006, mAb 0.007→0.006
- Xv Cov stuck at 75% (specific outlier points)

**Round 3:** Xv→0.007, mAb→0.007, Gal 0.003→0.004
- Gal: NIV 1.37→1.01 (well-calibrated)

**Round 4:** Final candidates compared:
```
A: Xv=0.005, mAb=0.006 -> Xv(NIV=0.70,Cov=75%) mAb(NIV=0.45,Cov=75%)
B: Xv=0.006, mAb=0.007 -> Xv(NIV=0.58,Cov=75%) mAb(NIV=0.35,Cov=88%)
C: Xv=0.005, mAb=0.007 -> Xv(NIV=0.70,Cov=75%) mAb(NIV=0.36,Cov=88%)  <-- SELECTED
```

### Final values (Candidate C)

| State | CV | NIV | Cov 2σ | Notes |
|-------|-----|------|--------|-------|
| Xv | 0.005 | 0.70 | 75% | Cov limited by late-culture outlier points |
| mAb | 0.007 | 0.36 | 88% | Good coverage |
| Gal | 0.004 | 1.05 | 88% | Well-calibrated |
| Urd | 0.006 | 2.22 | 94% | High NIV from precise HPLC measurements |
| Glc | 0.008 | 5.81 | 75% | Structural model bias (accepted) |
| Amm | 0.006 | 1.09 | 88% | Well-calibrated |
| Gln | 0.003 | 0.80 | 94% | Well-calibrated |
| Lac | 0.007 | 1.60 | 100% | Good |

### Per-state reasoning

**Xv (CV=0.005):** Model systematically overpredicts viable cell density.
The 75% coverage is limited by specific late-culture timepoints where model
bias is largest. Increasing CV beyond 0.005 makes the filter over-dispersed
(NIV < 0.5) without improving coverage.

**mAb (CV=0.007):** The original R_mAb=10 was 57x too small, causing the
filter to over-correct and produce worse RMSE than the open-loop model
(Reviewer 3 point 5). With corrected R=573.3 and CV=0.007, mAb RMSE is
now consistently better than open-loop.

**Gal (CV=0.004):** Galactose is supplemented via bolus feeding. Multiplicative
noise naturally handles zero-feeding periods. NIV=1.05, nearly perfectly calibrated.

**Urd (CV=0.006):** Measured by HPLC with higher precision than the biological
triplicate error bars used for R. NIV=2.22 reflects this mismatch. Acceptable
trade-off; coverage is 94% (near target).

**Glc (CV=0.008):** Largest structural model bias (NIV=5.81). Model
systematically underpredicts glucose accumulation under galactose supplementation.
Accepted as model limitation rather than masked by inflated noise (Option A).
Directly addresses Reviewer 3 point 2.

**Amm (CV=0.006):** Well-calibrated. NIV=1.09.

**Gln (CV=0.003):** Model predicts glutamine accurately after correction.
Small CV sufficient. NIV=0.80.

**Lac (CV=0.007):** Some structural model rigidity. 100% coverage at 2σ
suggests bands could be slightly narrower, but kept for robustness.

---

## 5. Additive Process Noise for Unmeasured States

| State | Q (variance/step) | Reasoning |
|-------|-------------------|-----------|
| Asn | 1e-5 | Strongly coupled to growth, indirectly corrected |
| Glu | 4.8e-5 | Weakly coupled |
| UDPGal | 0 | All uncertainty from propagation (see Section 7) |
| UDPGalNAc | 1e-5 | Receives cross-covariance correction |
| UDPGlc | 6e-5 | Receives cross-covariance correction |
| UDPGlcNAc | 1e-5 | Receives cross-covariance correction (see Section 8) |
| GDPMan | 2e-7 | Minimal coupling, receives correction |
| GDPFuc | 2e-7 | Minimal coupling, receives correction |
| CMPNeu5Ac | 2e-4 | Receives cross-covariance correction |

---

## 6. IQR Clipping for Unobservable States

### Problem
Even with localization and Q=0, some NSD ensemble members diverged to extreme
values through nonlinear model propagation. Members with slightly different
upstream states (from multiplicative noise) produced wildly different NSD
synthesis rates. Near zero concentration, the NSD model lacks negative feedback
(consumption terms vanish via Monod kinetics), so divergent members grow
unchecked.

### Solution
After each predict step, for all states in `no_update_indices`, clip ensemble
members to `[0, Q3 + 5*IQR]` where Q1, Q3 are the 25th/75th percentiles and
IQR = Q3 - Q1. This removes extreme outliers while preserving the bulk of
the ensemble distribution.

### Justification
This is a standard constrained EnKF technique. The 5*IQR threshold is generous
enough to preserve meaningful spread but prevents the few extreme members that
inflate the reported std and create visual spikes in the uncertainty bands.

---

## 7. Selective Localization

### Decision
Only **UDP-Gal** is excluded from the Kalman update (`NO_UPDATE_STATES`).
All other NSDs receive cross-covariance corrections from measured states.

### Reasoning

**UDP-Gal must be localized:** Before galactose feeding (t < 96h), UDP-Gal is
near zero. The Kalman gain K[UDPGal, Urd] reached 16.07 at t=24h, meaning a
1 mM innovation in uridine pushed UDP-Gal by 16 mM. This spurious correction
arises from the large initial P0 for Urd relative to the small Urd value,
combined with the strong mechanistic coupling (uridine drives UDP-Gal synthesis
via r6_urd). At higher concentrations (after feeding), the consumption Monod
terms provide self-regulation, but near zero there is no restoring force.

**Other NSDs should NOT be localized:** Testing showed that allowing
cross-covariance corrections for UDP-GlcNAc, UDP-Glc, UDP-GalNAc, GDP-Man,
GDP-Fuc, and CMP-Neu5Ac improves their tracking. UDP-GlcNAc in particular
benefits significantly from its strong coupling to glutamine -- the
cross-covariance correction pulls it away from the overpredicting model
toward values consistent with corrected Gln measurements. The IQR clipping
prevents any divergence artifacts.

### Testing performed
- All 7 NSDs localized: UDP-GlcNAc stayed close to model (poor)
- 4 localized (UDPGal, GDPMan, GDPFuc, CMPNeu5Ac): UDP-GlcNAc improved
- Only UDPGal localized: similar results, simplest configuration (selected)
- CMP-Neu5Ac tuning tested: no effect on UDP-GlcNAc (weakly coupled)

---

## 8. UDP-GlcNAc Investigation

### Observation
UDP-GlcNAc is the most accurately estimated NSD, but the EnKF estimate still
does not fully reach the experimental measurements in late culture. The model
overpredicts UDP-GlcNAc (model=9.88 mM vs measurements=3-5 mM at t=240h);
the EnKF corrects to ~4.5 mM.

### Investigations performed
1. **Increase UDP-GlcNAc additive Q** (1e-5 to 1e-2): Higher Q made the
   estimate WORSE (moved toward model, not measurements). The additive noise
   amplifies the model's upward bias through nonlinear dynamics.
2. **Decrease UDP-GlcNAc additive Q** (1e-5 to 0): No effect. The trajectory
   is determined by the cross-covariance correction from Gln, not additive noise.
3. **Modify Gln CV and R** (various combinations): Modest improvement (~0.2 mM)
   by trusting Gln measurements more, but diminishing returns.
4. **CMP-Neu5Ac tuning**: No coupling effect on UDP-GlcNAc.

### Conclusion
The remaining gap between EnKF estimates and measurements is a structural
limitation of the NSD submodel, not a filter tuning issue. The cross-covariance
correction from Gln is already near-optimal. This is discussed in the manuscript
as a model fidelity limitation.

---

## 9. Version History

| Run | Key change | Git commit |
|-----|-----------|------------|
| run_v1 | Original notebook values, KQ=2e-6 | (pre-refactor) |
| tuned_v1 | Empirical R, additive per-state Q | `16c094f` |
| tuned_v2 | Multiplicative CV noise, clip bands at zero | `cb1e814` |
| tuned_v3 | Increased CVs (over-dispersed) | `3d0a3d2` |
| tuned_v4 | Per-state CV tuning (Candidate C) | `7be85c7` |
| tuned_v5 | IQR clipping + localization (all NSDs) | `7ad9e26` |
| **tuned_v6** | **Selective localization (UDPGal only)** | **`1adfc95`** |
