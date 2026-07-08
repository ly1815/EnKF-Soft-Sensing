# EnKF Covariance Tuning Log

This document records all tuning decisions for the EnKF noise parameters, including
the reasoning, data, and diagnostics behind each choice. It serves as a reference
for the manuscript methods section and for future re-tuning.

> For the clean, self-contained **procedure** (ordered stages, criteria, dependencies,
> reproduce commands), see [`tuning_strategy.md`](tuning_strategy.md). This file is the
> chronological *log* (what was tried, superseded, and why); that file is the *method*.

**Current configuration (as of 2026-07-01):**
- **Measured-state process noise:** per-state multiplicative CVs (tuned_v6 values,
  git `1adfc95`), still adopted in `config.py`. An automated re-calibration
  (`scripts/tune_cv.py`) is in progress — see Section 12.
- **Unmeasured-state process noise:** a single universal two-stage additive scalar —
  `PROCESS_NOISE_ALPHA = 0.01` (7 NSDs) and `PROCESS_NOISE_ALPHA_OBS = 0.002`
  (Asn, Glu). This "Option B" scheme supersedes the hand-picked per-state Q of
  Section 5 — see Section 10.
- **Localization:** none (`NO_UPDATE_STATES = []`); UDP-Gal's near-zero instability is
  now handled by IQR clipping alone. This supersedes Section 7 — see Section 11.

**Tuning dataset:** P4 (used exclusively for covariance calibration)
**Validation datasets:** P1, P2, P3 (not used during tuning)

**Diagnostic criteria:**
- Normalised Innovation Variance (NIV): target ~1.0 per measured state (now driven to
  a fixed point automatically — Section 12)
- 2-sigma coverage: target ~95% (fraction of measurements within mean +/- 2*std)
- NSD NRMSE (RMSE / mean measurement): selection metric for the universal alpha (Section 10)
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

> **Superseded by Section 10.** The hand-picked per-state additive Q below was
> replaced by a single universal two-stage alpha (Option B). Retained here as the
> historical record for tuned_v6.

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

> **Superseded by Section 11.** Localization was subsequently removed entirely
> (`NO_UPDATE_STATES = []`); UDP-Gal is now stabilised by IQR clipping alone while
> retaining its uridine cross-covariance coupling. Retained here as the historical
> record for tuned_v6.

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
| tuned_v6 | Selective localization (UDPGal only) | `1adfc95` |
| option_b | Universal two-stage additive alpha (0.01 NSD / 0.001 Asn,Glu); localization removed (clip only) | `4e2bcf8` |
| cv_auto | Automated NIV=1 CV fixed-point (`scripts/tune_cv.py`); **in progress, not yet adopted** | `results/cv_tuning/` |

---

## 10. Universal Two-Stage Additive Process Noise (Option B)

**Supersedes Section 5.**

### Decision
Unmeasured states no longer use individually hand-picked additive variances. Instead a
single universal scalar sets each state's per-step additive variance as a fixed fraction
of that state's characteristic magnitude:

    Q_ii = (alpha * scale_i)^2

where `scale_i` is the fixed median magnitude of state i (config `PROCESS_NOISE_SCALE`)
and `alpha` is the one tunable knob shared across all unmeasured states. This is the
standard nondimensionalisation of process noise (Q proportional to a climatological
covariance): one knob sets the overall magnitude while the relative uncertainty
structure across states stays fixed.

### Why a magnitude (median) rather than a spread
Using each state's characteristic level (median) avoids the "flat-model" failure:
states whose open-loop trajectory is nearly constant (GDP-Man, CMP-Neu5Ac) still have a
well-defined scale. For the measured NSDs the scale is the median of the pooled P1-P4
measurements; Glu (never measured) uses its open-loop median. The scale is therefore
independent of any single validation dataset.

### Two-stage alpha
- **7 structurally-unobservable NSDs** (UDPGal, UDPGalNAc, UDPGlc, UDPGlcNAc, GDPMan,
  GDPFuc, CMPNeu5Ac): `PROCESS_NOISE_ALPHA = 0.01`.
- **Observable unmeasured states** (Asn, Glu): `PROCESS_NOISE_ALPHA_OBS = 0.002`. These
  are strongly coupled to the measured states and well constrained by cross-covariance
  corrections, so they need far less injected noise. A single-tier alpha (same value
  everywhere) put too much noise on Asn (git `4e2bcf8`, "one tier noise too big for
  asn"), motivating the split. ALPHA_OBS is calibrated *separately* from the NSD alpha via
  an Asn-only sweep (`scripts/tune_alpha_asn.py`) — the NSD pathway is downstream of Asn (no
  feedback), so Asn's calibration is independent of the NSD alpha. Asn and Glu share the
  swept value; only Asn is scored (Glu is never measured). P4 sweep (0.001–0.01), Asn:
  0.001 → NRMSE 0.34 / cov 24% / ss 0.22; **0.002 → NRMSE 0.40 / cov 59% / ss 0.37 (adopted)**;
  0.01 → NRMSE 0.43 / cov 88% / ss 1.00. 0.002 is a bioprocessing-judgement pick — a tight,
  physically sensible band accepting under-coverage (as with Glc). Excluded from the NSD sweep.

### Scale values (config `PROCESS_NOISE_SCALE`, mM)

| State | scale (median) | alpha stage |
|-------|----------------|-------------|
| Asn | 5.13 | OBS (0.002) |
| Glu | 3.735 (open-loop median) | OBS (0.002) |
| UDPGal | 0.5198 | NSD (0.01) |
| UDPGalNAc | 0.1615 | NSD (0.01) |
| UDPGlc | 0.5224 | NSD (0.01) |
| UDPGlcNAc | 0.9022 | NSD (0.01) |
| GDPMan | 0.5799 | NSD (0.01) |
| GDPFuc | 0.0543 | NSD (0.01) |
| CMPNeu5Ac | 1.112 | NSD (0.01) |

### Calibration
`alpha` is per-step (dt=0.01h); accumulated process-noise std over an N-step interval
grows ~ alpha*scale*sqrt(N), so a per-24h relative uncertainty beta corresponds to
alpha = beta / sqrt(2400). The NSD alpha is selected on P4 by minimising the mean NSD
NRMSE (RMSE / mean measurement — dimensionless, so states of very different magnitude
are comparable), then cross-validated on P1-P3. Selected value: **alpha = 0.01** (git
`4e2bcf8`). Scripts: `scripts/tune_alpha.py` (sweep + cross-validation) and
`scripts/run_option_b.py` (sweep + save mean/std bands for all 17 states; run stored in
`results/ob_wide/`).

### Re-sweep on the adopted CVs (2026-07-08)
The NSD alpha was **re-swept on the adopted CVs** (cap 0.006) with `scripts/tune_alpha_nsd.py`
(P4, grid 0.005–0.04), so it no longer inherits its value from the superseded tuned_v6 CVs.
**alpha = 0.01 is confirmed.** Aggregate over the 7 NSDs: mean NRMSE 1.28→1.33→1.42 across
0.005/0.01/0.02, so 0.01 is within a hair of the accuracy optimum while coverage rises
46%→60% and median spread-skill 0.20→0.43 (0.01 balances accuracy vs band quality; UDP-Glc
lands perfectly calibrated, ss≈1.06). The aggregate NRMSE is dominated by **UDP-GalNAc**
(NRMSE≈4.9, cov 27%) — a structural overprediction the model cannot escape at any alpha; the
other six NSDs are all NRMSE<1.3. **GDP-Man** is similarly structural (model near-flat,
coverage ~9% at every alpha). Both are disclosed as model-fidelity limitations, not tuning
failures. Remaining: cross-validate 0.01 on P1–P3 and save the final all-state bands
(`run_option_b.py --fixed-alpha 0.01`).

---

## 11. Localization Removed (Clipping Only)

**Supersedes Section 7.**

### Decision
`NO_UPDATE_STATES = []` — no state is excluded from the Kalman update. Every NSD,
including UDP-Gal, now receives cross-covariance corrections from the measured states.

### Reasoning
In tuned_v6, UDP-Gal was localized to avoid its near-zero instability (a spurious
`K[UDPGal, Urd] ~ 16` before galactose feeding). That instability is now bounded
directly by IQR clipping (Section 6): after each predict and update, all 7 NSDs
(`CLIP_STATES`) are clipped to `[1e-12, Q3 + 5*IQR]`. Clipping removes the few divergent
outlier members without discarding UDP-Gal's mechanistic coupling to uridine, so UDP-Gal
keeps a useful measurement-informed correction instead of evolving open-loop.

### Configuration
- `NO_UPDATE_STATES = []`
- `CLIP_STATES = [UDPGal, UDPGalNAc, UDPGlc, UDPGlcNAc, GDPMan, GDPFuc, CMPNeu5Ac]`

---

## 12. Automated CV Calibration (in progress)

The manual 4-round per-state CV tuning of Section 4 has been replaced by an automated
fixed-point calibration, `scripts/tune_cv.py`. Each measured state's per-step CV is
driven to filter consistency (NIV = 1) by the fixed-point update

    CV_j  <-  CV_j * sqrt(NIV_j)

which converges because the innovation variance `S_jj` (hence NIV_j) is monotone
decreasing in CV_j. All 8 NIVs come from a single EnKF pass, so a handful of iterations
suffice. CVs are clamped to `[CV_MIN, CV_MAX]`; a state left under-dispersed at the cap is
flagged as structural model bias (raising CV there would only mask the bias with
inflated noise).

### Current checkpoint (`results/cv_tuning/tune_cv_checkpoint.json`, iteration 8)

| State | CV | NIV | Status |
|-------|------|------|--------|
| Xv | 0.00114 | 1.09 | converged |
| mAb | 0.00584 | 1.02 | converged |
| Gal | 0.00259 | 0.94 | converged |
| Urd | 0.01913 | 1.01 | converged |
| Glc | 0.05 | 1.36 | **capped — structural bias** |
| Amm | 0.00530 | 1.03 | converged |
| Gln | 0.0001 | 0.38 | **floored — over-dispersed** |
| Lac | 0.01056 | 0.97 | converged |

- **Glc** stays under-dispersed even at the CV cap (NIV 1.36), consistent with the known
  structural glucose bias (Section 4). Left at the cap by design.
- **Gln** is over-dispersed at the CV floor. With CV already negligible, `S ~ R`, so the
  residual over-dispersion is driven by `R_Gln` (7.35e-3), not by process noise, and
  cannot be removed by lowering CV further. This flags `R_Gln` (pooled P1-P4 mean) as
  likely overestimated for P4.

> **Note:** the iteration-8 checkpoint above predates the convergence fix below and will
> be regenerated on the next run.

### Termination (general, no per-state special-casing)
The fixed point iterates until all `|NIV-1| < tol`, or a maximum number of iterations
(`--iters`) is reached. There is **no bound-based exclusion** from the convergence test:
a state that cannot reach NIV=1 (Glc capped at `CV_MAX` by structural glucose bias; Gln
floored at `CV_MIN`, R-dominated) simply keeps the loop running to `--iters`, at which
point it stays pinned at its bound. The extra iterations leave the pinned CV/NIV
unchanged, so the final CVs are identical to stopping early — running to the cap is the
expected, honest behaviour when a state is genuinely un-tunable to NIV=1. The
`capped`/`floored` flags are recorded for reporting (tables, plots, JSON) only.

### CV cap: 0.05 → 0.02 → 0.01 → 0.006 (physical band-plausibility ceiling)
`CV_MAX` is a physical ceiling on the per-interval uncertainty (`CV·√2400`), reframed to
bound each state's band to a physically plausible range rather than to enforce consistency:
- **0.05:** Glc pinned there; its multiplicative noise compounded to `0.05·√2400 ≈ ×11`,
  blowing the Glc band up to ~1000 mM against tens-of-mM data.
- **0.02** (~100%/24h): still ~200 mM for Glc — physically impossible for a ~144 mM feed.
- **0.01** (~50%/24h): ~80 mM for Glc — still too wide.
- **0.006** (~30%/24h): **adopted.** Sits just above mAb's NIV=1 CV (~0.0058) — the tightest
  ceiling that still caps only the three genuinely bias-limited states (Glc, Urd ≈ 0.019,
  Lac ≈ 0.011) and leaves Xv, mAb, Gal, Amm at NIV ≈ 1 (Gln floors, R-driven). Those three
  are left **under-dispersed (NIV > 1)** by design — a bioprocessing decision that a
  physically bounded band beats a statistically consistent but physically impossible one.
- IQR-clipping glucose's upper tail was also trialled (Glc added to `CLIP_STATES`) but did
  **not** help: the band width is bulk multiplicative spread, not outliers, so trimming the
  tail barely moved it. Reverted — the tighter cap bounds Glc, not clipping.

One global cap, no per-state special-casing.

### Adopted CVs (2026-07-08, cap 0.006)
The automated CVs are now in `config.PROCESS_NOISE_CV`, replacing the tuned_v6 hand values:

| State | CV | NIV | Status |
|-------|------|------|--------|
| Xv  | 0.0056 | 1.00 | well-calibrated |
| mAb | 0.0056 | 1.00 | well-calibrated |
| Gal | 0.0051 | 1.00 | well-calibrated |
| Urd | 0.0060 | 2.29 | capped (structural bias) |
| Glc | 0.0060 | 8.21 | capped (structural bias) |
| Amm | 0.0060 | 1.16 | capped (structural bias) |
| Gln | 0.0028 | 1.00 | well-calibrated (no floor at cap 0.006) |
| Lac | 0.0060 | 3.17 | capped (structural bias) |

Stage 4 (NSD alpha) will be re-swept on these CVs. Asn's observable alpha is examined
separately first (`scripts/tune_alpha_asn.py`) — the NSD pathway is downstream of Asn, so
Asn's calibration is independent of the NSD alpha.

### Artifacts (per run, `results/<run>/`)
`tune_cv.py` now saves, at the final CVs: `pkl/cv_tuned_<DS>.pkl` (all-17-state mean +
std trajectories, ±1σ/±2σ bands, and the open-loop model trajectory),
`figures/cv_niv_convergence_<DS>.png` (NIV→1 per iteration), and
`figures/cv_tuned_states_<DS>.png` (all-state grid: EnKF mean + bands + model + meas).

### Status — adopting the automated pipeline
Decision (2026-07-07): the tuning pipeline must be **reproducible from data end-to-end**
for the manuscript, with no dependency on the hand-tuned tuned_v6 values. The automated
NIV=1 calibration replaces the manual 4-round tuning as the measured-state procedure.
NIV=1 is the principled target for an *assimilated* state; where manual tuning deviated
from it to protect coverage (e.g. Urd held at CV=0.006 despite NIV=2.22, vs auto 0.019),
the automated value is the more defensible one. 2σ coverage is still **reported** as a
validation diagnostic (not a selection criterion), and the `R_Gln` over-estimate is
disclosed as a transparent limitation rather than fudged. Because changing the measured
CVs changes the cross-covariance flow into the NSDs, **the universal NSD alpha must be
re-swept on the adopted CVs** (Section 10) — the current alpha=0.01 was optimised against
tuned_v6 and is not carried over unchanged.
