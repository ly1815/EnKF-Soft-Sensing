# Systematic EnKF Tuning Strategy

**Purpose.** This document is the authoritative, self-contained description of *how* the
EnKF soft sensor is tuned — the ordered procedure, the criteria, and the reasoning behind
each choice. It is written to be manuscript-ready and reproducible **end-to-end from data**,
with no dependency on any hand-tuned or frozen intermediate result.

The chronological decision history (what was tried, superseded, and why) lives in
[`tuning_log.md`](tuning_log.md). This file is the *clean procedure*; that file is the *log*.

- **Tuning dataset:** P4 (used exclusively for calibration).
- **Validation datasets:** P1, P2, P3 (never used during tuning).

---

## 0. At a glance

| Quantity | How it is set | Metric / target | Source | Tuned? |
|---|---|---|---|---|
| **R** (measurement noise) | Biological triplicate error-bar variance, pooled P1–P4 | — (from data) | `config.MEASUREMENT_NOISE_VAR` | No — fixed from data |
| **P0** (initial ensemble cov) | Measured = R; unmeasured = process-noise variance | — | `config.INITIAL_COV_OVERRIDE` | No — derived |
| **Noise model** | Measured → multiplicative CV; unmeasured → additive | — (structural) | `enkf.py` | No — by design |
| **Measured CVs** (8) | Fixed-point `CV ← CV·√NIV` to filter consistency | **NIV = 1** | `scripts/01_tune_cv.py` | **Yes — automated** |
| **NSD α** (1 scalar, 7 states) | Sweep; minimise mean NSD NRMSE + band inspection | **min NRMSE**; band validated by coverage/spread-skill | `scripts/03_tune_alpha_nsd.py` (+ `04_cross_validate.py`) | **Yes — swept** |
| **α_OBS** (Asn, Glu) | Asn-only sweep on P4 (Asn & Glu share it) | Asn NRMSE + coverage | `config.PROCESS_NOISE_ALPHA_OBS` | Yes — Asn sweep |
| **IQR clipping** | 7 NSDs clipped to `[1e-12, Q3+5·IQR]` after predict/update | — (stability) | `config.CLIP_STATES` | No — structural |
| **Ensemble size N** | Sensitivity sweep on final config | NIS, coverage, spread-skill vs N | `scripts/05_ensemble_size.py` | Verification |

**Two tunable knobs, two different criteria — and the asymmetry is the whole point:**
- **Measured states are *assimilated*** → a *consistency* target (NIV = 1) is both available
  and correct. This is the textbook definition of a well-tuned filter.
- **NSD states are *not assimilated*** (structurally unobservable; no rows in H) → there is
  **no innovation to make consistent**, so NIV is undefined for them. α is instead selected
  by **accuracy (NRMSE)** against held-out NSD measurements, and the resulting uncertainty
  **band is validated** (not selected) by coverage and spread-skill.

---

## 1. State vector and observability structure

17 states, in three observability classes — this structure dictates every tuning choice:

| Class | States | In Kalman update? | Noise model | Tuned via |
|---|---|---|---|---|
| **Measured** (8) | Xv, mAb, Gal, Urd, Glc, Amm, Gln, Lac | Yes (H rows) | Multiplicative CV | NIV = 1 |
| **Observable-unmeasured** (2) | Asn, Glu | No | Additive (α_OBS) | Asn sweep |
| **NSDs** (7) | UDPGal, UDPGalNAc, UDPGlc, UDPGlcNAc, GDPMan, GDPFuc, CMPNeu5Ac | No | Additive (α) | NRMSE |

Asn/Glu are unmeasured in the update but **strongly coupled** to the measured states, so
they are well constrained by cross-covariance corrections and need only a small, fixed
process noise (α_OBS = 0.002). The NSDs are structurally unobservable and rely on the model
plus propagated corrections; their process noise (α = the swept knob) is the main handle.

---

## 2. Stage 0 — Measurement noise R (from data, fixed)

R is set from the **biological triplicate error-bar variance**, averaged across P1–P4.
These error bars capture batch-to-batch biological variability plus measurement noise — a
conservative, physically grounded, and dataset-independent estimate.

**R is never tuned to filter diagnostics.** Its independence from the filter is what makes
the methodology defensible. Where the consistency calibration later flags an R value as
conservative (see §8, Gln), that is *disclosed as a diagnostic*, not fixed by fitting R to
the filter's own residuals (which would be circular and would overfit R to the tuning set).

## 3. Stage 1 — Initial ensemble covariance P0 (derived, fixed)

P0 is set **separately** from the per-step process noise Q, so the ensemble starts with
meaningful spread reflecting initial-condition uncertainty:
- Measured states: `P0 = R` (initial condition ≈ first measurement, so uncertainty ≈ R).
- Unmeasured states: `P0 = ` process-noise variance.

## 4. Stage 2 — Noise-model selection (structural, fixed)

- **Measured extracellular states → multiplicative noise**, `noise_i ~ N(0, (cv_i·x_i)²)`.
  Fixed additive noise is physically wrong near zero concentration (e.g. Gal before feeding),
  producing artificial oscillations and negative members. Multiplicative noise gives zero
  perturbation at zero and scales with concentration.
- **Unmeasured states → additive noise**, `noise_i ~ N(0, Q_ii)`. Multiplicative noise on
  unmeasured states blows up: with no measurement correction there is no restoring force.

The unmeasured additive variance is nondimensionalised through a single scalar:

$$Q_{ii} = (\alpha \cdot \text{scale}_i)^2$$

where `scale_i` is the fixed **median magnitude** of state *i* (measured NSD median pooled
over P1–P4; Glu uses its open-loop median) and α is the one tunable knob. Using a magnitude
(median) rather than a spread avoids the "flat-model" failure — states whose open-loop
trajectory is nearly constant still have a well-defined scale. This freezes the *relative*
uncertainty structure across states while one knob sets the *overall* magnitude (Q
proportional to a fixed climatological covariance — standard process-noise scaling).

## 5. Stage 3 — Measured-state CVs → filter consistency (NIV = 1)

**Script:** `scripts/01_tune_cv.py` (on P4). **Automated, replaces the historical manual tuning.**

For each measured state the per-step CV is driven to the consistency target NIV = 1, where

$$\text{NIV}_j = \text{mean}\!\left(\frac{d_j^2}{S_{jj}}\right),\quad d = z - H\hat x,\quad S = P_{zz} + R.$$

NIV > 1 ⇒ under-dispersed (need more spread); NIV < 1 ⇒ over-dispersed. Because `S_jj`
grows like `CV_j²`, NIV is monotone decreasing in CV, so the fixed-point iteration

$$CV_j \leftarrow CV_j \cdot \sqrt{\text{NIV}_j}$$

contracts to NIV = 1. All 8 NIVs come from one EnKF pass, so a handful of passes suffice.

**Termination (general, no per-state special-casing):** iterate until all `|NIV−1| < tol`
(default 0.15) or a max-iteration safeguard (`--iters`) is reached. CVs are clamped to
`[CV_MIN, CV_MAX] = [1e-4, 0.006]`. A state that would leave that range stays pinned at the
bound and is *flagged* (capped / floored) for reporting only — the flags do not gate
convergence. A pinned state keeps the loop running to `--iters`; the extra iterations leave
its CV/NIV unchanged, so the result is identical to an early stop.

**`CV_MAX = 0.006` is a physical band-plausibility ceiling, not a tuned value, and it can
override the NIV=1 target.** Per-step CV compounds over an N-step interval as `CV·√N`, so
`0.006·√2400 ≈ 0.29` — a ~30% relative model error accumulated over a 24 h measurement
interval, which keeps each state's uncertainty band within physically plausible metabolite
ranges. NIV = 1 is the primary target, but where consistency would demand a larger CV —
driven by structural model bias, chiefly **glucose** (its NIV=1 band reached ~1000 mM
against a ~144 mM feed, i.e. physically impossible) — the state is pinned at the ceiling and
left **under-dispersed (NIV > 1)**. We deliberately prefer a physically bounded band to a
statistically consistent but physically impossible one — a bioprocessing constraint that
trumps the statistical one. The value 0.006 sits just above mAb's NIV=1 CV (~0.0058), the
tightest ceiling that still caps only the three genuinely bias-limited states — **Glc, Urd
(NIV=1 CV ≈ 0.019) and Lac (≈ 0.011)** — while leaving Xv, mAb, Gal, Amm at NIV ≈ 1 (Gln
floors, R-driven). (The ceiling was lowered through 0.05 → 0.02 → 0.01 → 0.006 as each still
left glucose's band physically implausible: ~1000, ~200, ~80 mM respectively. IQR-clipping
glucose's upper tail was also trialled but did not help — the width is bulk multiplicative
spread, not outliers — so the tighter cap, not clipping, bounds it.)

**Adoption:** the resulting CVs are written into `config.PROCESS_NOISE_CV`, replacing any
prior hand-tuned values. NIV = 1 is the principled target for an assimilated state; where
earlier manual tuning deviated from it to protect coverage, the automated value is the more
defensible one. 2σ coverage is still **reported** as a validation diagnostic, not used to
override the NIV target.

## 6. Stage 4 — NSD process noise α → accuracy (NRMSE), band validated

**Scripts:** `scripts/03_tune_alpha_nsd.py` (NSD α sweep + per-α band inspection), `scripts/04_cross_validate.py` (full-fold cross-validation). *(The legacy `tune_alpha.py` / `run_option_b.py` in `scripts/legacy/` are the superseded originals.)*
(sweep + save all-state bands/figures). **Runs on the Stage-3 CVs (must be re-swept if the
CVs change — see §9).**

**Selection metric — why NRMSE and not consistency.** The NSDs are never assimilated, so no
innovation exists and NIV is undefined for them. Their *mean* estimate is driven by
cross-covariance corrections propagated from the measured extracellular states; α controls
how open that channel is (too small → ensemble collapses, correction is weak, estimate
sticks near the biased open-loop model; too large → injected noise swamps the propagated
signal). The **NRMSE-minimising α is precisely the one that best transmits measured-state
information into the NSD means** — which is the operational goal. Selection metric:

$$\text{NRMSE}_i = \frac{\text{RMSE}_i}{\text{mean}(\text{NSD}_{\text{meas},i})}\ \text{(dimensionless)},\qquad \alpha^\* = \arg\min_\alpha \frac{1}{7}\sum_{i} \text{NRMSE}_i \ \text{on P4.}$$

**Two-stage α:** the NSD sweep here applies only to the 7 NSDs. The observable Asn/Glu share
a smaller α_OBS, calibrated *separately* on P4 by an Asn-only sweep
(`scripts/02_tune_alpha_asn.py`) — the NSD pathway is downstream of Asn (no feedback), so Asn's
calibration is independent of the NSD α. Asn and Glu take the same swept value; only Asn is
scored (Glu is never measured). Selected **α_OBS = 0.002** (P4: Asn NRMSE 0.40, 2σ coverage
59%, spread-skill 0.37) — a bioprocessing-judgement pick favouring a tight, physically
sensible band over full statistical coverage (as with Glc). Asn/Glu are excluded from the
NSD sweep.

**Band validation (report, do not re-select).** NRMSE scores only the *mean*; the ±2σ
*band* must be separately validated. At the selected α, report for the NSDs on P4 **and**
the P1–P3 hold-outs:
- **2σ coverage** (target ≈ 95%),
- **spread-skill** = mean(std)/RMSE (target ≈ 1).

These confirm the band is honest. If coverage is degenerate, that is disclosed as a
limitation — α is *not* re-selected on it. (Optional reviewer defence: a variance
attribution — what fraction of NSD ensemble variance is propagated-from-measured vs
injected-by-α — quantifies the claim that "NSD uncertainty comes from the measured states.")

**Cross-validation:** the α selected on P4 is re-evaluated on P1–P3; NRMSE on the validation
sets must not blow up relative to P4.

## 7. Stage 5 — Verification: ensemble size

**Script:** `scripts/05_ensemble_size.py` (on P4, final adopted config).

Sweep N (e.g. 25, 50, 75, 100, 150, 200), N_RUNS independent passes each, reporting mean ±
std of: measured normalised RMSE / NIS / 2σ coverage; NSD normalised RMSE / coverage /
spread-skill; Asn normalised RMSE; wall-clock per pass. This justifies the production
`ENSEMBLE_SIZE = 100` (accuracy and calibration stable, no instability) and quantifies
Monte-Carlo run-to-run variability at N = 100.

---

## 8. Auxiliary components (fixed, structural)

- **IQR clipping.** After each predict *and* update, the 7 NSDs (`CLIP_STATES`) are clipped
  to `[1e-12, Q3 + 5·IQR]`. Near-zero NSD dynamics lack restoring feedback, so a few members
  can diverge through nonlinear propagation; clipping removes outliers while preserving the
  bulk spread. Standard constrained-EnKF technique.
- **Localization: none** (`NO_UPDATE_STATES = []`). Every NSD, including UDP-Gal, receives
  cross-covariance corrections; UDP-Gal's near-zero instability is bounded by clipping rather
  than by removing it from the update (which would discard its useful uridine coupling).

**Disclosed limitations (not defects — deliberately not masked):**
- **Glc — capped** at `CV_MAX = 0.006`, deliberately **under-dispersed (NIV ≫ 1)**. This is
  **structural model bias** (the glucose submodel is systematically off); forcing NIV = 1
  would inflate its multiplicative-noise band to ~1000 mM against a ~144 mM feed —
  physically impossible. Capped at the physical band-plausibility ceiling and reported as
  under-dispersed, by design. IQR-clipping glucose's upper tail was trialled but did not
  help (the width is bulk multiplicative spread, not outliers), so the cap bounds it, not
  clipping. **Urd and Lac** likewise pin at the cap (their NIV=1 CVs exceed it), trading
  NIV > 1 for physically sensible bands.
- **Gln — floored** at `CV_MIN` with NIV ≈ 0.38 (over-dispersed). With CV negligible, `S ≈ R`,
  so this reflects `R_Gln` (pooled P1–P4 mean) being conservative for P4, **not** a process-
  noise issue. R is kept from data; the mild over-dispersion is conservative (wider band, the
  safe direction) and disclosed. It is **not** fixed by fitting R to innovations.
- **Single α for 7 heterogeneous NSDs** is a parsimony choice; one knob cannot perfectly
  calibrate all seven (e.g. a residual structural gap remains for UDP-GlcNAc).

---

## 9. Ordering and dependencies (why the sequence is fixed)

```
Stage 0  R          (data)              ── upstream of everything
   │
Stage 1  P0         (from R / Q)
   │
Stage 2  noise model (structural)
   │
Stage 3  measured CVs -> NIV=1          (01_tune_cv.py)   ── adopt into config
   │        (changes measured-state spread => changes cross-cov flow into NSDs)
   ▼
Stage 4  NSD alpha  -> min NRMSE        (tune_alpha / run_option_b, on Stage-3 CVs)
   │        band validated by coverage + spread-skill; cross-validated P1-P3
   ▼
Stage 5  verify     -> ensemble-size sweep (final config)
```

**Key dependency:** Stage 4 depends on Stage 3. Changing the measured CVs changes the
cross-covariance transmitted to the NSDs, so **α must be re-swept whenever the CVs change**.
Likewise, if R were ever revised (Stage 0), the entire pipeline re-runs from Stage 3. No
stage may be tuned against a frozen output of a superseded configuration.

---

## 10. Diagnostics reference

| Diagnostic | Definition | Target | Role |
|---|---|---|---|
| **NIV** (measured) | mean(d²/S) per state, forecast innovations | 1.0 | **Selection** (CVs) |
| **NRMSE** (NSD) | RMSE / mean(measurement) | minimise | **Selection** (α) |
| **2σ coverage** | % of measurements within mean ± 2·std | ≈ 95% | Validation |
| **Spread-skill** | mean(std) / RMSE | ≈ 1 | Validation (band) |
| **NIS** (ensemble) | mean(d²/S) across measured states per pass | ≈ 1 | Ensemble health |

---

## 11. Reproduce the full pipeline (macOS venv)

```bash
# Stage 3 — automated measured-state CV calibration on P4 (NIV -> 1)
./.venv/bin/python scripts/01_tune_cv.py --dataset P4 --iters 10 --cv-max 0.006 --cv-min 1e-4
#   -> results/cv_tuning/{cv_final.json, pkl/cv_tuned_P4.pkl, figures/*}
#   Adopt the printed CVs into config.PROCESS_NOISE_CV, then:

# Stage 4 — re-sweep NSD alpha on the adopted CVs (NRMSE), save bands/figures for all states
./.venv/bin/python scripts/04_cross_validate.py --scheme rotate --retune both
#   -> results/option_b/{pkl/option_b_*.pkl, figures/option_b_*.png, summary}
#   Adopt PROCESS_NOISE_ALPHA = <selected> into config, then:

# Stage 5 — ensemble-size verification on the final config
./.venv/bin/python scripts/05_ensemble_size.py --sizes 25,50,75,100,150,200 --n-runs 10 --run ensemble_sens
```

Every run saves pkl trajectories **including uncertainty bands** (mean, std, ±1σ, ±2σ) and
visualization figures under `results/<run>/`.

---

## 12. Current status (2026-07-08)

- **Stage 0–2:** settled (R from data; P0 derived; noise model structural).
- **Stage 3:** ✅ **adopted** — automated CVs (cap 0.006) in `config.PROCESS_NOISE_CV`.
- **Stage 4 (obs tier):** ✅ **adopted** — `PROCESS_NOISE_ALPHA_OBS = 0.002` (Asn/Glu).
- **Stage 4 (NSD α):** ✅ **confirmed = 0.01** by re-sweep on the adopted CVs
  (`03_tune_alpha_nsd.py`, P4, 0.005–0.04). Best calibration/accuracy balance (mean NRMSE ≈
  min, coverage 60%, UDP-Glc perfectly calibrated). **Remaining:** cross-validate 0.01 on
  P1–P3 (via `04_cross_validate.py`) and produce the final all-state bands
  (`scripts/legacy/run_option_b.py --fixed-alpha 0.01`, or `run_enkf.py` + `plot_results.py`).
  UDP-GalNAc and GDP-Man are structural model limitations no α fixes (disclosed).
- **Stage 5:** ensemble-size sweep to be re-run on the final config (prior run reached only
  N = 25 fully + N = 50 partial before interruption).
