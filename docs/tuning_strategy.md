# EnKF Systematic Tuning & Validation Strategy

**Authoritative, end-to-end description of how the EnKF soft sensor is tuned and validated.**
Written to be manuscript-ready and reproducible from data. The chronological decision record
(what was tried and superseded) is in [`tuning_log.md`](tuning_log.md); this file is the
*clean method*.

---

## 0. Overview

A 17-state Ensemble Kalman Filter estimates unmeasured intracellular nucleotide-sugar donors
(NSDs) and other unmeasured states from routine extracellular measurements, over four
independent fed-batch datasets **P1–P4**.

The filter has three tunable noise ingredients, each set by a principled, ordered procedure:

| Ingredient | States | Noise model | How it is set |
|---|---|---|---|
| **R** (measurement) | 8 measured | — | from data (biological triplicate variance), **not tuned** |
| **P0** (initial cov) | all | — | derived from R / process-noise variance |
| **Measured CVs** | 8 measured | multiplicative `N(0,(cv·x)²)` | fixed-point to filter **consistency** (NIV=1), capped |
| **α_obs** | Asn, Glu | additive `(α·scale)²` | **calibration** of the uncertainty band (inspect Asn) |
| **α_nsd** | 7 NSDs | additive `(α·scale)²` | **calibration** of the uncertainty band (inspect NSDs) |

**Validation is full-fold cross-validation with per-fold independent tuning:** each dataset in
turn is the training set; the filter is tuned *entirely* on it (its own CVs **and** its own
α), then evaluated on the three held-out batches it never saw. What is shared across folds is
a single **selection RULE**, *not* a single α value — no parameter leaks across folds; only
the method generalizes.

---

## 1. State vector & observability

| Class | States | In update? | Noise | Tuned by |
|---|---|---|---|---|
| Measured (8) | Xv, mAb, Gal, Urd, Glc, Amm, Gln, Lac | yes | multiplicative CV | NIV = 1 |
| Observable-unmeasured (2) | Asn, Glu | no | additive (α_obs) | band calibration |
| NSDs (7) | UDPGal, UDPGalNAc, UDPGlc, UDPGlcNAc, GDPMan, GDPFuc, CMPNeu5Ac | no | additive (α_nsd) | band calibration |

Asn/Glu are **upstream** of the NSD pathway → α_obs affects the NSDs, but α_nsd does not
affect Asn/Glu. Hence the tuning order **CVs → α_obs → α_nsd**.

---

## 2. Stage 0–2 — fixed, structural

- **R (measurement noise):** biological triplicate error-bar variance, pooled over P1–P4. An
  independent, instrument-level quantity (assay replicate precision), **never fit to filter
  diagnostics** — that independence is what makes the methodology defensible. Where the
  consistency calibration later flags an R value as conservative (Gln), it is *disclosed*, not
  fixed by fitting R to residuals (which would be circular).
- **P0 (initial covariance):** set separately from per-step Q. Measured: `P0 = R`; unmeasured:
  `P0 =` process-noise variance.
- **Noise model:** measured → multiplicative (zero noise at zero concentration, scales with
  level); unmeasured → additive (multiplicative blows up with no measurement to restrain it).
  The additive variance is nondimensionalised through one scalar per tier:
  `Q_ii = (α · scale_i)²`, where `scale_i` is the fixed **median magnitude** of state *i*
  (climatological scaling — one knob sets overall magnitude, relative structure frozen).

---

## 3. Stage 3 — measured-state CVs → filter consistency (NIV = 1)

**Script:** `01_tune_cv.py` (automated). For each measured state the per-step CV is driven to
the consistency target by the fixed point

    NIV_j = mean(d_j² / S_jj),  d = z − Hx̂,  S = P_zz + R
    CV_j ← CV_j · √NIV_j          (contracts because S_jj grows ~ CV_j²)

NIV > 1 ⇒ under-dispersed, < 1 ⇒ over-dispersed. Iterate to `|NIV−1| < tol` or `--iters`.

**Cap `CV_MAX = 0.006` is a physical band-plausibility ceiling, not a tuned value.** Per-step
CV compounds over the ~2400-step measurement interval as `CV·√N`, so `0.006·√2400 ≈ 0.29`
(~30 %/24 h model error) — the most process noise we attribute to any single state before its
band exceeds physically plausible ranges. States that cannot reach NIV=1 within `[1e-4, 0.006]`
are pinned and flagged (reporting only; no per-state special-casing):
- **Glc — capped** (structural glucose-model bias; forcing NIV=1 would inflate its
  multiplicative-noise band to ~1000 mM vs a ~144 mM feed — physically impossible). Left
  under-dispersed, disclosed.
- **Gln — floored** (`S ≈ R`, so the residual over-dispersion is an `R_Gln`-is-conservative
  signal, not a CV one). R kept from data; disclosed.

The ODE step uses `scipy.integrate.odeint` (single self-cleaning LSODA call) rather than a
per-call `ode` object — numerically identical but bounds native memory over the millions of
per-pass calls (long runs previously OOM-ed).

---

## 4. Stage 4 — additive α by a principled trade-off (never argmin-NRMSE)

The NSDs (and Asn/Glu) are **not assimilated** — there is no innovation, so NIV is undefined
for them. α is the only handle on their uncertainty band. It is **never** chosen by minimising
NRMSE: argmin-NRMSE always picks the *smallest* α — a tight band that hugs the point estimates
but ignores whether it *covers* the data (empirically the narrowest α gives ~30 % coverage, a
grossly over-confident band). NRMSE is used only as an accuracy *guard rail*, not as the thing
being minimised.

The two tiers use trade-off rules matched to their role, both selected on the training set of
each fold and reported as a single shared **rule**:

- **α_obs** (Asn & Glu share it; Glu has no measurements and rides along). Asn's NRMSE over the
  sweep is **non-monotone** — its minimum is at the *smallest* α (an over-confident band,
  ~28–37 % coverage), and it rises sharply just above the calibrated region. So α_obs is set by
  an **accuracy-guarded width rule**:

  > **α_obs = the largest α whose Asn NRMSE stays within 25 % of its minimum-achievable value**
  > — i.e. the widest band we can afford before accuracy degrades sharply, maximising coverage
  > subject to a near-optimal point estimate.

  The knee is unambiguous: at the selected α the NRMSE is ~1.13× the minimum, while the next
  grid step jumps to ~1.44×, so any tolerance in **[15 %, 43 %]** gives the same pick in all
  four folds. This rule selects **α_obs = 0.002** for P1–P4 (the value coincides; the *rule* is
  what is shared). Choosing α_obs carries **no downstream cost**: a controlled test
  (`scripts/test_asn_alpha_on_nsd.py`, α_obs 0.002 → 0.01 at fixed CVs and α_nsd) showed the
  NSD accuracy is unchanged (mean NSD NRMSE flat) and the coverage change is small and
  inconsistent in sign across folds — Asn/Glu process noise does not propagate materially into
  the downstream NSD estimates, so the two tiers are **empirically independent** and α_obs is a
  pure Asn choice.

- **α_nsd** (7 NSDs — the reported soft-sensor outputs): selected by **band calibration** —
  coverage → ~target and spread-skill (mean std / RMSE) → ~1 on the states we can actually
  cover — because honest uncertainty on the sensor outputs is what matters, and (unlike Asn)
  the NSDs have no clean NRMSE knee to exploit. *(Rule pending inspection of the α_nsd sweep.)*

**Structural limitations disclosed (no α fixes these):**
- **UDP-GalNAc** — model overpredicts; raising α widens the band *upward*, away from the data.
- **GDP-Man** — model sits ~flat near zero while data are 0.4–0.9 mM; the band can't bridge it.

These two are reported as model-fidelity limitations; α is calibrated on the five tractable
NSDs (which reach 82–100 % coverage at α_nsd ≈ 0.03).

---

## 5. Full-fold cross-validation (the validation)

**Script:** `04_cross_validate.py`, two stages, output under `results_single_sweep/`.

**Principle — per-fold independent tuning; a shared rule, not a shared value.** For each fold
(training set P_k), the filter is tuned entirely on P_k — its own calibrated CVs *and* its own
α_obs/α_nsd — and then validated on the other three batches. The four folds are independent;
the only thing shared is the *selection rule* used to pick α. This is the honest generalization
test: nothing from a held-out batch touches the filter applied to it (R stays pooled as an
instrument-level constant).

**Stage `sweep`** (per fold, on the training set): calibrate CVs (save the full
per-iteration NIV log + CVs + trajectory), then sweep α_obs and α_nsd, saving **every** α's
full 17-state mean/std trajectory + uncertainty bands + metrics as pkl, plus overlay figures.
Nothing is auto-selected.

**Inspect → pick → derive the rule:** inspect the per-fold overlays, hand-pick each fold's
α (recorded in `results_single_sweep/picks.json`), and articulate the single criterion that reproduces
those picks (e.g. "smallest α whose median spread-skill on the tractable NSDs ≥ threshold").

**Stage `validate`:** each fold loads *its own* CVs + *its own* picked α and applies them to
its held-out sets, saving all-state bands + grids + a cross-fold summary.

**Everything is pkl-backed** — any plot or statistic can be regenerated offline without
re-running. The sweep (~12 h at N=100) is one-time; validation (~1.6 h) and re-inspection are
fast.

---

## 6. Stage 5 — ensemble size

**Script:** `05_ensemble_size.py`. Sweeps N ∈ {25,50,100,150,200} on P4 with the adopted
config, N_RUNS seeds each, reporting NIS / coverage / spread-skill / NRMSE / cost vs N.
Confirms **N = 100** sits on the plateau (accuracy and calibration stable, no small-N
instability, diminishing returns above). Per-run mean + std trajectories saved.

---

## 7. Auxiliary (fixed, structural)

- **IQR clipping** of the 7 NSDs to `[1e-12, Q3+5·IQR]` after predict/update — bounds outlier
  divergence through the nonlinear NSD dynamics (standard constrained-EnKF). Glc is **not**
  clipped (its wide band is bulk multiplicative spread, not outliers — bounded by the CV cap).
- **Localization: none** — every NSD receives cross-covariance corrections; UDP-Gal's near-zero
  instability is handled by clipping, not by removing it from the update.

---

## 8. Reproducibility model

- **`config.py` holds the final tuned values** — the canonical filter the paper reports.
- **Numbered scripts (`01`–`05`) reproduce the analysis** that justifies those values; each is
  self-contained (reads `data/` + `config`, writes to `results*/`, no run-to-run dependency).
- **α selection is a human inspection made once**, encoded as a documented rule; a reader
  reproduces the sweep figures and sees why the rule is right.
- **Data:** `data/raw/P{1..4}.xlsx` must be present — either committed or obtained per the
  README — otherwise a fresh clone cannot run.

Run order (empty `results*/` → populated):
```
01_tune_cv.py         Stage 3   measured CVs → NIV=1 (cap 0.006)
02_tune_alpha_asn.py  Stage 4a  α_obs sweep + inspection
03_tune_alpha_nsd.py  Stage 4b  α_nsd sweep + inspection
04_cross_validate.py  full-fold CV: --stage sweep  → inspect → picks.json → --stage validate
05_ensemble_size.py   Stage 5   ensemble-size verification (N=100)
run_enkf.py + plot_results.py    final production soft-sensor figures at config values
```

---

## 9. Disclosed limitations (model, not tuning)

- **Glc** — structural glucose-model bias; capped, under-dispersed by design.
- **Gln** — `R_Gln` (pooled) conservative for P4; over-dispersed at the CV floor.
- **UDP-GalNAc, GDP-Man** — structural NSD-model bias; uncovered at any α.
- **Single α per tier** — a parsimony choice; one knob cannot perfectly calibrate seven
  heterogeneous NSDs.

These are reported honestly rather than masked by inflated noise or fitted R.

---

## 10. Status

- Stages 0–3 (R, P0, noise model, CV calibration incl. cap 0.006, odeint) — settled.
- Stage 4 **α_obs — settled**: accuracy-guarded width rule → **0.002** all folds (see §4),
  confirmed downstream-inert on the NSDs. Recorded in `results_single_sweep/picks.json`.
- Stage 4 **α_nsd — pending**: band-calibration rule to be fixed after inspecting the
  `results_single_sweep/fold_*/alpha_nsd` sweep figures; then written into `picks.json`.
- Stage 5 (N = 100) — verified.
- Full-fold CV — pipeline built (`04` two-stage); α_obs picked, α_nsd + validate pending.
