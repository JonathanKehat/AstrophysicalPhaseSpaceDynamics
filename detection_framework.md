# Constraining ∂f/∂t: A Direct Steady-State Test of the Galactic Disc
## Robustness and sensitivity framework for bounding the Boltzmann time-derivative from phase-space snapshots

**The object of inference.** The collisionless Boltzmann equation in the 1-D vertical problem is

```
∂f/∂t + w ∂f/∂z − (∂Φ/∂z) ∂f/∂w = 0 .
```

Steady state is the statement ∂f/∂t ≡ 0. The goal of this research is to **measure, or place
quantitative upper limits on, the term ∂f/∂t itself** — the direct time-derivative of the
phase-space distribution function — from a single observed snapshot of stars in (z, w). Not to
fit a time-dependent potential, not to characterize potential families for their own sake:
those enter only as *nuisances* that must be profiled out to make a statement about ∂f/∂t.

Rearranged, the CBE turns a snapshot into an estimator of the non-stationarity **field**:

```
∂f/∂t (z, w) = − w ∂f/∂z + K_z(z) ∂f/∂w ,      K_z ≡ ∂Φ/∂z at t_obs,
```

or, in the numerically superior score form (working with s ≡ ∇ ln f, which normalizing flows
give by autodiff),

```
ε(z, w) ≡ ∂ ln f/∂t = − w s_z(z, w) + K_z(z) s_w(z, w)      [units: 1/time].
```

ε is the **fractional non-stationarity rate**; its inverse τ_ss(z,w) = 1/|ε| is the local
steady-state timescale, to be compared with the vertical orbital period
T_z = 2π/√(4πGρ_tot(0)) ≈ 90 Myr. All results are reported in these physical units:
*"the local DF is stationary at the level |ε| < X Gyr⁻¹ (τ_ss > 1/X)"* — or a detection of
ε ≠ 0 with a reconstructed map of where in (z, w) the flow of probability is going.

**Two structural advantages over the back-integration approach** (worth stating up front):

1. **No history assumption.** The CBE at t_obs involves only the *instantaneous* force K_z(z).
   No equilibrium-at-t_init assumption, no back-integration, no dependence on Φ(t) at earlier
   times. The snapshot-degeneracy problem of the MLE method does not arise in this form.
2. **The potential enters through one function of one variable.** K_z(z) is a single 1-D
   profile. At each height z it is a *single number* that must cancel the w-dependence of an
   entire function — this is what makes ∂f/∂t partially identifiable even with a completely
   free potential (§1).

---

# Part I — Identifiability: which part of ∂f/∂t a snapshot can constrain

## 1.1 The fundamental degeneracy, stated exactly

Given only f̂(z, w), *any* K_z(z) inserted into the CBE yields a mathematically valid ∂f/∂t.
So (K_z, ∂f/∂t) are inferred jointly, and the irreducible ambiguity is:

> "stationary in some wrong potential" is indistinguishable from "non-stationary in the true
> potential" — *for the component of ∂f/∂t that a potential can absorb.*

The framework therefore stratifies every statement by how much is assumed about K_z. Three
levels, from assumption-free to physics-informed:

## 1.2 Level 0 — the continuity statistic (potential-free)

The zeroth w-moment of the CBE eliminates K_z exactly (∫ K_z ∂f/∂w dw = 0):

```
∂ν/∂t = − ∂(ν ⟨w⟩)/∂z ,        ν(z) = ∫ f dw .
```

The vertical mass-flux divergence is a **completely potential-independent measurement of
∂ν/∂t** — the w-integral of ∂f/∂t — estimable from the data alone (the mean-velocity profile
ν⟨w⟩(z) and its z-derivative). Any statistically significant flux divergence is unconditional
evidence of non-stationarity. This is the cheapest, most robust detector and should head the
pipeline.

## 1.3 Level 1 — the per-slice profiled residual (free potential)

At fixed z, the steady-state hypothesis requires one scalar a = K_z(z) to satisfy
w s_z(z, w) = a s_w(z, w) **for all w**. Project in L²(f dw):

```
a*(z)   = ⟨ w s_z s_w ⟩_f / ⟨ s_w² ⟩_f            (the best steady-state force)
R⊥(z,w) = − w s_z + a*(z) s_w                       (the profiled residual)
```

This gives an exact and useful decomposition of the non-stationarity field:

- the component of (w s_z) **parallel** to s_w in each z-slice is *absorbable* by a potential —
  it is invisible at Level 1 (it becomes a bias of a*(z) instead);
- the **orthogonal complement** R⊥ is *potential-proof*: no K_z(z) whatsoever can cancel it.
  ⟨R⊥²⟩_f > 0 (beyond its noise floor) is evidence of ∂f/∂t ≠ 0 that survives total ignorance
  of the potential.

Note the dual reading: minimizing ⟨(−w s_z + a s_w)²⟩_f over a(z) is precisely the
steady-state-based potential estimator of the Deep-Potential literature (Green & Ting 2020;
Buckley et al.; An et al.). This framework runs the same computation but treats the *residual*
as the signal and the potential as the nuisance — and, as a by-product, quantifies how much
disequilibrium biases steady-state-based K_z inference (the contamination output, §4.4).

## 1.4 Level 2 — physical-potential family (assumption-strengthened)

Restrict a(z) to a physical family: K_z from Poisson with ρ ≥ 0, baryon-budget priors,
optionally the parametric baryons + sech² halo with priors, or a constrained NN. Now part of
the Level-1-absorbable component becomes detectable too, because the wrong-potential
explanation is no longer *allowed*. Constraints on ∂f/∂t tighten monotonically with assumption
strength; the report always states results at all three levels — the stratification *is* the
scientific honesty of the method.

The first w-moment (momentum Jeans equation) sits naturally here:
∂(ν⟨w⟩)/∂t = −∂(ν⟨w²⟩)/∂z − ν K_z — potential-dependent, so it belongs to Level 2, and adds
sensitivity to breathing-type perturbations that leave the Level-0 flux small.

## 1.5 Statistics (what gets a null distribution and a limit)

All estimable as averages over the observed stars (i ranges over stars; scores from the flow):

| Statistic | Definition | Level | Detects |
|-----------|-----------|-------|---------|
| T₀ | ∫ dz [∂(ν̂⟨w⟩)/∂z]² / norm | 0 | net probability flux (bending-like, one-sided) |
| T₁ | (1/N) Σ_i R⊥(z_i, w_i)² | 1 | any Φ-orthogonal reshaping (spirals, breathing asymmetries) |
| T₂ | (1/N) Σ_i ε(z_i, w_i)² with K_z restricted to the physical family | 2 | everything, conditional on priors |
| ε-map | pointwise ε̂(z, w) with per-point errors | 1–2 | reconstruction, not just detection |

T-statistics are positive-definite ⇒ estimation noise biases them **upward**; debiasing is a
first-class problem (§2.3), and detection is always relative to the empirical null (Part II),
never to zero.

---

# Part II (method core) — Estimation pipeline

1. **DF and score estimation.** Fit a normalizing flow to the (z, w) sample
   (`NormalizingFlow_Examples.ipynb` is the starting point); scores s_z, s_w by autodiff.
   KDE as a cross-check at low dimension. Evaluate scores only inside the data-supported
   region (define by a density threshold); everything outside is reported as unconstrained.
2. **Cross-fitting.** Split the sample; fit the flow on fold A, evaluate the statistics as
   averages over fold B stars (and swap). This removes the overfitting component of the
   positive bias in T; the remaining score-variance bias is calibrated by the Part-III null.
3. **Uncertainty.** Flow ensembles (seeds × architectures) + bootstrap over stars → per-point
   error bars on ε̂ and sampling distributions for T₀/T₁/T₂. Architecture spread is part of the
   error budget, not a footnote.
4. **Profiling.** a*(z) per slice by the weighted least squares of §1.3 (closed form);
   Level 2 by constrained optimization of K_z within the physical family.
5. **Ground truth in simulations.** Two independent routes, cross-validated:
   (a) *CBE route*: plug the true K_z and the true (large-N) f into the CBE identity;
   (b) *finite-difference route*: f̂ at t_obs ± Δt from adjacent `evolve_record` snapshots.
   Agreement of (a) and (b) validates the whole numerical chain before any inference is run.

Existing machinery mapping: the simulator (equilibrium ICs — the hard-won lesson of the
Widmark work — satellite kick, `h_fourier` time-dependent halo, snapshot recording) is the
data factory and truth source; the flow notebook is the estimation engine; the back-integration
MLE becomes a *complementary channel* (it integrates information over the orbital history and
so can be more sensitive to old transients, at the price of the t_init assumption) — useful as
a cross-check, no longer the primary method.

---

# Part III — Robustness front: the null distribution of apparent non-stationarity

## 3.1 Definition

Generate data from an **exactly stationary** truth: the equilibrium DF in a static potential
(the self-consistent ICs; note this equilibrium must be exact — a 26×-too-thin IC self-mixes
and is a *real* signal, not a null). Run the full pipeline. The null distributions of T₀, T₁,
T₂ and of the ε̂-map define the **false-non-stationarity background**: how much apparent
∂f/∂t the method reports when the true value is identically zero.

Everything is empirical Monte Carlo — the statistics are positive-definite, the flow is a
nonparametric estimator, and no asymptotic χ² is trustworthy here.

## 3.2 False-signal hierarchy (what can fake ∂f/∂t ≠ 0)

Ordered by expected severity for this method:

1. **Score-estimation noise and flow bias (irreducible).** Finite N ⇒ noisy s_z, s_w ⇒
   positive bias in all T's; flow smoothing distorts scores near edges and in low-density
   wings (exactly where spiral signal lives). Sets the statistical floor as a function of N.
   Ladder: N ∈ {500, 2000, 8000, 3×10⁴}; flow width/depth/seeds; KDE bandwidths.
2. **Solar-frame errors (the dangerous systematic).** A wrong w_⊙ shifts ⟨w⟩ globally → fake
   flux divergence in T₀; a wrong z_⊙ tilts ν(z) → both. These produce *stable, coherent*
   false signals, not noise. They must be free nuisance parameters in the fit, and the null
   must include their residual uncertainty.
3. **Selection effects.** An unmodeled z-dependent selection S(z) adds −w ∂ln S/∂z to ε̂ —
   an odd-in-w spurious term. Subtlety worth measuring: for near-Gaussian slices
   (s_w ∝ −w/σ²) this term is *parallel* to s_w, so it biases a*(z) (the potential) rather
   than R⊥ (the signal) — Level 1 is partially self-protecting; T₀ is not. Ladder: |z| cuts,
   one-sided dust-like cuts, w-dependent cuts, each modeled/unmodeled.
4. **Measurement errors.** Error convolution smooths f and rescales scores; underestimated
   errors masquerade as washed-out structure; correlated (z, w) errors generate fake cross-
   gradients. Ladder at Gaia-like magnitudes, modeled and mismodeled.
5. **Level-2 prior misspecification.** A too-rigid potential family inflates T₂ (residual
   that honest freedom would have absorbed) — the classic "wrong model reads as
   disequilibrium". Measured by fitting deliberately wrong families to stationary data.
   (Level 0/1 are immune by construction — this is why the stratification exists.)
6. **Numerics.** Snapshot spacing for the finite-difference truth, integrator drift in the
   simulated equilibrium (a not-quite-equilibrated "equilibrium" is a real signal — verify
   stationarity of the simulation itself over t_obs before using it as a null).

## 3.3 Outputs

- Null quantiles (50/95/99%) of T₀, T₁, T₂ vs N and error/selection configuration — the
  **detection thresholds**;
- the noise floor of the ε̂-map (per-pixel null band in Gyr⁻¹) — the smallest mappable signal;
- the **false-feature catalogue**: which (z, w) structures each systematic paints into ε̂
  (solar-frame → coherent dipoles; selection → odd-in-w bands; flow edge bias → rim
  artifacts) — so that a Gaia detection can be screened against each known artifact shape;
- the calibrated **debiasing constants** (mean of each T under the null) to subtract;
- stability classification per statistic under the battery (architecture, seeds, folds,
  bandwidths): stable ⇒ usable; fragile ⇒ reported but never claimed.

---

# Part IV — Sensitivity front: injected disequilibrium and recovery of ∂f/∂t

## 4.1 Definition

Inject a physically motivated perturbation of controlled amplitude A; the simulation provides
the **true field ∂f/∂t(z, w)** (§2.5). Run the pipeline; score, per level:

- **Detection power**: P(T > null 99% | A) → minimum detectable amplitude (logistic fit in
  log A, A_min at 90% power) and, in physical units, the **minimum detectable ε_rms in
  Gyr⁻¹** — the headline sensitivity number of the whole program;
- **Field reconstruction**: correlation and RMS of ε̂(z, w) vs truth over the supported
  region; does the map localize *where* the DF is changing?
- **Level attribution**: which level fires (a parity/moment classifier for the perturbation
  type, §4.3);
- **Absorbable fraction**: ⟨R∥²⟩/⟨ε²⟩ of the true signal — how much of each physical signal is
  invisible in principle at Level 1 (the identifiability theorem made quantitative per signal).

## 4.2 Injection library (unchanged physics, new observables)

| ID | Source | Implementation | Expected signature |
|----|--------|----------------|--------------------|
| S1 | satellite flyby | `Kz_sat_common`, Σ_sat | strong ε during/after passage; T₀ + T₁ fire; ε-map = displaced-centroid dipole winding into spiral |
| S2 | bending mode | IC offset z or w | pure T₀ (net flux), one-armed ε pattern |
| S3 | breathing mode | IC rescale of w (or even forcing) | T₀ small, T₁/T₂ fire; two-armed ε pattern, ⟨zw⟩ growth |
| S4 | adiabatic halo h(t) | `evolve_record(h_fourier)` | ε ≈ 0 by adiabatic invariance — the DF *tracks* the slow potential. Prediction: near-blindness, now for the *correct physical reason*; measure the residual ε ∝ (T_z/τ_h) scaling |
| S5 | pre-formed phase spiral | Widmark-form DF IC | the canonical target: large R⊥ (spiral is exactly a not-absorbable w-structure), tiny moments — T₁ ≫ T₀; winding rate of the ε pattern dates it |
| S6 | bar/arm periodic forcing | even K_z forcing | periodic ε in the orbital band; resonant bands in (z, w) where Ω_z matches |

Amplitude ladders span from below the null band to the strong-signal regime; ≥ 25 seeds per
rung; the timescale axis (impulsive ↔ orbital ↔ adiabatic) is swept for S1/S4/S6 — the
sensitivity-vs-timescale curve, with its adiabatic cutoff, is a primary result.

## 4.3 Classification by construction

The level stratification doubles as a physical classifier: T₀-only ⇒ bulk flux (bending-like);
T₁-dominant ⇒ fine-grained phase-space reshaping (spiral/breathing); T₂-only ⇒ signal
degenerate with a free potential but excluded by physical priors (report with its prior
dependence). The measured cross-talk table (injected type vs firing pattern) makes the
classifier quantitative.

## 4.4 Dual output: contamination of steady-state potential inference

For every injection, record the bias of the Level-1/2 best steady-state force a*(z) (and the
implied ρ_dm, h_dm) as a function of A. This is the answer to "how wrong is a Deep-Potential-
style measurement if the disc is actually non-stationary at amplitude A" — publishable on its
own and directly relevant to the literature.

---

# Part V — Combined criterion and reporting

A **detection of non-stationarity** requires:

1. **Statistical**: T_ℓ exceeds its misspecification-inflated empirical null (99%) at level ℓ;
2. **Stability**: survives the battery — flow architecture/seeds, cross-fit folds, bandwidths,
   solar-frame nuisance refits, selection-model swaps; no support-edge dominance (the
   statistic must not be carried by the rim pixels);
3. **Coherence**: the ε̂-map matches a physical template class (§4.3) and not a catalogued
   artifact shape (§3.3); parity/moment signature consistent with the firing level;
4. **Scale honesty**: quoted with its level — e.g. "non-stationary at Level 1
   (potential-free), ε_rms = X ± Y Gyr⁻¹, τ_ss = 1/X ≈ Z × T_z".

Otherwise, **upper limits**: ε_rms < X Gyr⁻¹ (τ_ss > 1/X) at 95%, per level and per timescale
band, with the adiabatic band explicitly marked "not constrainable in principle" (§4.2/S4)
rather than quoted as an empty limit.

Headline figure: per level, the null band of T vs N (Part III) overlaid with power curves vs
injected ε_rms (Part IV) — the detectable window in physical units — plus one ε̂-map example at
threshold amplitude showing what a marginal detection looks like.

---

# Part VI — Gaia strategy

1. **G0 — matched mocks.** Solar-neighbourhood column mocks with DR3 error sampling, the
   selection function, and free (z_⊙, w_⊙); re-derive all null bands and A_min at the true N.
   *No Gaia number is quoted against a clean-simulation null.*
2. **G1 — positive control.** The Gaia phase spiral is a known ∂f/∂t ≠ 0 at Level 1. The
   pipeline must (a) detect it (T₁ above null), (b) map it (ε̂-map showing the winding
   pattern), (c) date it (winding rate of the ε pattern), and (d) classify it (T₁ ≫ T₀ up to
   the known bending component). Failure here invalidates the pipeline, not the Galaxy.
3. **G2 — science products.**
   (a) the measured ε̂(z, w) map of the solar column with calibrated errors — *the first direct
   map of ∂f/∂t in the Milky Way disc*, if it clears the bands;
   (b) stratified limits: potential-free (Level 0/1) and physics-informed (Level 2) bounds on
   ε_rms per timescale band — "the local disc is in steady state to within τ_ss > X";
   (c) the by-product steady-state force a*(z) with a disequilibrium-corrected error bar
   (§4.4 contamination applied at the measured/bounded ε);
   (d) split-sample coherence (azimuth/radius/magnitude): real dynamics repeats, systematics
   don't.

---

# Part VII — Identifiability catalogue

1. **The absorbable component (fundamental).** The part of ∂f/∂t parallel to s_w per slice is
   exactly degenerate with the instantaneous force; only priors on K_z (Level 2) buy it back.
   Every claim carries its level. The absorbable fraction per physical signal is a measured
   output (§4.1), not a hand-wave.
2. **Adiabatic invariance (physical, not statistical).** Slowly varying potentials transport f
   along adiabats with ε suppressed by (T_z/τ_drive) — near-zero true signal, so
   non-detection of slow evolution is a property of the Galaxy's f, not of the method. The
   sensitivity-vs-timescale curve makes this quantitative.
3. **Snapshot locality.** Only K_z(z, t_obs) enters; nothing about Φ's history is assumed or
   constrained. (The back-integration channel, with its t_init assumption, is the
   complementary probe of history — the two methods bracket the problem.)
4. **Support limits.** Scores and residuals are defined only where stars are; the wings and
   high-|z| region — where spiral contrast is largest — are also where flow bias is worst.
   The trade-off is quantified by the rim-artifact entries of the false-feature catalogue.
5. **Selection–force degeneracy.** Near-Gaussian slices route unmodeled selection gradients
   into a*(z) rather than R⊥ — protective for detection, poisonous for the force by-product;
   the two failure modes are complementary and both must be tracked.
6. **Frame degeneracy.** (z_⊙, w_⊙) offsets are exactly a rigid phase-space translation —
   degenerate with a bending-mode signal at small amplitude. Bending claims at Level 0 are
   therefore only as good as the frame determination; report the frame-marginalized limit.
7. **Statistic positivity.** T ≥ 0 always; without the empirical null and debiasing constants,
   any noise level "detects" non-stationarity. The null campaign is not optional plumbing —
   it is the definition of the measurement.

---

# Execution order

| Phase | Content | Builds on |
|-------|---------|-----------|
| 0 | Pipeline: flow + scores + cross-fitting + T₀/T₁/T₂ + ε-map evaluator; harness `run_experiment(data_cfg, pipe_cfg, seed)`; records to disk. Validate the two ground-truth routes (§2.5) against each other on one satellite run. | — |
| 1 | Stationarity audit of the simulator: confirm the equilibrium IC gives ε_true ≈ 0 over t_obs at large N (this certifies the null factory). | 0 |
| 2 | Null campaign: T-statistics and ε-map noise floors vs N, flow config; debiasing constants. First deliverable: the detection thresholds in Gyr⁻¹. | 1 |
| 3 | Sensitivity: S5 (spiral — the canonical Level-1 target) and S1 (satellite) amplitude ladders → power curves, minimum detectable ε_rms, field-reconstruction fidelity, absorbable fractions. | 2 |
| 4 | Remaining injections S2/S3/S6 + the S4 adiabatic-cutoff curve; level-attribution cross-talk table; contamination of a*(z) (§4.4). | 3 |
| 5 | Systematics ladder: solar frame, selection, measurement errors, Level-2 family misspecification → inflated nulls + false-feature catalogue. | 2 |
| 6 | Gaia mocks (G0) → re-derived bands → phase-spiral positive control (G1) → ε̂-map + stratified limits (G2). | 3–5 |

Phases 0–3 form a complete, standalone result: calibrated thresholds and the first
injection-verified measurement of ∂f/∂t on simulated data, in physical units, with the
potential fully profiled out at Level 1. Phase 6 is the Gaia paper: a direct map of — or the
strongest direct limits on — the Boltzmann time-derivative term in the local Galactic disc.
