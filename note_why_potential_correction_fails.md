# Why a Time-Dependent NN Potential Correction Cannot Recover h_true Under an Initial-PDF Mismatch

## Setup

We back-integrate observed stellar coordinates `(z_i, w_i)` at observation time `t_obs` to initial coordinates `(z_i(0), w_i(0))` using a model gravitational potential

```
Φ_model(z, t) = Φ_static(z; h) + V_nn(z, t)
```

where `Φ_static` is the fixed baryon + DM halo contribution parameterised by the dark-matter scale height `h`, and `V_nn(z, t)` is an arbitrary neural-network potential correction. The likelihood assumes a Gaussian initial distribution:

```
f_0(z, w) = N(0, σ_z_fit) × N(0, σ_w_fit)
```

Maximum-likelihood estimation minimises

```
NLL(h, V_nn) = Σ_i [ 0.5 · (z_i(0) / σ_z_fit)² + 0.5 · (w_i(0) / σ_w_fit)² ] + const
```

over `(h, V_nn)` jointly. The data was generated with `(σ_z_true, σ_w_true)`. The "PDF mismatch" we study is the case in which `σ_w_fit < σ_w_true` (or analogously `σ_z_fit < σ_z_true`).

## Claim

**No `V_nn(z, t)` — regardless of architecture, time-dependence, network capacity, or physical constraints (positivity, smoothness, finite mass, etc.) — can move the joint MLE to `h* = h_true`.**

Below is the argument, in five short steps. The conclusion is structural: the bias lives in the loss function itself, not in the model space, and no enrichment of the model space can compensate for that.

---

## Step 1: At the truth, the loss is large, not small

Plug in the correct model: `h = h_true`, `V_nn ≡ 0`. The dynamics are then exactly those used to generate the data, and the back-integration is exact:

```
w_i(0) ~ N(0, σ_w_true),    z_i(0) ~ N(0, σ_z_true)
```

The expected per-star w-term in the NLL is

```
E[ 0.5 · (w(0) / σ_w_fit)² ] = 0.5 · σ_w_true² / σ_w_fit²
```

When `σ_w_fit < σ_w_true`, this is much larger than the ½ it would be if the PDF were correct. The truth is *not* a stationary point of the loss; the gradient `∂NLL/∂h` at `(h_true, 0)` is non-zero, and the optimiser will move *away* from the truth to lower the loss.

## Step 2: The loss minimum lies wherever the back-integrated w-distribution is narrow

The optimiser wants the back-integrated initial-velocity variance to satisfy

```
⟨ w(0)² ⟩  ≈  σ_w_fit²    (so the squared-term in the NLL is of order one).
```

Two **physically independent** mechanisms in the model can produce a narrow `w(0)` distribution:

**(a) Drift in h.** Lowering h increases the magnitude of the anharmonic `z⁴/h²` term in `Φ_DM(z)`. High-amplitude orbits become "stiffer" — their back-integration compresses them more — so the distribution of `w(0)` is narrower.

**(b) A positive-curvature `V_nn`.** Adding a midplane mass concentration produces a quadratic correction `V_nn(z) ≈ ½ Δω² z²` near midplane. This raises the effective harmonic frequency at midplane, which in turn modifies the back-integration so that `w(0)` is narrower. A flexible NN finds this very easily.

The MLE landscape is therefore *not* point-like at any single minimum; it has a continuous ridge of nearly-degenerate solutions parameterised by trade-offs between (a) and (b). The truth `(h_true, 0)` lies *off* this ridge. Adam picks a point on the ridge based on initialisation and learning rates, but it will not, in general, land at the truth.

## Step 3: Time-dependence cannot help — it can only enlarge the degeneracy

A time-dependent correction `V_nn(z, t)` is a strict superset of `V_nn(z)`:

* Every static V_nn that absorbs the PDF bias is still available as a `t`-constant member of the larger class.
* Additional time-dependent modes — for example, periodic stiffening of the well at orbital frequencies — can produce *additional* mechanisms that narrow the back-integrated `w(0)` distribution.

The dimension of the degenerate ridge grows. Time-dependence *cannot remove* any direction; it can only add more low-NLL configurations to choose from. So a time-dependent NN strictly cannot do better than the static NN — it can only do equally badly or worse.

## Step 4: Physical constraints (e.g. `ρ_nn ≥ 0` via Poisson) don't remove the relevant ridge

A natural physical restriction is to require that the NN's potential correspond to a non-negative mass distribution:

```
∂²V_nn / ∂z² = 4πG · ρ_nn(z) ≥ 0
```

This forces `V_nn` to be everywhere convex. It successfully blocks one direction of bias-absorption — concave corrections like `−z⁴/h²` that would mimic h-decreases by reshaping the anharmonic structure of the potential.

But it does **not** block mechanism (b). A constant positive density near midplane gives a quadratic `V_nn(z) ≈ const · z²`, which is convex — fully allowed under `ρ_nn ≥ 0`. This is exactly the midplane-mass-concentration that narrows `w(0)`. The bias-absorbing direction therefore survives the positivity constraint. The ridge persists.

Any constraint we can write down on the *dynamics-side* model can only carve along its surface. It cannot move the surface itself to a place where `(h_true, 0)` is the optimum, because that's not what the constraint does.

## Step 5: The bias is in the loss function, not the model

The structural obstruction is this. The loss is

```
NLL(h, V_nn) ~ Σ_i (w_i(0; h, V_nn) / σ_w_fit)² + …
```

with `σ_w_fit` **frozen at the wrong value**. The minimum of any such loss over any model class — potential corrections, time-dependent corrections, physics-informed corrections, neural-network corrections, all of them — sits at the model point that produces `⟨ w(0)² ⟩ ≈ σ_w_fit²`. By assumption that's not the truth, because the truth produces `⟨ w(0)² ⟩ ≈ σ_w_true² > σ_w_fit²`.

> **No flexibility on the model side can change where the loss function's minimum sits, because the loss function is defined by the wrong σ_w_fit.**

A NN potential correction is a tool for redistributing flexibility *within* the dynamics. The bias does not live in the dynamics. It lives in the loss function, via the assumed initial distribution. So the tool is pointed at the wrong target.

---

## Corollary: what *can* recover h_true

Only two strategies can succeed.

### (1) Change the loss function

Make `σ_w_fit` (and `σ_z_fit`) **free parameters** of the fit. The joint MLE then minimises over `(h, σ_z_fit, σ_w_fit, …)`, and the joint minimum is at `(h_true, σ_z_true, σ_w_true, …)`.

This is the structurally correct fix: the bias was that the loss had a wrong frozen constant; allowing that constant to float removes the bias at its source.

### (2) Add information beyond the per-star NLL

Cross-validation against held-out stars, multiple datasets at different `t_obs`, an external prior on `h`, observed spatial structure of `σ_w(z)`, etc. Each of these adds a constraint that the wrong-σ_w bias cannot survive simultaneously, because the bias produces different signatures at different datasets or scales.

### Where a NN can usefully live

A NN solution to the PDF-mismatch problem is to put NN flexibility where the bias actually lives — **the initial PDF itself**. Replace the rigid Gaussian `f_0(z, w)` with a NN-parameterised density (a normalising flow over `(z, w)`, or its parametric special case: a Gaussian whose parameters are learnable, or a Gaussian mixture). The likelihood becomes

```
NLL = − Σ_i log f_0_NN(z_i(0), w_i(0))
```

This NN can absorb the PDF mismatch, but **cannot shift orbits**, so it has no leverage on `h` other than the structural information in the back-integration. The joint MLE then recovers `h ≈ h_true` along with the correct `f_0`.

---

## One-sentence summary

A NN potential correction adds knobs to the *dynamics*, but the bias from a wrong initial PDF lives in the *loss function* — so the dynamics-side knobs can never reach the truth, while a NN that parameterises the initial PDF itself can.
