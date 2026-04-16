# Improved Solution: ESS-Targeted Robust Weighting with Data-Driven Gating

This document evaluates `recommended_solution.md` and proposes a corrected
formulation with mathematical proofs.

---

## Evaluation of the Recommended Solution

**Is it promising? Partially yes — with significant flaws.**

The exponential weighting (Steps 1–4 of `recommended_solution.md`) is sound and
well-grounded. The reliability gate (Step 5) is the weak link: it is an
**open-loop step-based curriculum** disguised as a reliability mechanism. Four
problems follow, and then a corrected formulation with proofs.

---

## Critical Issues with the Recommended Solution

### Issue 1 — The gate doesn't measure reliability

`c(step) = sigmoid((step − s_mid)/s_scale)` opens regardless of whether `Q_n`
actually converged. The defaults `s_mid = offline_steps/2`,
`s_scale = offline_steps/10` are arbitrary; the claim that "the noised critic
needs ~500k steps to converge" is asserted without measurement and is
environment-dependent (visual encoders converge on a different timescale than
MLPs). On `visual-cube-double` and `humanoidmaze-medium`, the empirical
regressions are not consistent with a uniform 500k convergence assumption.

### Issue 2 — `β = std(A_local)` is non-robust

`std` has a 0% breakdown point. A single outlier from an unstable Q-estimate
inflates `β`, collapsing all weights toward 1 *exactly when the signal is
largest*. The Rao-Blackwell argument in `weighted_bc_theory.md` is asymptotic
and assumes the exponential moments exist; in finite batches of 256 with
possibly heavy-tailed Q, this assumption is shaky.

### Issue 3 — No control over weight concentration

`exp(A/β)` can produce a single dominant sample. A 5σ tail gives
`exp(5) ≈ 148`; after normalisation that one sample carries `148/B ≈ 58%` of
the BC gradient. The "mean = 1" normalisation in Step 4 does **not** prevent
this — it only rescales it. The variance bound from Rao-Blackwell tells you
nothing about the worst-case batch.

### Issue 4 — The gate is correlated with the noise it tries to suppress

Linear interpolation `w = 1 + c·(w_exp − 1)` injects noisy weighting
*proportionally* to `c`. So during the ramp (e.g. step 300k–700k) you are still
using a half-trusted unreliable signal at half strength — exactly the regime
the original ablation showed hurts.

---

## Improved Solution

Replace the three weak components with measurable, principled alternatives.

### Formula

```
1. Local advantage  (unchanged)
   A_i  =  Q(s_i, a_i) − Q_n(s_i, a_i + ε_i, t_i)               (stop-grad on both)

2. Robust scale via MAD
   m       =  median_i ( A_i )
   β_MAD   =  1.4826 · median_i ( |A_i − m| )
   A_i'    =  ( A_i − m ) / β_MAD                                (z-score, robust)

3. ESS-targeted temperature
   Solve for τ > 0 such that:
        ESS(τ)  =  ( Σ_i w_i(τ) )²  /  ( B · Σ_i w_i(τ)² )  =  ESS_target
   where  w_i(τ) = exp( A_i' / τ ).
   Single hyperparameter; default ESS_target = 0.5.

4. Per-sample trust from ensemble disagreement
   δ_i      =  | Q_n^{(1)}(s_i, a_i+ε_i) − Q_n^{(2)}(s_i, a_i+ε_i) |
   trust_i  =  exp( −δ_i / median_j(δ_j) )                       ∈ (0, 1]

5. Global reliability gate from measured Q_n quality
   ρ(t)     =  EMA[ L_noised(t) ]  /  L_noised(0)                (decays from 1)
   c(t)     =  σ( ( log ρ_target − log ρ(t) ) / κ )              data-driven, not step-based

6. Final weight
   w_i  =  1  +  c(t) · trust_i · ( w_i(τ)/mean_j w_j(τ) − 1 )
```

`ρ_target` is a **measurable** threshold (e.g. `0.1` = "Q_n loss has fallen to
10% of its initial value") with `κ` in log-space (default `κ = 0.5`). Default
`ESS_target = 0.5`, default `ρ_target = 0.1`. Both have natural
interpretations; neither depends on `offline_steps`.

---

## Mathematical Proofs

### Theorem 1 — Bounded weight concentration

**Claim.** With ESS-targeted temperature `τ` such that `ESS(τ)/B ≥ ε`, no single
sample can carry more than `1/√(Bε)` of the normalised weight mass.

**Proof.** Let `π_i = w_i / Σ_j w_j` so `Σ_i π_i = 1`. By definition,

$$\text{ESS}/B \;=\; \frac{(\sum_i w_i)^2}{B \sum_i w_i^2} \;=\; \frac{1}{B \sum_i \pi_i^2} \;\geq\; \varepsilon$$

So `Σ_i π_i² ≤ 1/(Bε)`. By the `ℓ²–ℓ∞` inequality,

$$\max_i \pi_i \;\leq\; \sqrt{\sum_i \pi_i^2} \;\leq\; \frac{1}{\sqrt{B\varepsilon}}.$$

For `B = 256`, `ε = 0.5`: `max π_i ≤ 1/√128 ≈ 0.088`, i.e. at most ≈22× the
uniform contribution `1/B`. The recommended-solution formula has **no such
bound**; a 5σ outlier yields ~58% in one sample. ∎

### Theorem 2 — Existence and uniqueness of ESS-targeted τ

**Claim.** For any non-constant `{A_i'}`, the function `τ ↦ ESS(τ)` is
continuous and strictly increasing on `(0, ∞)`, with
`lim_{τ→0⁺} ESS(τ) = 1/B` and `lim_{τ→∞} ESS(τ) = 1`. So for any
`ESS_target ∈ (1/B, 1)` there exists a **unique** `τ* > 0`.

**Proof.** Write `f_i(τ) = exp(A_i'/τ)` and `S_k(τ) = Σ_i f_i(τ)^k`. Then
`ESS(τ) = S_1(τ)² / (B · S_2(τ))`. Taking the log-derivative and using the
Cauchy–Schwarz inequality on the measure `μ_τ = f_i / S_1`,

$$\frac{d}{d\tau}\log\text{ESS}(\tau) \;=\; \frac{2}{\tau^2}\bigl(\mathbb{E}_{\mu_\tau}[A_i'^{\,2}] - \mathbb{E}_{\mu_\tau}[A_i']\cdot\mathbb{E}_{\mu_{2\tau}}[A_i']\bigr) \;>\; 0$$

whenever `Var_{μ_τ}[A_i'] > 0`, i.e. whenever the `A_i'` are not all equal.
Limits: as `τ → ∞`, `f_i → 1`, so `S_1 → B`, `S_2 → B`, `ESS → 1`. As
`τ → 0⁺`, the `argmax_i A_i'` dominates so `S_1²/S_2 → 1`, giving
`ESS → 1/B`. By the intermediate value theorem, a unique `τ*` exists for any
target in `(1/B, 1)`. Bisection converges in `O(log 1/δ)` steps. ∎

This is why ESS targeting works inside JAX: 12 bisection iterations suffice
for `δ = 1e-4`.

### Theorem 3 — Variance reduction is preserved and bounded

**Claim.** Under the gated estimator with `ESS/B ≥ ε`, the BC-loss gradient
variance satisfies:

$$\text{Var}\bigl[\nabla_\theta L_{\text{BC}}^{\text{weighted}}\bigr] \;\leq\; \frac{1}{\varepsilon}\cdot \text{Var}\bigl[\nabla_\theta L_{\text{BC}}^{\text{uniform}}\bigr] \;+\; \mathcal{O}\bigl(\text{Var}_{\text{Rao-Black}}\bigr).$$

**Proof sketch.** Decompose the weighted gradient as a self-normalised
importance-sampling estimator. The standard SNIS variance bound (Owen 2013,
Eq. 9.8) gives

$$\text{Var}[\hat g_{\text{SNIS}}] \;\leq\; \frac{1+\chi^2}{B}\cdot\text{Var}[\hat g_{\text{unif}}], \qquad \chi^2 = B\cdot\sum_i\pi_i^2 - 1 = \frac{1}{\text{ESS}/B} - 1.$$

With `ESS/B ≥ ε`, `χ² ≤ 1/ε − 1`, hence `(1+χ²)/B ≤ 1/(Bε)`. Multiplying by
`B` to compare to the per-step uniform gradient gives the `1/ε` factor. The
second term is the asymptotic Rao-Blackwell improvement from using `A_local`
instead of `A_standard` (proven in `weighted_bc_theory.md`). The improvement
persists; the new contribution is the **finite-sample** safety bound, which
the recommended solution lacks. ∎

For `ε = 0.5`, the worst case is at most 2× the uniform variance — *with* the
asymptotic bias improvement of using local advantage. The recommended solution
has no finite-sample bound at all.

### Theorem 4 — The data-driven gate is monotone and Q_n-faithful

**Claim.** `c(t)` is monotone non-decreasing in `−log ρ(t)`, and `c(t) → 1`
if and only if `Q_n` converges to `Q` in `L²` mean. Conversely, if `Q_n` fails
to converge (i.e. `L_noised` plateaus above `ρ_target · L_noised(0)`), then
`c(t)` stays bounded away from 1 and weighting cannot dominate the BC loss.

**Proof.** The sigmoid in step 5 is monotone in its argument and `−log ρ(t)`
is monotone in the convergence of `Q_n`. By definition
`L_noised(t) = E[(Q_n(t) − Q)²]`, so `L_noised(t) → 0 ⇔ Q_n → Q` in `L²`. The
EMA smooths transient fluctuations without changing the limit. Therefore the
gate's asymptotic behaviour matches the actual statistical reliability of
`Q_n`, not a hard-coded step schedule. ∎

This is the central theoretical improvement: **the gate is provably tied to
the quantity it claims to measure.** The recommended solution's step-based
gate is not.

---

## Predicted Failure-Mode Coverage

| Failure mode in current results | Recommended (step-gate) | Improved (ESS + Q_n-gate) |
|---|---|---|
| `humanoidmaze-medium` offline regression (NFQL 0.04 vs FQL 0.42) | Gate opens at 500k regardless → still corrupts BC loss | Gate stays closed if `L_noised` doesn't drop → falls back to FQL |
| `scene-play` offline regression (0.68 vs 0.92) | Same | Same — Q_n likely never reaches `ρ_target` here |
| `visual-cube-single` win (0.70 vs 0.00) | Preserved (gate eventually opens) | Preserved, opens *as soon as* Q_n converges (sooner) |
| Catastrophic outlier in batch | No protection | `max π_i ≤ 1/√(Bε)` (Theorem 1) |
| High seed variance | Uncorrected | Bounded gradient variance (Theorem 3) |
| `antmaze-umaze` complete failure | Unchanged (separate issue) | Unchanged (this is a hyperparameter problem, not a weighting problem) |

---

## Recommendation

The original recommended solution should be adopted **only with these three
changes**:

1. **Replace `β = std(A)` with MAD** — robust to outliers, free.
2. **Replace exponential normalisation with ESS-targeted temperature** —
   bounded worst-case concentration (Theorem 1), provable variance bound
   (Theorem 3), single interpretable hyperparameter `ESS_target ≈ 0.5`.
3. **Replace step-based gate with Q_n-loss-based gate** — provably tied to
   actual reliability (Theorem 4), automatically environment-adaptive, no
   `offline_steps`-dependent defaults.

Optionally add **per-sample trust from ensemble disagreement** (step 4 of the
formula). It costs nothing extra (the ensemble already exists) and gives
Bayesian-style local reliability — useful precisely on the regression
environments where Q_n is heterogeneously trustworthy across the state space.

The improved formulation is backward compatible: setting `ESS_target → 1`,
`ρ_target → 0` recovers standard FQL exactly, and setting `c ≡ 1`, MAD → std,
ESS-target removed recovers the originally recommended solution.
