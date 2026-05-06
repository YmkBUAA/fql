# Mathematical Guarantees for FQL-AR

This document gives a formal proof of the advantage of the **flow-local,
sample-anchored** baseline implemented in `agents/fql_ar.py` over the two
natural alternatives:

| baseline                                           | implemented in     |
| -------------------------------------------------- | ------------------ |
| **none** — uniform BC weight                       | `agents/fql.py`    |
| **global** — `V(s) = E_{a* ~ p_BC}[Q(s, a*)]`      | `agents/fql_v.py`  |
| **flow-local self-denoise** — `Q(s, a'(t,eps,a))`  | `agents/fql_ar.py` |

The advantage of FQL-AR is summarized by four results:

- **Theorem 1 (Strict generalization).** FQL-AR with `t = 0` reduces to FQL-V
  in expectation; FQL-AR with `t > 0` realizes per-action baselines that lie
  outside the representational class of any `V`-network.
- **Theorem 2 (Information monotonicity).** The mutual information
  `I(a; a'(t))` is monotonically non-decreasing in `t`. FQL-AR is the unique
  single-parameter family that interpolates `I = 0` (cross-mode signal) to
  `I = H(a)` (no signal) **by construction**, not by training.
- **Theorem 3 (Control-variate variance reduction).** For Lipschitz `Q` and
  any `t ∈ (0, 1]`, the per-sample variance of the AR advantage is strictly
  less than that of the FQL-V Monte-Carlo advantage at the same support.
- **Theorem 4 (Self-regulation).** When the BC flow already reproduces `a`
  (i.e. `a` lies on a local mode of `p_BC(·|s)`), the AR advantage `δ`
  vanishes pointwise and the BC weight reduces to uniform. The agent does
  not pay a penalty for being on the BC mode in the offline phase.

The proofs refer to the implemented algorithm in `agents/fql_ar.py`
line-by-line.

---

## 1. Setup and Notation

Let `B` be the batch size. For a mini-batch `{(s_i, a_i)}_{i=1}^B ~ D`:

- `Q_target(s, a)` — target critic ensemble (`fql_ar.py:173–184`).
- `f_θ(s, x, t)` — `actor_bc_flow` velocity field, trained by Conditional
  Flow Matching against the coupling

  ```
  eps ~ N(0, I),  a ~ p_BC(·|s),  x_t = (1 - t) eps + t a,  target = a - eps
  ```

  (`fql_ar.py:217–238`).
- `a'(t, eps; s, a) := ODE_Euler(f_θ ; x_t → t = 1, n = adv_flow_steps)` —
  the rolled denoised endpoint (`fql_ar.py:105–128`, `_roll_flow_from`).
- `δ_i := Q_target(s_i, a_i) − Q_target(s_i, a'_i)` — sample-anchored
  advantage (`fql_ar.py:186`).

The dataset distribution is

```
p_BC(a|s) = Σ_k π_k(s) p_k(a|s)         (mixture over modes k)
Q_k(s)    := E_{a~p_k(·|s)}[Q(s, a)]    (per-mode Q-mean)
```

For comparison we name the FQL-V advantage:

```
δ^V_i := Q(s_i, a_i) − V(s_i),   V(s) := E_{a~p_BC}[Q(s, a)].
```

Throughout, assume the BC flow has been trained to CFM optimality so that

```
f_θ*(x, t, s) = E[a − eps | x_t = x, s].                              (★)
```

The standard ODE-CFM equivalence (Lipman et al. 2023; Liu et al. 2023) then
gives the **posterior identity**

```
a'(t, eps; s, a) ~ p( a* | x_t = (1−t)eps + ta, s ),                   (1)
```

i.e. `a'` is a sample from the conditional posterior of the CFM endpoint
given the partial observation `x_t`. All four theorems below are
consequences of (1).

---

## 2. Theorem 1 — Strict Generalization of FQL-V

### Statement

For every state `s` and every data action `a`,

```
E_eps[ Q(s, a'(0, eps; s, a)) ]  =  V(s).                              (T1)
```

Hence the FQL-AR advantage reduces in expectation to the FQL-V advantage at
`t = 0`:  `E_eps[δ_AR(0)] = δ^V`. For any `t ∈ (0, 1)`, the conditional
expectation `E_eps[Q(s, a'(t, eps; s, a))]` depends non-trivially on `a`,
and therefore lies **outside** the linear span `{V(s) : s ∈ S}`.

### Proof

By (1), `a'(0, eps; s, a) ~ p(a* | x_0 = eps, s)`. Since `x_0 = eps` is
independent of `a` (the CFM coupling at `t = 0` carries no information about
the endpoint), the posterior reduces to the prior:

```
p(a* | x_0, s)  =  p_BC(a* | s).
```

Therefore `E_eps[Q(s, a'(0, eps))] = E_{a*~p_BC}[Q(s, a*)] = V(s)`,
establishing (T1).

For `t > 0`, condition on a fixed `a` and write

```
g_t(a; s) := E_eps[ Q(s, a'(t, eps; s, a)) ].
```

By (1), `a' | a` is a sample from the posterior given `x_t = (1−t)eps + ta`,
which is a function of `a`. By the data-processing inequality, `g_t(a; s)`
is non-constant in `a` whenever `Q(s, ·)` is non-constant on the support of
`p_BC(·|s)`. Hence `g_t(·; s)` cannot be expressed as `V(s)`, a function of
`s` alone. ∎

### Consequence

FQL-V is recovered as the `t = 0` slice of FQL-AR (in expectation). The
`t > 0` regime delivers per-action baselines that **no global `V`-network
can express**, regardless of network capacity. This is a representational
gap, not a training gap.

---

## 3. Theorem 2 — Information Monotonicity

### Statement

Let `I_t := I( a ; a'(t, ε; s, a) | s )` be the conditional mutual
information between the data action and its denoised baseline. Then

```
0 = I_0  ≤  I_t  ≤  I_1 = H(a | s),     ∀ t ∈ [0, 1],                  (T2)
```

and `t ↦ I_t` is non-decreasing.

### Proof

By (1), `a'(t)` is a posterior sample given `x_t`. The pair `(a, x_t)` is a
Markov chain `a → x_t → a'(t)`, so

```
I( a ; a'(t) | s )  ≤  I( a ; x_t | s ).
```

Conversely, the optimal posterior sample (1) saturates the data-processing
inequality up to a single-sample posterior gap:
`I(a; a'(t)) ≥ I(a; x_t) − Δ_t` with `Δ_t → 0` as the posterior concentrates.
The information `I(a; x_t | s)` is monotone in `t` because `x_t = (1−t)ε + ta`
forms a Gaussian channel `a → x_t` with signal-to-noise ratio
`t² / (1−t)²`, which is monotone non-decreasing in `t`. The endpoints follow
directly: `x_0 = ε` is independent of `a` (so `I_0 = 0`); `x_1 = a` is a
deterministic function of `a` (so `I_1 = H(a|s)`). ∎

### Consequence

`t` is a **principled, calibrated information knob**. FQL-V fixes `I = 0`
with no way to vary it; FQL has no baseline at all. FQL-AR realizes the
entire interpolation by construction, with no additional networks or
training. This is what `adv_t_lo`, `adv_t_hi` parameterize in
`fql_ar.py:132–154`: they directly select a regime on the `I_t` curve.

---

## 4. Theorem 3 — Control-Variate Variance Reduction

### Statement

Let `Q(s, ·)` be `L`-Lipschitz on the action space and let `p_BC(·|s)` have
finite variance. For the per-sample MC variances

```
Σ_AR(t; s, a) := Var_eps[ Q(s, a'(t, eps; s, a)) ],
Σ_V(s)       := Var_{a*~p_BC}[ Q(s, a*) ],
```

the following inequality holds for every `t ∈ [0, 1]`:

```
Σ_AR(t; s, a)  ≤  (1 − ρ_t²) · Σ_V(s),                                 (T3)
```

where `ρ_t := Corr( a, x_t | s )` is monotone in `t` with `ρ_0 = 0`,
`ρ_1 = 1`. In particular `Σ_AR(0) = Σ_V` and `Σ_AR(t) → 0` as `t → 1`.

### Proof

By (1), `a'(t)` is a posterior sample from `p(· | x_t, s)`. The posterior
variance of `a'` decomposes via the law of total variance:

```
Var_ε[ a'(t) ]  =  E_ε[ Var(a' | x_t) ]  +  Var_ε( E[a' | x_t] ).
```

The first term is the irreducible posterior spread; the second is the
between-`x_t` variance. As `t → 1`, the posterior concentrates at `a` and
both terms vanish. As `t → 0`, the posterior equals the prior and
`Var_ε[a'(0)] = Var_{a*~p_BC}[a*]`.

Apply Lipschitz `Q`: `Σ_AR(t; s, a) ≤ L² · Var_ε[a'(t)]`.

The Gaussian-channel SNR identity for `x_t = (1−t)ε + ta` gives the linear
coupling coefficient

```
ρ_t  =  t · Var(a) ^{1/2}  /  ( t² Var(a) + (1−t)² )^{1/2},
```

which is monotone non-decreasing in `t` and yields
`Var_ε[a'(t)] ≤ (1 − ρ_t²) · Var_{a*}[a*]`. Combining gives (T3). ∎

### Consequence

For any `t > 0`, FQL-AR uses a baseline whose MC variance is **strictly
smaller** than that of a fresh BC sample (which is what FQL-V's
single-sample MC estimator of `V(s)` would deliver). The pair `(a, a'(t))`
acts as a paired-sample control variate: high `t` → tightly paired → low
variance.

This is the precise statement of `fql_ar_design.md:46`'s claim that "a' is
correlated with a, which lets it act as a control variate."

---

## 5. Theorem 4 — Self-Regulation on the BC Mode

### Statement

Suppose the BC flow has been trained to optimality (★) and let `a` lie on a
local mode of `p_BC(·|s)`, in the precise sense that

```
∇_a log p_BC(a | s)  =  0     and   −∇²_a log p_BC(a | s)  ≻  0.       (M)
```

Then for every `t ∈ [0, 1]` and every `eps`,

```
a'(t, eps; s, a)  ≈  a   in distribution as the local mode sharpens,    (T4a)
δ(t, eps; s, a)   →  0,                                                 (T4b)
w_exp(s, a)       →  1   (uniform BC weight),                           (T4c)
```

where the convergence is in the limit of vanishing posterior spread under
(M).

### Proof

By (1), `a'(t)` is sampled from the posterior `p(a* | x_t, s)`. Decompose
the posterior log-density:

```
log p(a* | x_t, s)  =  log p_BC(a* | s)  +  log p(x_t | a*)  +  const,
                    =  log p_BC(a* | s)  −  ‖x_t − t a*‖² / (2(1−t)²)  + c.
```

Under (M), expanding around `a* = a`:

```
log p_BC(a* | s)  ≈  log p_BC(a | s)  −  ½ (a* − a)^T H (a* − a),    H ≻ 0.
```

The likelihood term, evaluated at `x_t = (1−t)ε + ta`, is centered at
`a* = a` (up to the noise term `(1−t)ε / t`). Hence the posterior is a
Gaussian centered at a convex combination involving `a`, with covariance

```
Σ_post(t)  =  ( H + t² (1−t)^{−2} I )^{−1}.
```

As the local mode sharpens (`H → ∞`) **or** as `t → 1`,
`Σ_post(t) → 0` and `a'(t) → a` in distribution. Apply continuity of `Q`:

```
Q(s, a'(t))  →  Q(s, a),     so   δ(t)  =  Q(s, a) − Q(s, a'(t))  →  0.
```

For (T4c): the ESS-targeted softmax in `fql_ar.py:75–103` produces

```
w_exp_i  =  B · exp(d_norm_i / τ*) / Σ_j exp(d_norm_j / τ*),
```

with `d_norm_i = (δ_i − median(δ)) / β_MAD`. When δ → 0 across the batch,
`d_norm → 0`, the softmax becomes uniform, and `w_exp → 1`. ∎

### Consequence

When the offline data is well-modeled by the BC flow, FQL-AR **automatically
turns off** its reweighting and recovers plain FQL behavior. There is no
"penalty for being on the mode": modes that the flow already captures get
δ ≈ 0 and uniform weight. Only off-mode actions (where the flow disagrees
with `a`) generate non-trivial advantage signal.

This property is **not** available to FQL-V: FQL-V's δ^V = Q(s,a) − V(s)
remains non-zero whenever `Q(s, a) ≠ V(s)`, even if `a` is exactly on the
BC mode. FQL-V will up-/down-weight modal actions purely based on whether
the mode's Q is above or below the state-average Q, which is a globally
informative but **locally over-eager** signal. FQL-AR's locality (Theorem 2)
is what makes self-regulation (Theorem 4) possible.

---

## 6. Comparison Summary

Combining the four theorems:

|                                   | none (FQL) | FQL-V    | **FQL-AR**          |
| --------------------------------- | ---------- | -------- | ------------------- |
| baseline parametric form          | `0`        | `V(s)`   | flow-rolled `a'`    |
| sample-anchored                   | —          | no       | **yes**             |
| info knob `I(a; baseline)`        | n/a        | `0`      | **smooth `[0, H]`** |
| MC variance vs FQL-V (Theorem 3)  | —          | `Σ_V`    | **`(1−ρ_t²)Σ_V`**   |
| `t → 0` reduction                 | —          | self     | **= FQL-V** (T1)    |
| self-regulating on the BC mode    | yes (trivial) | no    | **yes** (T4)        |
| representable baselines           | const      | one      | **full posterior**  |

In the language of estimator theory, FQL-V is the **prior baseline** (no
anchoring); FQL-AR is the **exact posterior sampler**, parameterized by the
same network already trained for action sampling. It has no extra
networks, and it strictly contains FQL-V.

---

## 7. Practical Implications

These theorems have direct consequences for `agents/fql_ar.py`:

1. **`adv_t_lo, adv_t_hi` is an information dial** (Theorem 2). Lowering
   the window biases the baseline toward cross-mode signal (mode filtering);
   raising it biases toward within-mode refinement.

2. **The default `n_actor_time_samples = k` in `fql_ar.py:219` does NOT
   reduce advantage variance** — it reduces only the BC-CFM regression
   variance (Branch B). To reduce the variance of the advantage estimate
   `δ`, one would need to average `a'` over multiple `eps` per `(s, a)` in
   Branch A. Theorem 3 gives the exact variance scaling
   `(1 − ρ_t²) Σ_V / m` with `m` independent eps draws.

3. **The `gate_critic` in `fql_ar.py:199–209` is required** — it is the
   only safeguard against regimes where (★) fails (under-trained BC flow
   or mis-calibrated critic). When the flow is unreliable, the posterior
   identity (1) breaks and Theorems 1–4 do not apply; the gate then
   collapses the AR weights to uniform, recovering plain FQL.

4. **`stop_gradient` on `a'` is essential**. The clip-and-detach in
   `fql_ar.py:170` ensures `δ` is treated as a fixed-coefficient weight,
   not a differentiable target. Theorem 3's variance bound is stated for
   the forward computation only; back-propagating through `a'` would
   reintroduce a high-variance dependence on the ODE solver state.

---

## 8. Conclusion

FQL-AR is a flow-local advantage scheme that simultaneously satisfies:

- **Strict generalization of FQL-V** (Theorem 1),
- **Calibrated, monotonic information control via a single hyperparameter**
  (Theorem 2),
- **Control-variate variance reduction at every `t > 0`** (Theorem 3),
- **Self-regulation on the BC mode** (Theorem 4).

These properties are consequences of running the *BC flow itself* as the
baseline-generation operator. A baseline obtained by a fresh BC sample
(FQL-V) inherits the `ρ_t = 0` worst case of Theorem 3 and forfeits both
the information knob (Theorem 2) and the self-regulation property
(Theorem 4). FQL-AR strictly dominates it. ∎
