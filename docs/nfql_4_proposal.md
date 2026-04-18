# Critical assessment of NFQL₃ and a proposal for NFQL₄

> **Revision notes.**
>
> - **Rev 1.** An earlier draft proposed replacing `Q_n(s, x_t, t)` with a
>   vanilla state-value `V_φ(s)`. That defeats the core NFQL intuition:
>   under the rectified-flow linear-interpolant property, a single noised
>   point `x_t` is crossed by many branches with different velocities
>   and different values, so reweighting must happen at the
>   **timestep / flow-point level**, not at the state level. A `V(s)`
>   baseline is state-uniform and cannot express this. Rewritten to
>   keep Q_n and repair the *mechanisms* around it.
> - **Rev 2.** Added §3.3 — a two-t *martingale* bootstrap for Q_n,
>   inspired by Temporal-Difference learning and by Consistency Models.
>   Added §8 (related work).

---

## 1. Why timestep-level weighting is the right idea

The rectified-flow linear-interpolant property means that at a fixed
`(x_t, t)`, multiple dataset actions `a_1, …, a_k` can have
interpolated through the same neighbourhood via different `x_0`. Each
branch carries its own velocity `(a_i − x_0_i)` and its own value
`Q(s, a_i)`. The optimal velocity at `(x_t, t)` is an average over
branches:

```
u*(s, x_t, t) = E[ a − x_0 | s, x_t, t ]
```

Standard BC learns the *unweighted* mean. If we want the learned flow
to concentrate on high-value branches, we apply per-sample weights
`w_i` inside the BC loss, and the fixed point becomes:

```
u*_weighted(s, x_t, t) = E[ w · (a − x_0) | s, x_t, t ]
                        / E[ w           | s, x_t, t ]
```

**Why Q_n(s, x_t, t) is the right baseline, and V(s) is not.**
A V(s)-based advantage is state-uniform: the magnitude of `|Q − V(s)|`
depends only on `s`, not on whether the point `(x_t, t)` is actually in
a regime where multiple branches compete. A `(x_t, t)`-conditional
baseline `Q_n ≈ E[Q(s, a) | s, x_t, t]` makes the advantage magnitude
track **the amount of branch ambiguity at this specific point**:

- Low t, `x_t ≈ noise`: many branches share the neighbourhood;
  `Q_n ≈ V(s)`; `|Q − Q_n|` is spread; weighting is strong.
- High t, `x_t ≈ a_i`: one branch dominates the neighbourhood;
  `Q_n ≈ Q(s, a_i)`; `|Q − Q_n| → 0`; weighting naturally becomes a
  no-op — **correctly**, because there is no competing branch to pull
  the velocity toward.

**NFQL₄ keeps Q_n(s, x_t, t).** Proposition 3.1 in `nfql_3_theory.md`
is correct in spirit: the advantage magnitude *should* scale with
branch ambiguity at the point, and the natural radius is `O(1 − t)`.

---

## 2. Where the current implementation underdelivers

The architecture is right; three mechanisms around it degrade the
signal the architecture is supposed to produce.

### 2.1 The R²_qn gate is calibrated on the trivial regime

With `t ~ U(0, 1)` in `noised_critic_loss`, roughly half the minibatch
asks Q_n to predict `Q(s, a)` from `(s, a, 1)` — a near-identity that
any reasonable network solves in a few thousand steps. The other half
asks Q_n to predict `Q(s, a)` from `(s, noise, 0)` — genuinely hard.

The reported `L_noised` is pulled down by the easy half, so
`R²_qn = 1 − L_noised / Var[Q]` reaches ≈ 0.99 almost immediately. The
dual gate opens because the easy regime is easy, not because Q_n has
become a useful conditional baseline in the low-t regime — which is
where `Q − Q_n` actually differs from plain AWR.

Empirically, all 14 nfql_3 runs hit `r²_qn ≈ 0.99` and
`gate_c ≈ 0.99` from step 5k on.

### 2.2 Q_n training allocates capacity to the regime where it does not matter

Uniform `t ~ U(0, 1)` means half of every gradient step teaches Q_n a
trivial identity mapping and half teaches it a hard conditional
expectation. Gradient pressure concentrates on the easy half. Q_n ends
up accurate where useless (high t) and underconverged where needed
(low t).

### 2.3 Global MAD / ESS-target normalization mixes two regimes

Across a batch with uniform `t`, low-t samples have `|a_local|` of
order `√Var[Q]`, while high-t samples have `|a_local|` near 0. The
global MAD is somewhere between the two, so low-t extremes are
compressed and high-t near-zeros are amplified into spurious weight
variation. This is visible in `bc_weight_mean ≈ 1.0`.

### 2.4 Trust term has ambiguous polarity

`trust = exp(−|Q_n^{(1)} − Q_n^{(2)}| / median(δ))` downweights
samples where the ensemble disagrees. But at low t the conditional
expectation legitimately has high *estimation* variance, so ensemble
disagreement is highest exactly where we want weighting strongest.

---

## 3. Proposal: NFQL₄

Each change targets one pathology in §2 while preserving
`Q_n(s, x_t, t)` and the timestep-level weighting intuition.

### 3.1 Gate on a low-t-restricted R²_qn

Split the Q_n loss by `t` and use only the low-t portion for the EMA
that feeds the R²_qn gate:

```python
t_sq       = t.squeeze(-1)                                  # (B, n)
mask_low   = (t_sq < 0.5).astype(jnp.float32)
sq_err_all = (noised_q - target_q) ** 2                     # (B, n)

noised_critic_loss     = sq_err_all.mean()                  # gradient (unchanged)
noised_critic_loss_low = (
    (sq_err_all * mask_low).sum() / jnp.maximum(mask_low.sum(), 1.0)
)                                                           # gate EMA target
```

Critic-side R²_critic is unchanged.

### 3.2 Bias Q_n's training distribution toward low t

Change the `t` sampling in `noised_critic_loss` (the critic-training t,
not the actor-loss t, which must stay `U(0, 1)` for BC-flow coverage):

```python
# Option A — truncated:
t = jax.random.uniform(t_rng, (B, n, 1), minval=0.0, maxval=0.5)

# Option B — smooth bias toward 0:
u = jax.random.uniform(t_rng, (B, n, 1))
t = u ** 2
```

Option A is the cleanest match to the §3.1 gate.

### 3.3 Two-t martingale bootstrap for Q_n  *(new — your idea)*

**Motivation.** Sampling a single t per interpolant forces Q_n to learn
the hard low-t conditional expectation *only* from the direct MSE
target `Q(s, a_i)`, which at low t is an extremely noisy target (one
branch sampled, many possible branches). This is the same problem
Q-learning faces at high-variance rewards — and the classical fix
(Watkins 1989, Sutton 1988) is to **bootstrap**: target the current
estimate at a *nearby, easier* state, not only the noisy Monte-Carlo
return.

The rectified-flow linear-interpolant structure gives us a natural
"nearby, easier state": the same flow path `(x_0, a)` evaluated at a
*later* timestep `t_b > t_a`. At `t_b` the point `x_{t_b}` is closer
to `a`, so Q_n at `(x_{t_b}, t_b)` has a tighter conditional
distribution and is easier to estimate. And importantly:

**Martingale property (verified below).** Under the joint distribution
over `(x_0, a)` with `t_a < t_b` paired on the same path,

```
E[ Q_n*(s, x_{t_b}, t_b)  |  s, x_{t_a}, t_a ]  =  Q_n*(s, x_{t_a}, t_a)
```

where `Q_n*(s, x_t, t) := E[Q(s, a) | s, x_t, t]`. This is the tower
property applied to the nested information sets "observe path up to
`t_a`" ⊂ "observe path up to `t_b`". So `Q_n(s, x_{t_b}, t_b)` is an
*unbiased* bootstrap target for `Q_n(s, x_{t_a}, t_a)`, exactly as
`r + γ Q(s', a')` is an unbiased target for `Q(s, a)` in TD.

**Method.** For each `(s_i, a_i)` and each interpolant index `j`:

1. Sample `x_0^{(j)} ~ N(0, I)`, `t_a^{(j)} < t_b^{(j)}` both in
   `U(0, 1)` (e.g. `sort(two_uniform_samples)`), **paired on the same
   path**.
2. Form `x_{t_a}^{(j)} = (1 − t_a^{(j)})·x_0^{(j)} + t_a^{(j)}·a_i`
   and `x_{t_b}^{(j)} = (1 − t_b^{(j)})·x_0^{(j)} + t_b^{(j)}·a_i`.
3. Evaluate Q_n at both points.
4. Loss:

```
L_anchor       = ( Q_n(s, x_{t_b}, t_b) − sg[Q(s, a)] )²      # direct regression at the easy end
L_martingale   = ( Q_n(s, x_{t_a}, t_a) − sg[Q_n(s, x_{t_b}, t_b)] )²   # TD-style bootstrap

L_noised_v4    = L_anchor + λ_boot · L_martingale
```

where `sg[·]` is `jax.lax.stop_gradient`. Typical `λ_boot ∈ [0.5, 1.0]`.

Biasing `t_b` toward 1 (e.g. `t_b ~ U(0.5, 1)`) makes the anchor
target a near-copy of `Q(s, a)`; biasing `t_a` toward 0 (e.g.
`t_a ~ U(0, 0.5)`) targets the low-t regime we actually care about.
This subsumes §3.2 — the biased `t_a` sampling replaces the
"uniform-t" default.

**Why this handles "multiple behaviors".** If the behaviour policy at
`s` is multi-modal (two modes `a_1, a_2` with different `Q`), then at
`(x_{t_a}, t_a)` both modes can be compatible flow paths. A single-t
MSE regression pulls `Q_n(x_{t_a}, t_a)` toward whichever `a_i`
happened to generate the sample — slow convergence, high variance,
and at any finite step, Q_n is biased by the most recently seen mode.
The martingale bootstrap instead pulls `Q_n(x_{t_a}, t_a)` toward
`Q_n(x_{t_b}, t_b)`, which itself averages over the modes compatible
with `x_{t_b}` (a tighter — but not trivial — set). As `t_b → 1` the
bootstrap target approaches `Q(s, a)`; for `t_b` in the middle,
`Q_n(x_{t_b}, t_b)` is already a partial mode-average. The resulting
propagation across t is the flow analog of value-iteration sweeping
information backward through the state space.

**Cost.** Two Q_n forwards per minibatch (anchor at `t_b`, bootstrap
at `t_a`) instead of one. The target side uses stop-gradient so only
the anchor forward creates gradients; the bootstrap-target forward is
free to be `jax.lax.stop_gradient`-wrapped entirely.

**Stability.** For robustness, the bootstrap target can be a slow EMA
of Q_n (a "target noised-critic") in the style of DQN / FQL's target
critic. In practice, `stop_gradient(Q_n_current)` is often sufficient
when `λ_boot ≤ 1` because the anchor term at `t_b ≈ 1` provides a
strong direct-supervision anchor to `Q(s, a)` and prevents drift.

**Connection to Consistency Models (Song et al. 2023).** Consistency
Models train `f_θ(x_t, t) → x_0` by enforcing self-consistency along
the probability-flow ODE: `f_θ(x_{t_1}, t_1) ≈ f_θ(x_{t_2}, t_2)` for
adjacent timesteps along the same trajectory, anchored by a boundary
condition at `t = 0`. The construction above is the Q-valued analog:
`Q_n(x_{t_a}, t_a) ≈ Q_n(x_{t_b}, t_b)` along the same flow trajectory,
anchored by the boundary condition `Q_n(s, a, 1) = Q(s, a)`.

### 3.4 Normalize advantages per t-bin, not globally

Compute median / MAD within the low-t subset of the actor batch, so
the `(x_t, t)`-conditional amplitude is preserved:

```python
mask_low = (t.squeeze(-1) < 0.5)                            # (B,)

a_med = masked_median(a_local, mask_low)
β_MAD = 1.4826 * masked_median(jnp.abs(a_local - a_med), mask_low)
β_MAD = jnp.maximum(β_MAD, 1e-6)
a_norm = (a_local - a_med) / β_MAD
```

(JIT-friendly implementations: sort-based quantile with high-t pushed
to `+inf`, or `jnp.quantile` on a mask-weighted histogram. Several
options exist.)

### 3.5 Replace ESS-targeted temperature with fixed-τ + clipping

```python
tau    = 1.0
w_exp  = jnp.exp(a_norm / tau)
w_exp  = w_exp / (jnp.mean(w_exp) + 1e-8)
w_exp  = jnp.clip(w_exp, 0.0, w_max)                        # w_max = 10
```

Drops the 14-step bisection in the hot loop; gradient variance is
bounded by `w_max²` via clipping.

### 3.6 Ablate the trust term — do not keep it by default

Run one env with `trust = 1` vs. current; keep only if it clearly
helps. Default for NFQL₄: `trust = 1`.

### 3.7 Dual gate retained, with the §3.1 R²_qn

```
gate_qn     = σ( (R²_qn_low_t  − r²_target)        / κ_qn     )
gate_critic = σ( (R²_critic     − r²_critic_target) / κ_critic )
c_gate      = gate_qn · gate_critic
```

### 3.8 Final weight

```
u_i = 1 + c_gate · (w_exp_i − 1)
```

No hand-designed `m(t)` mask — the `(x_t, t)`-conditional baseline
produces the taper structurally.

---

## 4. Why this strictly improves NFQL₃

| Pathology in §2 | NFQL₃ | NFQL₄ |
|---|---|---|
| Gate opens on trivial high-t regression | `R²_qn ≈ 0.99` at step ≈ 5k | low-t-only `R²_qn`; gate waits for genuine low-t convergence |
| Q_n capacity wasted on high-t identity map | `t ~ U(0, 1)` | biased `t_a` toward 0, `t_b` toward 1 |
| Q_n at low t learns from one-sample MC only | yes | **martingale bootstrap** from easier `t_b` |
| Multi-modal behaviour handling | single-sample MC pulls toward whichever mode was drawn | `Q_n(t_a)` averages via bootstrap from partially-resolved `Q_n(t_b)` |
| Global MAD mixes low/high-t regimes | yes | per-bin MAD |
| ESS bisection inside JIT | yes (14-step while-loop) | removed |
| Trust polarity suppresses signal at low t | possible | ablate, default off |
| **Timestep-level weighting intuition** | architecturally yes, attenuated in practice | architecturally yes and realized in the gradient |

---

## 5. Minimal-patch option

The two highest-value single changes, in order:

1. **§3.1 low-t gate** (10 lines): decouples the gate from the trivial
   high-t regime.
2. **§3.3 martingale bootstrap** (≈ 25 lines in `noised_critic_loss`):
   gives Q_n a principled learning signal at low t instead of the
   one-sample MC target.

Both can be applied without touching the actor loss, MAD, ESS, or
trust.

---

## 6. Recommended next diagnostics (before committing to NFQL₄)

### 6.1 Bucketed R²_qn by t

```python
var_q = jnp.var(original_q) + 1e-8
for b_lo, b_hi in [(0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)]:
    mask = ((t_sq >= b_lo) & (t_sq < b_hi)).astype(jnp.float32)
    mse_bin = (sq_err * mask).sum() / jnp.maximum(mask.sum(), 1.0)
    info[f'noised_critic/R2_t_{b_lo:.2f}_{b_hi:.2f}'] = 1.0 - mse_bin / var_q
```

**Prediction:** `R²_t_[0.00, 0.25] ≪ 0.75` throughout training while
`R²_t_[0.75, 1.00] ≈ 0.99` from step 5k. Confirms §3.1, §3.2, §3.3.

### 6.2 Bootstrap-vs-MC residual split

Once §3.3 is in, log `L_anchor` and `L_martingale` separately. The
bootstrap loss should decrease faster than the low-t portion of the
original MC loss, reflecting the TD-style variance reduction.

### 6.3 Mode-count probe

For a small held-out set of states, sample many `x_0` values and run
the one-step flow to get `â_j ~ π_θ(·|s)`. Cluster the `â_j` to
estimate mode count. On multi-modal states, §3.3 should yield a
Q_n(x_t, t) that — as `t` ranges from 0 → 1 — smoothly transitions
from a *mode-averaged* value at low t to a *specific-mode* value at
high t. Under the original NFQL₃ this transition is noisier because
low-t Q_n is trained only on one-sample MC.

---

## 7. One-line summary

The `(x_t, t)`-conditional baseline is the **right** architectural
choice for the intuition that a noised point is crossed by multiple
valued branches. NFQL₃'s surrounding mechanisms — gate metric,
training-t distribution, and advantage normalization — are dominated
by the trivial high-t regime where the intuition already says weights
should be ≈ 1. NFQL₄ keeps Q_n, focuses each mechanism on the low-t
regime, and adds a two-t martingale bootstrap so Q_n at low t learns
from easier targets at higher t instead of from the high-variance
one-sample MC alone.

---

## 8. Related work

**Flow matching and rectified flow.**
Lipman et al. (2023, *Flow Matching for Generative Modeling*) and Liu
et al. (2022, *Flow Straight and Fast: Rectified Flow*) establish the
linear-interpolant parameterization this repo builds on. The key
property used throughout §1 and §3.3 — that
`Q_n*(s, x_t, t) := E[Q(s, a) | s, x_t, t]` is a martingale in t along
the flow path — is a direct consequence of the linear-interpolant
joint distribution in those papers.

**Consistency models.**
Song et al. (2023, *Consistency Models*) train
`f_θ(x_t, t) → f_θ(x_{t'}, t')` along the probability-flow ODE,
anchored by a boundary condition. The NFQL₄ two-t bootstrap (§3.3)
is the Q-valued analog: same along-trajectory consistency, anchored
at `t = 1` to `Q(s, a)` instead of at `t = 0` to `x_0`.

**TD and n-step bootstrapping.**
Watkins (1989, *Q-learning*) and Sutton (1988, *Learning to Predict
by the Methods of Temporal Differences*) establish the bias-variance
trade-off between Monte-Carlo regression (low bias, high variance)
and single-step bootstrap (some bias, much lower variance). The §3.3
construction uses a *continuous-t* variant: `t_a` is the "current"
time, `t_b ∈ (t_a, 1]` is the "future" time on the same flow path,
and bootstrapping across `(t_a → t_b)` trades a controllable amount
of target-Q bias for large variance reduction.

**Flow / diffusion policies for RL.**
Wang et al. (2022, *Diffusion-QL*) and Hansen-Estruch et al. (2023,
*IDQL*) use Q-weighted diffusion policies but reweight at the
`(s, a)` level, not at the `(x_t, t)` level — so they cannot express
the branch-ambiguity-dependent taper discussed in §1. Psenka et al.
(2023, *Q-Score Matching*) directly shape the score with `∇_a Q`
instead of reweighting, an orthogonal approach. FQL (Park et al.
2024/2025, the baseline in this repo) does not reweight BC at all;
NFQL / NFQL₂ / NFQL₃ / NFQL₄ are the progression of this repo's
`(x_t, t)`-level reweighting line.

**Implicit and progressive distillation.**
Salimans & Ho (2022, *Progressive Distillation for Fast Sampling of
Diffusion Models*) and Meng et al. (2023, *On Distillation of Guided
Diffusion Models*) iteratively compress k-step flows into (k/2)-step
flows by enforcing two-t agreement along the flow. This is
architecturally similar to §3.3 but trains the *velocity field*
itself, not a Q-valued auxiliary — the two-t consistency idea is the
common thread across both lines.
