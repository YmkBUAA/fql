# FQL-AR — Flow-Anchored Reweighted BC

`agents/fql_ar.py`. Variant of FQL whose BC flow loss is reweighted by a
**flow-local, sample-anchored advantage**:

```
a' := ODE(actor_bc_flow ; from x_t = (1-t) eps + t a, t -> 1)
Delta(s, a; t) := Q_target(s, a) - Q_target(s, a')
w(s, a; t)     := softmax_{tau*}( MAD-norm( Delta ) )
L_BC           := E[ w(s, a; t) * || v_theta(s, x_t', t') - (a - eps') ||^2 ]
                  with (eps', t') sampled INDEPENDENTLY of (eps, t)
```

Everything else (critic loss, distillation, Q loss, target updates,
sampling) is identical to `agents/fql.py`.

The point of the design is to obtain a baseline that:

- is defined by the BC flow itself (no V network, no Q_n network),
- is anchored at each individual data action a (each sample has its own
  baseline a'), and
- has a continuous knob t controlling how local that baseline is.

## Why a flow-local baseline

Existing baselines for advantage-weighted BC differ in what they compare
Q(s, a) against:

| agent          | baseline                        | sample-anchored | flow-native |
| -------------- | ------------------------------- | --------------- | ----------- |
| FQL            | (none — uniform BC weight)      | -               | -           |
| FQL-V          | V(s) = E_{a' ~ pi}[Q(s, a')]    | no              | partial     |
| NFQL_5..8      | Q_n(s, x_t, t) (noised critic)  | no              | yes (collapses) |
| **FQL-AR**     | Q(s, a') with a' = denoise(noise(a), t) | **yes**         | **yes**     |

FQL-V's V is a *global* baseline — averaged over fresh BC samples, it
discards the identity of a. NFQL's Q_n was supposed to be a flow-time
conditional expectation, but its training target Q(s, a) is constant in
(x_t, t), so Q_n collapses to Q (R^2_t ~= 0.99 in every flow-time bucket;
documented in `docs/nfql_5_results.md`).

FQL-AR builds a baseline that is

- *Sample-anchored*: a' is generated from a perturbation of a, not from
  fresh noise. The pair (a, a') is correlated, which lets it act as a
  control variate.
- *Flow-native*: a' is defined only via the actor_bc_flow ODE. Replacing
  the flow with a Gaussian policy collapses a' to E[a | s] and removes
  the t-dependence — the flow structure is load-bearing.
- *Self-regulating*: when actor_bc_flow already reproduces a (a' ~= a),
  Delta vanishes and the BC weight is uniform. The agent does not pay a
  penalty for being on the BC mode in the offline phase.

## Algorithmic flow

Per training step (one batch of (s, a, r, s', mask, online_flag)):

```
1. Critic step
   target_q       = r + gamma * mask * Q_target(s', sample_actions(s'))
   critic_loss    = MSE(Q(s, a), target_q)
   var_target_q   = Var(target_q)                    # for the R^2 gate

2. Build the FlowAR advantage
   eps  ~ N(0, I)                                    (B, action_dim)
   t    ~ U(adv_t_lo, adv_t_hi)                      (B, 1)
   x_t  = (1 - t) * eps + t * a
   a'   = ODE_Euler(actor_bc_flow ; x_t -> t = 1, n = adv_flow_steps)
   a'   = stop_grad(clip(a', -1, 1))
   Delta = stop_grad( Q_target(s, a) - Q_target(s, a') )

3. Normalize and softmax with ESS = 0.7
   d_med   = median(Delta)
   beta    = 1.4826 * MAD(Delta)
   d_norm  = (Delta - d_med) / max(beta, 1e-6)
   tau*    = bisect( log_tau s.t. ESS(d_norm / tau) = ess_target )
   w_exp   = softmax(d_norm / tau*) * B               # mean-1 weights

4. Critic R^2 reliability gate
   r2          = 1 - critic_loss_ema / var_tq_ema     # (EMA-smoothed)
   gate_critic = sigmoid( (r2 - r2_critic_target) / gate_kappa_critic )
   gate_c      = gate_critic * (online OR not weighted_bc_online_only)
   bc_weights  = stop_grad( 1 + gate_c * (w_exp - 1) )

5. Reweighted BC flow loss (independent CFM samples)
   for k = 0 .. n_actor_time_samples-1:
       eps_k  ~ N(0, I)
       t_k    ~ U(0, 1)
       x_t_k  = (1 - t_k) * eps_k + t_k * a
       target = a - eps_k
       loss_k = || actor_bc_flow(s, x_t_k, t_k) - target ||^2
   bc_flow_loss = mean( bc_weights[:, None] * mean_k(loss_k) )

6. Distillation + Q loss (unchanged from FQL)
   noise           ~ N(0, I)
   target_flow_a   = compute_flow_actions(s, noise)        # full ODE
   onestep_a       = actor_onestep_flow(s, noise)
   distill_loss    = MSE(onestep_a, target_flow_a)
   q_loss          = -mean( Q(s, clip(onestep_a)) )

7. Total loss and updates
   loss = critic_loss + bc_flow_loss + alpha * distill_loss + q_loss
   apply gradients; soft target_critic update; EMA update of
   critic_loss_ema and var_tq_ema for the gate.
```

Two sources of stochasticity in step 2 / step 5 are kept **independent on
purpose**: (eps, t) decides the sample's BC weight; (eps_k, t_k) decides
the gradient direction. Sharing them would couple weight magnitude with
gradient direction at the same noise vector and bias the learned flow.

## Why this is offline-safe

a' is the BC flow's prediction. Q is queried at (s, a) — in-data — and at
(s, a') — predicted by the flow. a' lies on the BC manifold by
construction: it is the terminal of an ODE that lives entirely under the
BC flow. So Q is queried only on points the BC flow itself would produce.

This is the same safety profile as IQL / AWR / FQL-V. The agent never
trains v_theta toward an action it generated from a non-data
distribution; the policy gradient surface for the BC flow stays
data-anchored. Bootstrap-error cycles (where Q's overestimation on
out-of-data actions feeds back into more out-of-data sampling) are
broken because:

- the critic update uses sample_actions(s'), which itself comes through
  the BC-anchored one-step flow,
- the BC reweighting only changes per-sample magnitudes, never targets,
- a' is generated by the BC flow, not a separately optimized policy.

## Behaviour by phase

| phase    | typical Delta              | typical bc_weights         | net effect on BC |
| -------- | -------------------------- | -------------------------- | ---------------- |
| Offline (start) | ~ 0 everywhere     | ~ 1 (gate closed early)    | unweighted CFM   |
| Offline (mid)   | small, mixed sign  | mildly non-uniform         | mild reweight    |
| Online (start)  | systematically > 0 | high on improved actions   | distill best-of-n |
| Online (mid)    | shrinks as policy converges | back toward uniform | self-regularizing |

The "online" mechanism is the one we care about: best-of-n action
selection puts argmax-Q actions into the replay buffer. When those
actions are sampled as a, the BC flow's denoise of a perturbed copy of a
pulls back toward the BC mode (a' is close to BC(s) instead of best-of-n
choice), so Q(s, a) > Q(s, a') systematically, Delta > 0, weights up. The
BC flow then absorbs the best-of-n discoveries into its own distribution,
without an explicit distillation target on best-of-n actions.

When an action does worse than its BC denoise (Delta < 0), it is
*downweighted* in BC training, but never replaced by a' as a target. The
information that a' was better is preserved for the next sampling step:
the BC flow does not overwrite its existing local mode at a (because BC
weight is low, gradient pressure is low), and on subsequent rollouts
best-of-n samples in that neighborhood will surface in the buffer as new
data. This is the offline-to-online absorption loop, mediated by Q-as-gate
rather than Q-as-target.

## Hyperparameters specific to FQL-AR

```
adv_t_lo            = 0.4   # noise level lower bound for advantage path
adv_t_hi            = 0.7   # noise level upper bound for advantage path
adv_flow_steps      = 3     # Euler steps for partial denoising
ess_target          = 0.7   # ESS targeted by tau* bisection
weighted_bc_online_only = True
r2_critic_target    = 0.5
gate_kappa_critic   = 0.05
gate_ema_decay      = 0.999
n_actor_time_samples = 4    # CFM samples per state (BC loss)
```

The two FQL-AR-specific knobs are `adv_t_lo`, `adv_t_hi`. Choosing them:

- `t -> 0` makes a' ~= a (perturbation collapsed by the flow), Delta ~= 0,
  no signal. **Lower bound 0.4** keeps the perturbation outside the
  flow's near-linear recovery regime.
- `t -> 1` makes x_t ~= eps (pure noise), a' becomes an ordinary BC
  marginal sample, equivalent to single-sample FQL-V with high variance.
  **Upper bound 0.7** keeps a' anchored to a's neighborhood.
- `adv_flow_steps = 3` is the cheapest setting that visibly differs from
  Euler-1; raising to 10 changes <2% in our internal microbench.

The advantage signal is sensitive to both bounds and somewhat insensitive
to `adv_flow_steps` once it is >= 3. A `t`-sweep ablation is the natural
Phase-1 experiment and the source of the paper's signature `t -> success`
curve.

## Theoretical hook (control variate / variance reduction)

For an exact-expectation reweighting target

    g(s, a) = Q(s, a) - b(s, a)

the per-sample variance of Delta is

    Var[g] = Var[Q] + Var[b] - 2 Cov[Q, b]

so any baseline b with Cov[Q, b] > 0 reduces variance over Var[Q] alone.

| baseline      | Cov[Q(s, a), b]                           |
| ------------- | ----------------------------------------- |
| 0 (raw AWR)   | 0                                         |
| V(s)          | only via shared s; no a-dependent coupling |
| Q(s, a')      | flow-local + s-shared; **strictly larger** when a' close to a |

For Q(s, ·) L-Lipschitz on the geodesic from a to a', a standard control
variate bound gives

    Var[ Q(s, a) - Q(s, a') ] <= L^2 * E[ ||a - a'||^2 ]

which is O(t^2 Var[eps]) when actor_bc_flow has bounded velocity (the BC
flow's deterministic structure makes ||a - a'|| -> 0 as t -> 0 along the
recovery direction). This gives FQL-AR a continuous variance-reduction
parameter t that V-baseline AWR does not have.

This is the hook for the variance reduction theorem the paper would
formalize.

## Diagnostics

Check these on wandb to confirm FQL-AR is not silently degenerating:

| metric                            | target band | meaning                          |
| --------------------------------- | ----------- | -------------------------------- |
| `actor/flowar/dist_a_aprime_p50`  | 0.10 - 0.40 | a' is local but not collapsed    |
| `actor/flowar/delta_std`          | > 0.05      | advantage signal has spread      |
| `actor/flowar/frac_a_better`      | offline ~0.5, online > 0.6 | data action vs BC mode |
| `actor/bc_weight_std` (online)    | > 0.20      | reweighting is meaningfully active |
| `actor/ess_achieved`              | ~0.70       | ESS bisection healthy            |
| `actor/gate_critic` (after warmup)| > 0.5       | critic R^2 above target          |

Failure modes and what they look like:

- `dist_a_aprime_p50 < 0.05` from the start: BC flow is absorbing the
  perturbation; advantage signal will be zero. Increase `adv_t_lo` or
  `adv_t_hi`.
- `dist_a_aprime_p50 > 0.6`: perturbation is too aggressive; a' has
  drifted off a's neighborhood and the baseline degenerates to single-
  sample fql_v (high variance). Decrease `adv_t_hi`.
- `delta_std` low at all phases: the BC flow is unimodal at this state
  family, no genuine alternative completion exists. FlowAR cannot help on
  this env.
- `bc_weight_std` < 0.05 in online: reweighting effectively off; check
  `gate_critic` (may be closed) and `weighting_active`.

## Interaction with FQL pieces (sanity points)

- The distillation and Q loss are FQL's, untouched. `actor_onestep_flow`
  is trained the same way and used by `sample_actions` for environment
  interaction and by `critic_loss` for the bootstrap action.
- Critic ensemble is 2-Q `Value` (the FQL default). Min vs mean is
  selected by `q_agg`; both branches of the FlowAR advantage use the same
  aggregation as the TD bootstrap so the baseline is not silently biased
  upward (this is the inconsistency we flagged earlier in `acfql_v.py`
  and avoided here).
- `weighted_bc_online_only=True` keeps the offline phase as plain FQL.
  Reweighting only kicks in once the online flag flips. This isolates
  the FlowAR contribution to the online phase, where best-of-n produces
  the buffer asymmetry the method exploits.
- `gate_kappa_critic=0.05`, `r2_critic_target=0.5`, `gate_ema_decay=0.999`
  are inherited from `fql_v`. Don't tune these without re-running the
  fql_v baseline alongside, or the comparison is confounded.

## Differences from `fql_v` (line-by-line summary)

- Drop `value` head, `value_loss`, `value_loss_weight`, `n_v_samples`.
- Replace `a_local = original_q - v` (a state-only baseline) with
  `delta = q_a - q_aprime` (sample-local baseline via partial denoising).
- Add `_roll_flow_from(observations, x_t, t_start, n_steps)` helper.
- Add config fields `adv_t_lo`, `adv_t_hi`, `adv_flow_steps`.
- Add diagnostics under the `actor/flowar/*` namespace.
- Use `target_critic` for both Q(s, a) and Q(s, a') in the advantage,
  consistent with TD aggregation under `q_agg`.

Everything else — gate, ESS bisection, EMA tracking, distillation, target
update, sample paths — is preserved.

## Phase-1 verification recipe (single seed, 4-6 hours)

1. Run on `cube-double-noisy-singletask-task3-v0` (the env where
   plain FQL stalls at 0 and fql_v / nfql_7 reach 1.00 only after 1.4M
   steps). Use `seed=0`, default config.
2. After first 100k offline steps, check
   `actor/flowar/dist_a_aprime_p50` is in [0.10, 0.40]. If outside, abort
   and adjust `adv_t_*`.
3. After 1.0M (offline ends, online begins), check
   `actor/flowar/frac_a_better` jumps above 0.6 (online buffer carries
   improved actions).
4. By 1.2M, check `actor/bc_weight_std > 0.20` and
   `actor/flowar/delta_mean > 0` consistently. If yes, FlowAR is
   delivering its core mechanism.
5. By 2.0M, compare final `evaluation/success` against fql_v on the
   same env / seed. The ambition is **lift-off >= 200k steps earlier**
   than fql_v, with comparable or higher final success.

If steps 2-4 pass but step 5 does not show a lift, the empirical case
for FQL-AR is weaker than fql_v even though the mechanism is engaged —
this is the right moment to add the CFG-style sampling layer (Phase 2)
or to retire the line.
