# NFQL₅ Design Spec — V-anchored Conditional Q_n

**Date:** 2026-04-19
**Context:** Successor to NFQL₄, which empirically tied NFQL₃ (mean Δ=+0.013
across 15 envs) while its `(x_t, t)`-conditional baseline Q_n was confirmed to
trivialize to `Q(s, a_i)` regardless of t (martingale_loss/anchor_loss ≈ 0.009;
all four r2_t buckets converge together at ≈ 0.998). See
`docs/nfql_4_results.md`.

## 1. Goal

Make Q_n realize the true `(x_t, t)`-conditional expectation

```
Q_n*(s, x_t, t) = E_{a ∼ p(a | s, x_t, t)}[Q(s, a)]
```

instead of collapsing to the single sampled Q(s, a_i). This is achieved by
pinning Q_n at *both* endpoints of the flow:

- **t = 1:** Q_n(s, a, 1) → Q(s, a)   (kept from NFQL₃)
- **t = 0:** Q_n(s, x_0, 0) → V(s) = E_a[Q(s, a)]   (NEW)

With both boundaries fixed, the uniform-t interior samples force Q_n to
interpolate sensibly and cannot reach the trivial fixed point Q_n ≡ Q(s, a_i).

## 2. Architecture

Start from `agents/nfql_3.py`. Add one sub-module (`value`) to the main
`ModuleDict` and add one new loss (`value_loss`). Modify `noised_critic_loss`
to include a V-anchor at t=0.

### Network additions

| Name | Type | Shape | Role |
|---|---|---|---|
| `value` | `utils.networks.Value` called with `actions=None`; `num_ensembles=1`, `hidden_dims=value_hidden_dims`, `layer_norm=True` | returns `(B,)` | Predicts `V(s)` ≈ E_a Q(s, a) |

`Value` already supports state-only calls (`actions=None`) per
`utils/networks.py:177–194`; with `num_ensembles=1` there is no ensemble
axis in the output.

No target copy of `value` — the target critic already provides EMA
stability for V's training target.

The existing `noised_network` (Q_n) is unchanged structurally; only its loss
gains a second term.

## 3. Loss functions

### 3.1 Value loss (NEW)

```
def value_loss(self, batch, grad_params, rng):
    B = batch['actions'].shape[0]
    A = batch['actions'].shape[-1]
    n = self.config['n_v_samples']          # default 8

    # Flow-sampled actions from the current (frozen) one-step actor
    noises = jax.random.normal(rng, (B, n, A))
    obs_exp = broadcast(batch['observations'], (B, n, ...))
    a_samples = self.network.select('actor_onestep_flow')(obs_exp, noises)
    a_samples = jnp.clip(a_samples, -1, 1)
    a_samples = jax.lax.stop_gradient(a_samples)

    # Target via target_critic (EMA Q)
    q_samples = self.network.select('target_critic')(obs_exp, actions=a_samples)  # (2, B, n)
    v_target = jax.lax.stop_gradient(q_samples.mean(axis=(0, 2)))  # (B,)

    # Value module with num_ensembles=1 returns (B,) directly (no ensemble axis).
    v_pred = self.network.select('value')(batch['observations'], params=grad_params)
    # v_pred shape: (B,)

    loss = jnp.mean((v_pred - v_target) ** 2)
    return loss, {'value_loss': loss, 'v_pred_mean': v_pred.mean(), 'v_target_mean': v_target.mean()}
```

**Why `actor_onestep_flow`:** matches the action distribution the policy will
actually take. Using the BC flow directly (via `compute_flow_actions`) would
cost 10× more forward passes per step with no benefit; the distilled actor is
a strictly-better-than-BC sample of the same family.

**Why `target_critic`:** same rationale as NFQL₃'s critic_loss — EMA Q is
stable, online Q is noisy.

### 3.2 Modified noised critic loss (V-anchor added)

Extend NFQL₃'s `noised_critic_loss`. Before returning, append a t=0 term:

```python
# ---- V-anchor at t=0 (NEW) -----------------------------------------
rng, x0_rng = jax.random.split(rng)
x_0_anchor = jax.random.normal(x0_rng, (batch_size, action_dim))
t_zero = jnp.zeros((batch_size, 1))
xt_with_t_zero = jnp.concatenate([x_0_anchor, t_zero], axis=-1)

qn_t0_all = self.noised_network.select('noised_critic')(
    batch['observations'], actions=xt_with_t_zero, params=grad_params
)  # (2, B)
qn_t0 = qn_t0_all.mean(axis=0)                                    # (B,)

v_for_anchor = self.network.select('value')(batch['observations'])  # (B,)
v_for_anchor = jax.lax.stop_gradient(v_for_anchor)

v_anchor_loss = jnp.mean((qn_t0 - v_for_anchor) ** 2)

total = noised_critic_loss + self.config['lambda_v_anchor'] * v_anchor_loss

# in info dict:
info['v_anchor_loss'] = v_anchor_loss
info['qn_t0_mean'] = qn_t0.mean()
info['v_at_anchor'] = v_for_anchor.mean()
```

**Gradient flow to V:** V is stop-grad'd inside the Q_n loss (anchor target);
V is trained entirely by `value_loss` on a separate head. This keeps the two
heads decoupled and prevents Q_n from dragging V off its MC target.

### 3.3 Critic loss — unchanged from NFQL₃

### 3.4 Actor loss — unchanged from NFQL₃

Explicitly preserved:
- ESS-targeted τ* bisection (ess_target=0.7)
- Global MAD for a_norm (`β_MAD = 1.4826 · median|A − median(A)|`, floored at 1e-6)
- Dual R² gate (gate_qn × gate_critic)
- `trust` term (ensemble-disagreement down-weighting)

Dropped from NFQL₄: hard-clip at w_max, fixed τ, per-bin MAD, biased-t
sampling of Q_n, martingale bootstrap, low-t bucketed gate.

### 3.5 Total loss

```
total_loss = critic_loss + actor_loss + value_loss_weight * value_loss
```

where `value_loss_weight = 1.0` by default (V's output scale matches Q's).

`noised_critic_loss` stays on its separate `noised_network` / optimizer, as in
NFQL₃.

## 4. Diagnostics

Keep NFQL₃'s existing logs. Add:

- `noised_critic/v_anchor_loss` — should decay alongside anchor_loss
- `noised_critic/qn_t0_mean`, `v_at_anchor` — should converge
- `value/value_loss`, `v_pred_mean`, `v_target_mean`
- **Four r2_t buckets** (ported from NFQL₄, diagnostic only — never gated):
  `r2_t[0.00_0.25], r2_t[0.25_0.50], r2_t[0.50_0.75], r2_t[0.75_1.00]`

The success criterion for the V-anchor is visible in diagnostics:
**r2_t[0.00_0.25] should track v_anchor_loss**, not collapse to ≈ 1 while
Q_n trivializes.

## 5. Config (new fields over NFQL₃)

```python
n_v_samples=8,          # flow-policy samples for V training target (MC estimate of E_a Q)
lambda_v_anchor=1.0,    # weight on v_anchor_loss inside noised_critic_loss
value_loss_weight=1.0,  # weight on value_loss inside total_loss
v_ensemble_size=1,      # V(s) ensemble; 1 is enough (Q ensemble provides redundancy)
```

All other fields match NFQL₃ defaults.

## 6. Success criteria

Measured on wandb, direct comparison vs NFQL₃ on the same envs:

**Primary (mechanistic):**
1. `v_anchor_loss / anchor_loss` ≥ 0.3 throughout training (V-anchor is
   exerting real pressure).
2. `r2_t[0.00_0.25]` visibly lags `r2_t[0.75_1.00]` during training
   (gap > 0.05 at step 50k), confirming Q_n's low-t regime is actually
   hard.
3. NFQL₄'s `martingale_loss / anchor_loss ≈ 0.009` pathology should not
   recur (no martingale term, but the analogue — `v_anchor_loss ≪
   anchor_loss` — also should not appear).

**Secondary (performance):**
4. Match or beat NFQL₃ on `cube-double-play-singletask-task4-v0` (target
   ≥ 0.22 best-over-training, n=2 seeds).
5. Match or beat NFQL₃ on `antmaze-giant-navigate-singletask-v0` (target
   ≥ 0.30).
6. Net-positive mean Δ vs NFQL₃ across the shared env set.

**Failure threshold:**
If (1) or (2) fail (V-anchor does not change Q_n's behavior), the
hypothesis is wrong and the fallback is Option B (retire the
(x_t, t)-baseline entirely).

## 7. Implementation plan

1. Copy `agents/nfql_3.py` → `agents/nfql_5.py`; rename class to
   `NFQL5Agent`, update `agent_name='nfql_5'`.
2. In `create()`: add `value_def` to `networks`/`network_info`; init via
   same `network_def.init(...)`.
3. Add `value_loss()` method.
4. Modify `noised_critic_loss()` to append V-anchor term as in §3.2.
5. Modify `total_loss()` to include `value_loss_weight * value_loss`.
6. Port bucketed `r2_t_*` logging from `nfql_4.py` into
   `noised_critic_loss` (diagnostic only).
7. Register new config fields in `get_config()`.
8. Smoke test: `python main.py --env_name=cube-double-play-singletask-task4-v0 \
   --agent=agents/nfql_5.py --offline_steps=10000 --eval_interval=0 \
   --log_interval=1000` — confirm losses are finite and the new log keys appear.

## 8. Out of scope

- Online-to-offline phase — NFQL₅ inherits NFQL₃'s online path
  unchanged; no new online mechanics.
- Pixel-based (encoder) path is inherited as-is; V head also takes an
  encoder if configured.
- Visualization / tooling changes.

## 9. Risks and open questions

- **V-target variance.** `v_target = mean over n samples of Q_tgt(s, a_i)`
  is a 1/√n estimator of E_a Q. With n=8 and `actor_onestep_flow` being
  narrow, variance is ≈ 0.35·σ_Q per sample — acceptable but monitored
  via `v_pred_mean − v_target_mean` drift.
- **Policy coverage at initialization.** Early in training the
  one-step actor is near-random, so `V ≈ E_{random} Q`, not `E_{π*} Q`.
  As training proceeds V tracks the improving policy. This is fine
  because Q_n is an *instantaneous* baseline, not a policy-evaluation
  V.
- **Two heads, one optimizer.** V lives in the main `network` with
  shared Adam state — a V that diverges wouldn't directly affect the
  critic/actor (separate loss terms, separate params), but the Adam
  second moments could interact. If this surfaces, split V onto its
  own `TrainState` (as the noised_network is split).
- **Gate interaction.** NFQL₃'s dual gate keys off `r2_qn`. With the
  V-anchor pushing Q_n away from Q(s,a), `r2_qn` (which measures
  Q_n-vs-Q fit) will be structurally lower. The gate target
  `r2_target=0.75` may need a down-adjustment (e.g. 0.5). Add this as
  a tuning knob and leave default at 0.75 initially; diagnose in the
  first sweep.
