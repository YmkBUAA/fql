# Recommended Solution: Reliability-Gated Exponential Weighting

## Motivation

The experimental evidence to date gives us two hard facts:

1. **Exponential weighting is theoretically superior to binary weighting** — proven via two
   independent derivations in `weighted_bc_theory.md` (KL-constrained policy optimisation with
   Rao-Blackwell variance reduction, and posterior softmax selection with full information
   retention).

2. **The offline stage is where the current binary weighting hurts.** The ablation results show:

   ```
   antsoccer-arena:         online_only=True → 0.92   online_only=False → 0.82
   humanoidmaze-medium:     online_only=True → 0.36   online_only=False → 0.21
   ```

   The cause is mechanical: during offline training the noised critic is still learning to
   approximate the conditional neighbourhood mean, so its outputs carry high approximation noise
   relative to the actual signal. Using an unreliable critic's comparison to gate the BC loss
   from step 1 injects noise into the policy update.

The recommended solution addresses **both** issues in a single design: a theoretically principled
continuous weight that is automatically suppressed while the noised critic is unreliable and
smoothly activates as its approximation improves.

---

## The Full Formula

Let `Q` be the original critic (ensemble mean) and `Q_n` the noised critic (ensemble mean).
Let `step` be the current training step.

```
1. Local advantage
   A_local(s, a)  =  Q(s, a)  −  Q_n(s, a + ε, t)

2. Self-normalised temperature
   β  =  std_{(s,a) ∈ batch} [ A_local(s, a) ]

3. Raw exponential weight
   w_raw(s, a)  =  exp( A_local(s, a) / β )

4. Batch normalisation (mean weight = 1)
   w_exp(s, a)  =  w_raw(s, a)  /  mean_{batch}[ w_raw ]

5. Reliability gate
   c(step)  =  sigmoid( ( step − s_mid ) / s_scale )

6. Final weight
   w(s, a)  =  1  +  c(step) · ( w_exp(s, a) − 1 )
```

The weight is applied to the per-sample BC flow loss:

```
L_BC  =  mean_{batch} [ w(s, a) · ‖ pred − vel ‖² ]
```

---

## Intuition for Each Step

**Step 1 — Local advantage.** `Q_n` approximates the conditional mean of Q over the local
action neighbourhood (this is a property of rectified-flow linear interpolation — overlapping
noised targets force `Q_n` to learn the mean). `A_local` therefore measures how much better
action `a` is than its local neighbours under the current critic.

**Step 2 — Self-normalised temperature.** `β = std(A_local)` standardises the signal into a
z-score, making the weight invariant to the absolute scale of Q. This is the same principle as
`normalize_q_loss` in the existing codebase. No manual tuning.

**Step 3 — Raw exponential weight.** The exponential form is not arbitrary. It arises from two
independent derivations:
- KL-constrained policy improvement (AWR-style): `w ∝ exp(A/λ)` is the closed-form optimal policy.
- Softmax posterior: `exp(A_local/β)` is the probability that a locally-rational softmax agent
  would select `a` over its neighbours.

**Step 4 — Batch normalisation.** Ensures `mean(w_exp) = 1`, so the overall BC loss magnitude
is preserved. The weighting only reshapes **which samples matter most**, not the effective
learning rate.

**Step 5 — Reliability gate.** A smooth sigmoid ramp from 0 to 1:
- Early in training (`step ≪ s_mid`): `c ≈ 0`, the gate suppresses the weighting
- Late in training (`step ≫ s_mid`): `c ≈ 1`, the gate lets the full exponential weight through
- Around `s_mid`: smooth transition over a window of width `s_scale`

**Step 6 — Final weight.** `w = 1 + c · (w_exp − 1)` is the linear interpolation between
uniform weighting (`w = 1`, standard BC) and full exponential weighting (`w = w_exp`). When
`c = 0` the BC loss is unchanged; when `c = 1` it is fully weighted.

---

## Why the Reliability Gate Solves the Offline Problem

The ablation showed that applying weights during the offline stage actively hurts. The root
cause was traced to the noised critic producing unreliable outputs before it has converged.
There are three candidate ways to handle this:

1. **Hard stage switch** (current `weighted_bc_online_only=True`) — disables weighting in offline,
   enables it in online. Abrupt transition; loses potential offline benefit entirely.
2. **Ensemble disagreement gate** — use `|Q_n_1 − Q_n_2|` as a confidence signal. Principled but
   the initial disagreement is dominated by random-init noise, not genuine uncertainty.
3. **Sigmoid curriculum on step** — gradually ramp the weight as training progresses.

The sigmoid curriculum wins because:
- It produces a **smooth** transition that avoids discontinuities in the loss landscape
- It requires no assumption about what `Q_n`'s internal state looks like
- It exactly matches the observed failure mode: the noised critic needs ~500k steps to converge,
  after which the signal becomes reliable
- It has only two hyperparameters (`s_mid`, `s_scale`) and both have natural defaults:
  `s_mid = offline_steps / 2`, `s_scale = offline_steps / 10`

**Default schedule for the standard 1M-step offline + 1M-step online run:**
```
s_mid    = 500,000      (midpoint of offline phase)
s_scale  = 100,000      (transition width ~ 10% of offline budget)
```

This gives:
- step 0        → c = 5e-3  → effectively uniform BC
- step 200,000  → c = 0.05  → mostly uniform BC
- step 500,000  → c = 0.50  → half-weighted
- step 800,000  → c = 0.95  → mostly weighted
- step 1,000,000+ (online) → c ≈ 1  → fully weighted

The algorithm therefore behaves like standard FQL during the early offline stage (matching the
`weighted_bc_online_only=True` mode, which is empirically proven to work), but smoothly unlocks
the weighting once the noised critic is ready, extracting additional benefit from the last
500k offline steps and the entire online phase.

---

## Predicted Experimental Impact

| Regime | Current binary, `online_only=True` | Recommended reliability-gated exp |
|---|---|---|
| Pure offline, early training | No weighting applied | No weighting applied (gate closed) |
| Pure offline, late training | No weighting applied | Smoothly introduces exp weighting |
| Online fine-tuning | Binary {0.9, 1.2} | Full exponential weighting |
| Information content of weight | ≤ 1 bit | Full `H(A_local)` |
| Hyperparameters exposed | `w_high`, `w_low` | `s_mid`, `s_scale` (both with sensible defaults) |

**Expected gains over current binary + online_only:**

1. **During the late offline stage (steps ~500k–1M)**, the gate has opened and the exp
   weighting gets ~500k extra steps of effect that the hard-switch approach misses. On tasks
   where dataset quality is heterogeneous (e.g. cube-double, puzzle variants), this should
   improve the offline endpoint.

2. **During online fine-tuning**, the exponential weight uses the full magnitude of the local
   advantage instead of thresholding it. On the two ablation environments where we have
   evidence (antsoccer, humanoidmaze-medium), the binary variant gave gains of +0.10 and +0.15
   respectively. The theoretical information-content argument predicts the exponential variant
   should equal or exceed these — the magnitude information that binary discards is exactly
   what distinguishes a marginally-above-average action from a dramatically-above-average one.

3. **Stability via variance reduction.** The Rao-Blackwell argument guarantees lower-variance
   policy gradients than standard AWR. Combined with adaptive β normalisation, training should
   be more reproducible across seeds — directly addressing the high seed variance visible in
   the current results (e.g. antmaze-giant ranging from 0.04 to 0.10 between seeds).

4. **No hyperparameter tuning.** The current method has `bc_weight_high` and `bc_weight_low`
   which must be chosen. The recommended method has only `s_mid` and `s_scale`, which have
   principled defaults derived from `offline_steps`. Fewer knobs → easier to deploy.

---

## What This Design Is Not

To be explicit about scope:

- **It does not modify the Q loss or the distillation loss.** The weighting is applied only to
  the BC flow loss, which is the term that directly consumes dataset actions. The Q loss and
  distillation continue to operate unchanged.

- **It does not replace the noised critic's training objective.** The noised critic still
  learns its current MSE-against-stop-gradient-Q target. The weighting only consumes the
  critic's output; it does not restructure how the critic itself is trained.

- **It does not introduce a second network.** Everything is computed from the two networks
  that already exist (`critic` and `noised_critic`).

- **It is backward-compatible.** Setting `c = 0` always recovers standard FQL exactly, so any
  negative result can be cleanly attributed to the weighting itself.

---

## Summary

The recommended solution is the **exponential self-normalised weighting from the theory
document**, wrapped in a **sigmoid curriculum on the training step**. The exponential form is
theoretically optimal (two derivations agree); the curriculum eliminates the one empirical
failure mode we observed (unreliable weighting during early offline training). Together they
give a single method that should match the best current variant (`binary + online_only`) at
worst, and beat it on late-offline and online stages at best — while removing one hyperparameter
from the user's hands.
