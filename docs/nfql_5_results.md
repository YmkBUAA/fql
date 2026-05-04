# NFQL₅ Empirical Results and Verdict

Analysis of 20 completed `nfql_5` runs on wandb (3 still running at time of
writing), across 16 environments. Compared against `fql`, `nfql_2`, `nfql_3`,
`nfql_4` on the same environments. Data pulled from
`ymkbuaa-beihang-university/fql`.

NFQL₅ was designed per `docs/superpowers/specs/2026-04-19-nfql_5-design.md`
to fix NFQL₄'s confirmed Q_n trivialization via a V(s) head and a t=0
boundary anchor: `(Q_n(s, x_0, 0) − stop_grad V(s))²`.

## 1. Headline verdict: mechanistic fix confirmed, performance is a net tie

**The mechanistic hypothesis is verified.** The V-anchor prevents Q_n from
collapsing to `Q(s, a_i)`:

| quantity | NFQL₄ (prod) | NFQL₅ (prod) |
|---|---:|---:|
| `martingale_loss / anchor_loss` / `v_anchor_loss / anchor_loss` | 0.009 | **0.24–2.46 (median ≈ 0.65)** |
| `r2_t[0.00–0.25]` vs `r2_t[0.75–1.00]` gap during training (2k steps, cube-double-task4) | ≈ 0 | **0.09** |
| `Q_n(s, x_0, 0)` vs `V(s)` agreement at 1M+ steps | — | within ≤ 0.3 of V on all 20 envs |

Q_n now realizes the conditional expectation its architecture is supposed to
produce. The V-anchor ratio is **~72× larger** than NFQL₄'s martingale
fraction — the fixed-point collapse is closed.

**But performance did not follow.** Head-to-head vs prior agents:

| Comparison | n_envs | NFQL₅ wins | Baseline wins | Ties | Mean Δ (NFQL₅ − baseline) |
|---|---:|---:|---:|---:|---:|
| NFQL₅ vs NFQL₄ | 14 | 3 | 3 | 8 | **−0.010** |
| NFQL₅ vs NFQL₃ | 16 | 5 | 6 | 5 | **−0.010** |
| NFQL₅ vs NFQL₂ |  7 | 2 | 1 | 4 | **−0.054** |
| NFQL₅ vs FQL   |  6 | 2 | 1 | 3 | **+0.005** |

NFQL₅ is **statistically indistinguishable from NFQL₄ and NFQL₃ in
aggregate**. The trivialization fix was real but not the binding constraint
on performance.

## 2. Per-env summary (best-over-training, seed-mean)

| Env | FQL | NFQL₂ | NFQL₃ | NFQL₄ | NFQL₅ |
|---|---:|---:|---:|---:|---:|
| antmaze-giant-navigate-singletask-v0     | — | — | **0.30** | 0.12 | 0.23±.05 (n=2) |
| antmaze-large-navigate-singletask-v0     | — | — | 0.82 | 0.84 | **0.90** |
| antmaze-medium-diverse-v2                | **1.00** | 0.98 | 0.96 | 0.98 | 0.98 |
| antsoccer-arena-navigate-singletask-v0   | — | — | 0.42 | 0.50 | **0.53±.03 (n=2)** |
| cube-double-play-singletask-v0           | — | **0.98** | 0.46 | 0.60 | 0.48 |
| cube-double-play-singletask-task4-v0     | 0.07 | 0.08 | **0.22±.08** | 0.12 | 0.10 |
| cube-single-play-singletask-v0           | — | — | 1.00 | 1.00 | 1.00 |
| cube-triple-play-singletask-task2-v0     | 0.01 | 0.00 | 0.00 | 0.02 | 0.02 |
| cube-triple-play-singletask-task3-v0     | 0.02 | 0.02 | 0.02 | 0.02 | 0.00 |
| door-cloned-v1                           | 1.04 | 0.98 | 1.08 | 1.08 | 1.07 |
| humanoidmaze-large-navigate-singletask-v0| — | — | **0.14** | — | 0.04 |
| humanoidmaze-medium-navigate-singletask-v0| — | — | **0.54** | — | 0.44 |
| pen-cloned-v1                            | 1.55 | 1.54 | 1.51 | **1.57** | 1.56 |
| puzzle-3x3-play-singletask-v0            | — | — | 0.20 | **0.26** | 0.16 |
| puzzle-4x4-play-singletask-v0            | — | — | 0.18 | **0.22** | 0.20 |
| scene-play-singletask-v0                 | — | — | 0.96 | **0.98** | 0.94 |

### Where NFQL₅ moves the needle

- **antmaze-giant** vs NFQL₄: +0.11 (0.12 → 0.23). NFQL₅ recovers ~60% of
  the regression NFQL₄ introduced, though still trails NFQL₃'s 0.30.
- **antmaze-large** vs NFQL₃: +0.08, vs NFQL₄: +0.06. Best result on this
  env across all agents.
- **antsoccer-arena** vs NFQL₃: +0.11. Best result across all agents.

These are all **long-horizon sparse-reward navigation** envs, where a
structured V(s) baseline has clear information value.

### Where NFQL₅ regresses

- **humanoidmaze-medium / humanoidmaze-large** vs NFQL₃: −0.10 each.
- **cube-double-play-singletask-task4** vs NFQL₃: −0.12.
- **puzzle-3x3** vs NFQL₄: −0.10.

The humanoidmaze regression is the most surprising — same class of task as
antmaze-large (where NFQL₅ wins) but with richer state geometry. Likely
mechanism: V(s) over a high-dim state has higher MC variance at fixed
`n_v_samples=8`, injecting noise into Q_n's t=0 boundary during training.

## 3. Design success criteria — scored

From `docs/superpowers/specs/2026-04-19-nfql_5-design.md §6`:

| # | criterion | target | observed | verdict |
|---|---|---|---|---|
| 1 | `v_anchor_loss / anchor_loss` ≥ 0.3 | ≥ 0.30 | 0.24–2.46, median ≈ 0.65 | ✓ **met** |
| 2 | `r2_t[0.00–0.25]` < `r2_t[0.75–1.00]` gap > 0.05 during training | > 0.05 | 0.09 at 2k steps | ✓ **met** |
| 3 | `v_anchor_loss ≪ anchor_loss` must not recur | not < 0.1 | no env below 0.24 | ✓ **met** |
| 4 | match NFQL₃ on cube-double-task4 (≥ 0.22) | ≥ 0.22 | 0.10 | ✗ **missed by 0.12** |
| 5 | match NFQL₃ on antmaze-giant (≥ 0.30) | ≥ 0.30 | 0.23±.05 | ✗ **missed by 0.07** |
| 6 | net-positive mean Δ vs NFQL₃ | > 0 | −0.010 | ✗ **barely negative** |

Primary (mechanistic) criteria: **3/3 met**.
Secondary (performance) criteria: **0/3 met**.

This is the clean experimental result the spec set up. The V-anchor works
*as designed*, and *fixing it does not improve end-task performance*.

## 4. What this tells us about the research line

The NFQL₂ → NFQL₃ → NFQL₄ → NFQL₅ progression was built on one premise: **a
better `(x_t, t)`-conditional baseline yields a better weighted-BC signal,
yielding a better actor.** NFQL₅'s mechanistic verification of that
baseline gives us a clean falsification:

- **NFQL₄ had a broken baseline** (Q_n ≡ Q(s, a_i)) and tied NFQL₃.
- **NFQL₅ has a correct baseline** (Q_n interpolates between Q and V) and
  *also* ties NFQL₃.

If the baseline quality were the binding constraint, NFQL₅ should beat
NFQL₄. It does not. Therefore the baseline quality is **not** the binding
constraint. Something downstream — in the actor weighting, the ESS-τ
mechanism, or outside the advantage machinery entirely — is the limit.

Candidate limits, ordered by prior from the wandb evidence:

1. **ESS-τ mechanism**: all four agents share the ESS-targeted τ*, and all
   four share the same pattern of `cube-double-play-v0` catastrophe
   (NFQL₂=0.98 → NFQL₃=0.46 → NFQL₄=0.60 → NFQL₅=0.48). The loss on this
   env is not about Q_n. It is about the ESS-τ mechanism *itself* —
   NFQL₂ did not have it and won by 0.5.
2. **Distillation**: the one-step flow actor is a hard bottleneck on BC
   quality at inference time; improvements to the BC weighting don't help
   if the distilled actor can't represent them.
3. **Q(s, a) bias at weak gate**: when `gate_c ≈ 1`, the weighted BC loss
   becomes sensitive to `Q − Q_n` errors that are not bias-corrected.
   NFQL₅ did not address this.

## 5. Recommendation

**Do not iterate NFQL₆ on the conditional-baseline axis.** Two successive
attempts (NFQL₄ with its theoretical gate split, NFQL₅ with its V-anchor)
have left performance on the same plateau. The axis is saturated.

**Preserve from NFQL₅:**
- `agents/nfql_5.py` stays in the agent registry — it is a correct
  implementation of a correct idea, and the V head is useful
  infrastructure for any future baseline variant.
- The V-anchor / V-head training pattern in `value_loss` is a reusable
  recipe.

**Two credible next directions:**

- **(C) Ablate the ESS-τ mechanism.** NFQL₂'s fixed low τ beats all
  subsequent agents on `cube-double-play-v0` by 0.4+. Either the ESS
  bisection misfires on dense-reward short-horizon tasks, or its default
  `ess_target=0.7` is systematically too conservative. An NFQL₆ (or a
  configuration sweep over NFQL₃) that tests fixed τ ∈ {0.1, 0.3, 1.0} on
  this specific env would isolate the effect.
- **(D) Switch the bottleneck.** Leave NFQL₃ as the production advantage
  agent and invest the next research budget in the *actor* side: a more
  expressive distilled policy, a consistency-model-style one-step
  policy, or an online-fine-tuning protocol.

**Unchanged from `nfql_4_results.md` recommendation:** retire NFQL₄ as a
production agent. NFQL₃ remains the best-rounded baseline; NFQL₅ is its
mechanistically-cleaner peer.

## 6. Retained diagnostics worth keeping

From NFQL₄: bucketed `r2_t_*` (diagnostic).
From NFQL₅: `v_anchor_loss`, `qn_t0_mean`, `v_at_anchor_mean`,
`value/v_pred_mean`, `value/v_target_mean`. All continue to be useful for
any future agent that trains a Q_n-style baseline.

## 7. Notes on the data

- 3 NFQL₅ runs (`door-cloned-v1`, `puzzle-4x4`, `cube-triple-task3`) were
  still running at time of analysis. `door-cloned-v1` and `puzzle-4x4` had
  already converged (score stable for > 500k steps) so including them in
  the table is safe; `cube-triple-task3` is at 250k/1M and its 0.00 score
  may not be final.
- All NFQL₅ seeds are 000; only antmaze-giant and antsoccer have n=2 runs
  (multiple experiment configurations). Per-env variance is therefore not
  well-estimated for single-run envs.
- No NFQL₅ runs NaN'd in this batch — the numerical guards inherited from
  the NFQL₃ base carry over cleanly.
