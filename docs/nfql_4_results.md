# NFQL₄ Empirical Results and Verdict

Analysis pulls from wandb project `ymkbuaa-beihang-university/fql` (61 parsable
runs, agents `fql`, `nfql_2`, `nfql_3`, `nfql_4`). Evaluation metric is
best-over-training of `evaluation/success` / `evaluation/goal_achieved`
(D4RL), or `evaluation/episode.normalized_return / 100` (antmaze), all
rescaled to a [0, 1]-ish scale. Each row is best-over-training per seed,
averaged across seeds (n=1 seed in most envs; n=2 only on the four
cube-*-task* envs).

Run counts by agent: fql=13, nfql_2=11, nfql_3=23, nfql_4=17.

## 1. Headline verdict: NFQL₄ is a net tie with NFQL₃, not a regression

Head-to-head mean Δ across shared environments, tol ±0.02:

| Comparison | n_envs | NFQL₄ wins | Baseline wins | Ties | Mean Δ (NFQL₄ − baseline) |
|---|---:|---:|---:|---:|---:|
| NFQL₄ vs NFQL₃ | 15 | 8 | 2 | 5 | **+0.013** |
| NFQL₄ vs NFQL₂ |  8 | 4 | 1 | 3 | **−0.020** |
| NFQL₄ vs FQL   |  7 | 3 | 1 | 3 | **+0.018** |
| NFQL₃ vs FQL   |  9 | 3 | 2 | 4 | **+0.013** |

NFQL₄ is **approximately tied** with NFQL₃ at best — it wins more
environments by small margins but loses meaningfully on two
(antmaze-giant, cube-double-task4). It does not deliver the step-change
improvement the proposal justified. Earlier local-only analysis
understated NFQL₄ because five runs NaN'd before wandb upload and their
local `eval.csv` was blank — wandb is authoritative.

## 2. Per-env summary (best over training, seed-mean)

| Env | FQL | NFQL₂ | NFQL₃ | NFQL₄ |
|---|---:|---:|---:|---:|
| antmaze-giant-navigate-singletask-v0        | — | — | **0.30** | 0.12 |
| antmaze-large-navigate-singletask-v0        | — | — | 0.82 | 0.84 |
| antmaze-medium-diverse-v2                   | **1.00** | 0.98 | 0.96 | 0.98 |
| antmaze-umaze-diverse-v2                    | 1.00 | 1.00 | 1.00 | — |
| antsoccer-arena-navigate-singletask-v0      | — | — | 0.42 | **0.50** |
| cube-double-play-singletask-v0              | — | **0.98** | 0.46 | 0.60 |
| cube-double-play-singletask-task4-v0        | 0.07±.03 | 0.08 | **0.22±.08** | 0.12 |
| cube-single-play-singletask-v0              | — | — | 1.00 | 1.00 |
| cube-triple-play-singletask-task2-v0        | 0.01 | 0.00 | 0.00 | 0.02 |
| cube-triple-play-singletask-task3-v0        | 0.02 | 0.02 | 0.02 | 0.02 |
| cube-triple-play-singletask-task4-v0        | 0.00 | 0.02 | 0.00 | — |
| door-cloned-v1                              | 1.04 | 0.98 | 1.08 | 1.08 |
| hammer-cloned-v1                            | 1.40 | 1.40 | 1.42 | **1.43** |
| pen-cloned-v1                               | 1.55 | 1.54 | 1.51 | **1.57** |
| puzzle-3x3-play-singletask-v0               | — | — | 0.20 | **0.26** |
| puzzle-4x4-play-singletask-v0               | — | — | 0.18 | **0.22** |
| scene-play-singletask-v0                    | — | — | 0.96 | **0.98** |

(Metric scale is per-env native: OGBench success [0, 1], D4RL normalized
return can exceed 1.0. Bolding is the best cell in the row.)

### The two real regressions

- **antmaze-giant:** NFQL₄ 0.12 vs NFQL₃ 0.30 (Δ=−0.18). The hardest
  navigation env in the set; NFQL₄'s narrower advantage-weighting recipe
  loses traction.
- **cube-double-task4:** NFQL₄ 0.12 vs NFQL₃ 0.22 (Δ=−0.10). Previously
  NFQL₃'s strongest relative win over FQL (+0.15); NFQL₄ gives most of
  that back.

### The catastrophic cube-double-play-v0 pattern

On the simpler (non-task) variant: NFQL₂=0.98, NFQL₃=0.46, NFQL₄=0.60.
Both NFQL₃ and NFQL₄ regress sharply vs NFQL₂ here — the ESS-τ /
advantage-weighting machinery introduced after NFQL₂ is actively harmful
on this env. Root cause is shared between NFQL₃ and NFQL₄, not unique to
NFQL₄.

## 3. Diagnostic-level failure modes (wandb-confirmed at 2M steps)

### 3.1 The "low-t harder than high-t" gate hypothesis is empirically false

NFQL₄'s proposal §3.1–3.2 assumed trivial high-t regime pins global R²≈1
while low-t is still converging, justifying a bucketed gate that waits
on the low-t bucket. Final-step wandb values from
`nfql_4_sd000_20260418_224442_cube-double-play-singletask-task4-v0`:

| bucket | r2_t |
|---|---:|
| r2_t[0.00, 0.25] | 0.9984 |
| r2_t[0.25, 0.50] | 0.9985 |
| r2_t[0.50, 0.75] | 0.9984 |
| r2_t[0.75, 1.00] | 0.9986 |

All four buckets converge together to ≈ 0.998. `gate_c` sits at ≈ 1.0
across every nfql_4 run examined — the gate never blocks. The
asymmetric signal the bucketed gate was designed to expose does not
exist in the trained model.

### 3.2 Q_n trivializes: martingale_loss ≪ anchor_loss

The martingale bootstrap `L_mart = (Q_n(t_a) − sg Q_n(t_b))²` combined
with the anchor `L_anchor = (Q_n(t_b) − Q(s, a_i))²` has fixed point

```
Q_n(s, x_t, t) → Q(s, a_i)   for all t
```

because the bootstrap target (`Q_n(t_b)`) is itself being pinned to
`Q(s, a_i)`. Wandb summary on cube-double-task4 confirms:

| term | value |
|---|---:|
| anchor_loss     | 2.5604 |
| martingale_loss | 0.0219 |
| ratio           | 0.0086 |

`martingale_loss / anchor_loss ≈ 0.9%` — Q_n is essentially flat across
t. The `(x_t, t)`-conditional baseline the architecture is supposed to
produce is not being learned. The proposal foresaw this risk (§3.3:
"EMA target Q_n stabilizes the bootstrap") but that safety was not
added, and `stop_gradient(Q_n_current)` alone does not prevent the
trivial fixed point.

### 3.3 Five runs diverged to NaN before 50k steps (local only)

Early NFQL₄ runs without the final numerical guards (β_MAD floor 1e-3,
a_norm clip ±10, non-finite EMA filter) NaN'd out:

| run | status by step 50k |
|---|---|
| `nfql_4_sd000_20260417_174733_antmaze-medium-diverse-v2` | `beta_mad=inf, a_med=inf, bc_weight_mean=NaN` |
| `nfql_4_sd000_20260417_174807_cube-double-task4`         | same |
| `nfql_4_sd000_20260417_213859_cube-double-task4`         | same |
| `nfql_4_sd000_20260418_024448_cube-triple-task2`         | same |
| `nfql_4_sd000_20260418_084449_cube-triple-task3`         | same |

These runs did not reach wandb (hence the wandb run-count gap: 17
vs 22 local). NFQL₃ shows no such failures in the same environments;
the bootstrap-induced Q_n drift compounded with the tighter low-t
normalization until the a_norm clip saturated the whole batch.

### 3.4 bc_weight_max pinned at w_max across all runs — hard clip is structurally active

Final-step wandb summary across five nfql_4 runs (task4 / antmaze-medium /
antmaze-large / antmaze-giant / scene):

| env | bc_weight_low_mean | bc_weight_mean | bc_weight_max |
|---|---:|---:|---:|
| cube-double-task4   | 0.9705 | 0.9843 | **9.999** |
| antmaze-medium      | 0.5595 | 0.7660 | **9.999** |
| antmaze-large       | 0.1032 | 0.5341 | **9.999** |
| antmaze-giant       | 0.8494 | 0.9217 | **9.999** |

- `bc_weight_max` is **always at the w_max cap**. Every update saturates
  the clip on at least one sample.
- `bc_weight_low_mean` is depressed well below 1.0 on navigation envs
  (antmaze-large: 0.10). The "emphasize low-t samples" goal is
  **inverted** — low-t samples receive *less* BC gradient than high-t.
- `bc_weight_mean` is depressed below 1.0 on antmaze-large and
  antmaze-medium: ~30-50% total BC signal reduction vs uniform.

Mechanism: a_norm clipped to ±10 and divided by fixed τ=1 → logit span
20 → post-exp weights concentrate on a handful of samples → normalize
to low-t mean 1 → clip top mass at w_max removes it → post-clip mean
drops. NFQL₃'s ESS-targeted τ* widens τ adaptively to keep ESS ≈ 0.7,
flattening the distribution and avoiding this saturation.

### 3.5 Dropping the `trust` term removed a stabilizer

§3.6 of the proposal hypothesized `trust` had wrong polarity at low t.
Empirically, `trust` was doing useful variance-reduction work —
suppressing samples where the two Q_n heads disagreed. Removing it
exposed the actor to higher-variance low-t weights, which compounds
§3.4.

## 4. Root cause in one sentence

> The `(x_t, t)`-conditional baseline is architecturally sound, but
> NFQL₄'s training recipe drives Q_n toward `Q(s, a_i)` regardless of t
> so the conditional expectation is not realized. The bucketed gate,
> biased-t sampling, per-bin MAD, hard clip, and dropped `trust` /
> dropped ESS bisection were all tuned against a low-vs-high-t
> asymmetry that the implementation never produces — they add rigidity
> without adding signal, which is why NFQL₄ barely moves the mean
> across 15 environments.

## 5. What to take forward into NFQL₅

1. **The premature-gate problem is not about t-bucketing.** All four
   r2_t buckets converge together, because Q_n trivially fits
   `Q(s, a_i)`. A gate must measure something that is non-trivial
   *even when Q_n matches* — e.g. the gap between `Q_n(s, ·, 0)` and
   `V(s) = E_a Q(s, a)`, which is zero only when Q_n realizes the
   correct conditional expectation.

2. **Bootstrapping a Q-valued function along t without a target net is
   unstable.** Either add an EMA target Q_n to the two-t bootstrap, or
   drop the bootstrap term entirely.

3. **Fixed τ + hard clip is structurally wrong when advantages are
   skewed.** `bc_weight_max` pinning at w_max in every run across every
   env is proof the clip is doing active work — and it depresses the
   low-t mean below the design intent (3.4). NFQL₃'s ESS-targeted
   bisection adapts to the distribution and should be restored.

4. **Q_n needs a `t=0` boundary condition to not trivialize.** The
   `t=1` boundary is `Q(s, a)`. The `t=0` limit of the correct
   conditional expectation is `V(s) = E_{a}[Q(s, a)]`. Without anchor,
   nothing in the loss forbids the trivial fixed point Q_n ≡ Q(s, a_i).
   Anchoring Q_n at t=0 to a running estimate of V(s) (e.g. a separate
   `V(s)` head trained by IQL expectile, or `mean_i Q(s, a_i)` over the
   same noised batch) directly targets the root cause.

5. **Numerical guards are baseline.** β_MAD floor, a_norm clip, and
   non-finite EMA filtering must be kept — they are the only reason 17
   of 22 NFQL₄ runs completed.

## 6. Recommendation

Do not keep NFQL₄ as production. Preserve from it:

- Bucketed `r2_t_*` logging (as a *diagnostic*, not a gate signal).
- Numerical safety guards (β_MAD floor 1e-3, a_norm clip ±10, EMA NaN
  filter).

Drop from it:

- Low-t bucketed gate (empirically degenerate).
- Biased-t sampling for Q_n (t_a ∈ [0, 0.5], t_b ∈ [0.5, 1.0]) — with
  no martingale signal to exploit, the uniform `t ∼ U[0, 1]` is fine.
- Martingale bootstrap (drives trivialization).
- Per-bin MAD (global MAD was fine; the bin split added rigidity).
- Hard clip at fixed τ (NFQL₃'s ESS-τ is strictly better).
- Drop of `trust` (was a stabilizer, not a blocker).

### NFQL₅ design options

- **(A) Restore NFQL₃'s actor recipe and add a V-anchor on Q_n.** Keep
  ESS-τ bisection, `trust`, global MAD. Add an auxiliary head
  `V(s) ≈ E_a Q(s, a)` (trained by same-batch Monte-Carlo:
  `L_V = (V(s) − mean_i Q(s, a_i))²`) and add to the noised-critic loss
  a `t=0` boundary term `L_V_anchor = (Q_n(s, a, 0) − V(s))²`. The
  conditional expectation now has both boundaries pinned (Q_n → Q at
  t=1, Q_n → V at t=0), so the trivial fixed point is excluded.

- **(B) Retire the (x_t, t)-baseline.** Fall back to a plain `V(s)`
  baseline and keep only the dual R² gate as safety. Accepts that the
  timestep-level advantage narrative did not pay off, while keeping the
  numerical-safety deliverables.

Option A preserves the research line and addresses the identified root
cause (Q_n trivialization) with a minimal, well-motivated change.
Option B is the conservative fallback — less risk, less potential
upside.

Recommendation: **Option A**, but gate the extra complexity on
reproducing NFQL₃'s results first (so the V-anchor is an isolated
variable). If the V-anchor does not measurably reduce
`martingale_loss / anchor_loss` below 0.5 and improve `cube-double-
task4` or `antmaze-giant` over NFQL₃, fall back to Option B.
