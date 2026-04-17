# Critical assessment of NFQL₃ and a proposal for NFQL₄

The NFQL₃ intuition is elegant but the implementation has a structural flaw
that is visible in the training-side diagnostics
(`analysis_out/diag_*.png`, `nfql3_diagnostics.py`). This document walks
through the flaw and proposes a simpler design, NFQL₄, that keeps what is
doing real work and removes the parts whose only observable effect is to
obscure what is happening.

---

## 1. Where the theory breaks

### 1.1 At t → 1, the "multi-scale baseline" collapses to Q itself

The noised-critic loss (`agents/nfql_3.py:289–290`) regresses
`Q_n(s, x_t, t)` to `Q(s, a_i)` — the **same** target that also produced
`x_t = (1−t)·x_0 + t·a_i`. At `t ≈ 1`, `x_t ≈ a_i`, so the network's task
reduces to:

```
input  = (s_i, a_i, 1)
target = Q(s_i, a_i)
```

which is a trivial copy of Q. Therefore `Q_n(s, a, 1) ≈ Q(s, a)`, and the
advantage

```
A_matched = Q(s, a) − Q_n(s, x_t, t)   →   0   as t → 1.
```

Proposition 3.1 and Corollary 3.2 in `nfql_3_theory.md` describe the
conditioning *radius* `(1−t)·√d` shrinking to zero but treat the limit as a
"tight local mean". The tight local mean of a point mass is the point
itself — there is **no local structure** left. The signal at high t is
not "finer", it is **zero**.

### 1.2 The dual R² gate is miscalibrated by exactly this trivial regime

The diagnostics show `r²_qn ≈ 0.99+` essentially from step 1 in every run.
That is dominated by the easy high-t samples in the minibatch: predicting
`Q(s, a)` from `(s, a, 1)` is trivial with enough network capacity, so the
MSE is near-zero there, and those points alone push
`L_noised / Var[Q]` close to 0.

So the gate opens *because Q_n memorises Q at high t*, not because Q_n is
a good neighbourhood baseline at low t — which is where it actually
matters. Theorem 3.5 in the theory doc says "R²_critic prevents premature
opening"; it does *not* prevent R²_qn from being a bad proxy for the
property we actually care about.

### 1.3 Training data for Q_n is unbalanced in difficulty

With `t ~ U(0, 1)`, roughly half the minibatch has `t > 0.5` where the
regression is nearly trivial and half has `t < 0.5` where it is hard
(inferring Q(s, a) from Gaussian noise). The *reported* loss is pulled
down by the easy half, while Q_n's accuracy at low t — the only regime
where A_matched carries signal — is not what R²_qn measures.

### 1.4 The empirical fingerprint matches

From `nfql3_diagnostics.py`:

| Diagnostic | Observed | Implication |
|---|---|---|
| `r²_qn ≈ 0.99`, `gate_c ≈ 0.99` from step ~5k | all 14 runs | trivial high-t regression drives the gate |
| `bc_weight_mean ≈ 1.00` | all runs | integrated-over-t reweighting is a near-no-op |
| `bc_weight_spread` ≈ 2–5 per batch | all runs | variation exists, but concentrated in low-t samples |
| `frac_suboptimal ≈ 0.50` | all runs | Q_n ≈ Q at high t gives symmetric noise |

So what NFQL₃ is actually doing = **plain AWR at t ≈ 0, gradually tapering
to a no-op at t ≈ 1**, wrapped in a gate that opens based on a metric
dominated by a trivial regime. The elegant "scale-adaptive baseline"
story is undermined by its own construction.

### 1.5 Is the taper itself bad?

No — tapering weights to 1 at t=1 is **defensible**: at t=1 the velocity
target `a − x_0` only depends on a through a trivial additive term, so
reweighting has little to modulate. The problem is not that high-t weights
are ≈ 1; it is that

- (a) the gate *claims* Q_n is informative there when it is not, and
- (b) the whole Q_n apparatus + ESS targeting + MAD normalisation + trust
  is added complexity for what is ultimately *low-t AWR with a soft
  mask*.

---

## 2. Proposal: NFQL₄

Keep the parts doing real work; remove the parts whose only effect is to
obfuscate what is happening; make the taper explicit so it can be reasoned
about.

### 2.1 Replace Q_n with a V-network trained from the current one-step policy

```
V_φ(s)  ≈  E_{â ~ π_onestep(·|s)}[ Q(s, â) ]
```

Training: sample `k` noise vectors, run `actor_onestep_flow`, evaluate Q
on those actions, regress V_φ to the mean with stop-gradient. This is a
genuine state baseline with no architectural pathology, it is consistent
across all t, and its "goodness" can be measured cleanly by

```
R²_V = 1 − L_V / Var[E_π Q]
```

without the high-t trivial-regime confound.

Cost: one extra network forward per step, same order as Q_n. Actually
*cheaper* because no `n_noised_actions = 4`-way expansion.

### 2.2 Make the taper explicit (not emergent from a pathology)

```
A_i       = Q(s_i, a_i) − V_φ(s_i)                 # t-independent
w_exp_i   = exp( (A_i − median(A)) / (MAD · τ) )   # fixed τ, no ESS search
w_exp_i   = clip( w_exp_i / mean, 0, w_max )       # hard cap, e.g. w_max=10

m(t)      = max(0, 1 − 2·t)                        # taper: 1 at t=0, 0 at t ≥ 0.5
u_i       = 1 + m(t_i) · c_gate · trust_i · (w_exp_i − 1)
```

Rationale: the BC loss at t=1 is almost-trivial (predict `a − x_0` given
`(s, a, 1)`), so uniform weights there are correct and *intentional*. At
t ≈ 0 the velocity is solving the direction-from-noise problem and
AWR-style bias toward high-advantage samples is genuinely useful. `m(t)`
encodes this as a design choice rather than hoping matched-Q_n happens
to produce it.

### 2.3 Gate on what actually has to be true

Two conditions, and no more:

```
gate_V  = σ( (R²_V      − r²_V_target)      / κ_V )   # R²_V on low-t samples only
gate_Q  = σ( (R²_critic − r²_critic_target) / κ_Q )
c_gate  = gate_V · gate_Q
```

Keep the EMA machinery — it is sound. Drop the `trust` term (ensemble
disagreement) unless ablation shows it helps; a 2-ensemble head on a
convergent regression target tends to give near-zero disagreement by step
20k, so it adds compute and hyperparameters without signal.

### 2.4 Drop ESS-targeting; drop the 14-step bisection in the hot loop

ESS-targeting per batch is a constant rescaling that satisfies a moment
condition already roughly enforced by MAD + fixed τ + clipping.
Empirically `ess_achieved` sits near its target, which means τ* mostly
does not do useful work across batches. Replacing it with a fixed τ = 1
and `w_max = 10` is simpler, JIT-friendlier, and has trivial variance
bounds (by clipping).

### 2.5 Why this is strictly simpler than NFQL₃ on every axis

| Concern | NFQL₃ | NFQL₄ |
|---|---|---|
| Extra networks | Q_n (ensemble of 2) over `(s, x_t, t)` | V_φ over `s` |
| Per-step compute | B·n noised-critic forwards + one matched | B one V + k one-step flow evals |
| Hyperparams | `n_noised_actions`, `ess_target`, `r²_target`, `r²_critic_target`, `gate_kappa`, `gate_kappa_critic`, `gate_ema_decay` | `τ`, `w_max`, `r²_V_target`, `r²_critic_target`, `gate_kappa`, `gate_ema_decay` |
| Failure mode | Gate opens on trivial high-t regression | Gate opens iff V tracks π-value, which requires both networks sane |
| Gradient signal at t ≈ 1 | Emergent 0 (from Q_n collapse) | Explicit 0 (from m(t)) — diagnosable |
| What the advantage means | Scale-conditional (broken at t=1) | Standard AWR advantage |

### 2.6 Minimal patch if you want to keep NFQL₃'s architecture

If rebuilding is not desired, the single highest-value change is one line.

Replace `agents/nfql_3.py:147`:

```python
m_t = jnp.maximum(0.0, 1.0 - 2.0 * t.squeeze(-1))        # (B,)
a_local = m_t * (original_q - noised_q)
```

and `agents/nfql_3.py:192`:

```python
bc_weights = 1.0 + m_t * c_gate * trust * (w_exp - 1.0)
```

In addition, compute `R²_qn` from low-t samples only, by splitting the
Q_n loss into `t < 0.5` and `t ≥ 0.5` halves and using only the former
for the EMA. This decouples the gate from the trivial regime without
changing the rest of the design.

---

## 3. Recommended next steps

1. **Before writing NFQL₄**, add one diagnostic to the existing training:
   log `R²_qn` bucketed into `t ∈ [0, 0.25]`, `[0.25, 0.5]`, `[0.5, 0.75]`,
   `[0.75, 1.0]`. Prediction: buckets 3–4 show R² ≈ 0.99 and buckets 1–2
   show R² ≪ 0.75. If confirmed, the patch in §2.6 is well-motivated.
2. Run NFQL₄ against FQL on the cube-triple tasks (where current data
   shows NFQL₃ at 0.00 — a clean floor to compare against).
3. Add seeds on antmaze-large-navigate: the one nfql_3 seed scored 0.000
   with no FQL baseline, so the current result cannot distinguish "method
   failure" from "env/seed noise".

---

## 4. One-line summary

The Rao-Blackwell argument in `nfql_3_theory.md` is mathematically valid
in the continuous-action / infinite-capacity limit, but the finite-network
reality is that the quantity being computed (advantage at t=1) is **zero
by construction**. A shorter honest version of the theorem would be:
*NFQL₃ applies AWR at low t with a soft mask that tapers to a no-op at
high t, gated by a reliability signal.* NFQL₄ states that directly and
removes the Q_n over-engineering.
