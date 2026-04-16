# Fine-Grained BC Weighting via Local Advantage

## Background

The NFQL actor loss is a weighted sum of three terms:

```
L_actor = L_BC + α · L_distill + L_Q
```

The current weighting scheme applies a binary multiplier to the per-sample BC loss:

```
w(s, a) = 1.2   if  Q(s, a) ≥ Q_n(s, a+ε, t)
w(s, a) = 0.9   otherwise
```

where `Q_n(s, ã, t)` is the **noised-action value network**, trained to approximate:

```
Q_n(s, ã, t)  ≈  E[ Q(s, a′) | a′ + noise ≈ ã,  t ]
```

Because different dataset actions can produce the same or similar noised point `ã` under the
linear interpolation of rectified flow, `Q_n` learns a **conditional mean** of Q-values over the
local action neighbourhood — a neighbourhood baseline.

---

## Proposed Weighting Formula

Both derivations below arrive at the same expression.

Define the **local advantage**:

```
A_local(s, a)  =  Q(s, a)  −  Q_n(s, a+ε, t)
```

Define the **adaptive temperature** (batch-normalised, no extra hyperparameter):

```
β  =  std_{(s,a) ∈ batch} [ A_local(s, a) ]
```

Define the **normalised exponential weight**:

```
w_raw(s, a)  =  exp( A_local(s, a) / β )

w(s, a)      =  w_raw(s, a)  /  mean_{batch}[ w_raw ]
```

The weighted per-sample BC loss becomes:

```
L_BC  =  mean_{batch} [ w(s, a) · ‖ pred − vel ‖² ]
```

Properties of `w(s, a)`:
- `mean(w) = 1` — average learning rate is unchanged; only the relative emphasis shifts
- `w > 1` when `Q(s,a) > Q_n` — action is above its local mean → up-weighted
- `w < 1` when `Q(s,a) < Q_n` — action is below its local mean → down-weighted
- Scale-invariant — dividing by `β = std(A_local)` removes Q-value scale dependence
- No new hyperparameters — `β` is computed from the batch at each step

---

## Derivation A — KL-Constrained Policy Optimisation

### Setup

Standard offline RL with BC regularisation solves:

```
max_π  E_{(s,a)∼π_β}[ log π(a|s) ]
s.t.   E_s[ KL( π(·|s) ‖ π_β(·|s) ) ]  ≤  ε
```

Forming the Lagrangian and solving via KKT conditions gives the closed-form optimal policy
(the AWR result):

```
π*(a|s)  ∝  π_β(a|s) · exp( Q(s,a) / λ )
```

Training a parametric `π_θ` to match `π*` via importance-weighted BC yields the loss:

```
L  =  − E_{(s,a)∼π_β} [ exp( A(s,a) / λ ) · log π_θ(a|s) ]
```

where the standard advantage uses the global state-value baseline:

```
A(s, a)  =  Q(s, a)  −  V(s),     V(s)  =  E_{a′∼π_β}[ Q(s, a′) ]
```

### Substituting the Local Baseline

Replace `V(s)` with the neighbourhood conditional mean `Q_n(s, a+ε, t)`:

```
A_local(s, a)  =  Q(s, a)  −  Q_n(s, a+ε, t)
               ≈  Q(s, a)  −  E_{a′∼N(a, σ²)}[ Q(s, a′) ]
```

`Q_n` depends on **(s, a)**, whereas `V(s)` depends only on **s**. The local baseline therefore
carries strictly more information about the local quality of action `a`.

### Proof of Variance Superiority

By the **law of total variance**, decompose the Q-value variance across the behaviour distribution:

```
Var_{a∼π_β}[ Q(s,a) ]  =  Var[ Q_n(s,a,t) ]  +  E[ Var[ Q(s,a) | neighbourhood ] ]
```

Rearranging:

```
Var[ A_local(s,a) ]  =  E[ Var[ Q(s,a) | neighbourhood ] ]
                      ≤  Var[ Q(s,a) ]
                       =  Var[ A_standard(s,a) ]
```

The inequality is strict whenever `Q` varies across the support of `π_β` (the generic case).

**Corollary.** Since the gradient of the weighted BC loss is:

```
∇_θ L  =  − E_{(s,a)∼π_β} [ w(s,a) · ∇_θ log π_θ(a|s) ]
```

lower-variance weights produce lower-variance policy gradients.
Lower-variance gradients → more stable training → better sample efficiency.

By the **Rao-Blackwell theorem**: any estimator conditioned on a richer sufficient statistic has
variance no greater than the original estimator. Since `A_local` conditions on `(s, a)` while
`A_standard` conditions only on `s`, the theorem guarantees:

```
Var[ A_local ]  ≤  Var[ A_standard ]
```

with equality only if `Q` is constant within every neighbourhood — a degenerate case that does
not arise in practice.

---

## Derivation C — Information-Theoretic (Posterior Selection Probability)

### Local Softmax Policy

Given state `s` and a dataset action `a`, define the **local partition function** over the action
neighbourhood with noise scale `σ` and temperature `β`:

```
Z_local(s, a)  =  E_{a′∼N(a, σ²)}[ exp( Q(s, a′) / β ) ]
```

The **free energy** of the neighbourhood:

```
F(s, a)  =  β · log Z_local(s, a)
```

The probability that a softmax-rational agent selects action `a` *from within its own
neighbourhood* (i.e., prefers `a` over a random local alternative):

```
P( select a | s, neighbourhood )  =  exp( Q(s,a) / β )  /  Z_local(s, a)
                                    =  exp( ( Q(s,a) − F(s,a) ) / β )
```

### Approximating F via Q_n

By **Jensen's inequality** applied to the convex function `exp(·/β)`:

```
Z_local  =  E[ exp(Q/β) ]  ≥  exp( E[Q] / β )  =  exp( Q_n / β )
```

So `F(s,a) ≥ Q_n(s, a+ε, t)` always. A second-order Taylor expansion of `Q` around `a` gives:

```
F(s, a)  ≈  Q_n(s, a+ε, t)  +  (σ²/2β) · ‖∇_a Q(s,a)‖²
```

The curvature correction `(σ²/2β)‖∇Q‖²` is approximately uniform across batch samples and is
absorbed by the batch normalisation step. Therefore:

```
P( select a | s )  ≈  exp( A_local(s, a) / β )     (up to normalisation)
```

**Interpretation:** the weight `w(s,a)` is the posterior probability that a locally-rational agent
would choose action `a` over a randomly drawn neighbour. Actions that are locally superior
receive higher weight; locally inferior actions receive lower weight.

### Proof of Information Superiority over Binary Weights

The binary weight `{w_low, w_high}` is a threshold on the sign of `A_local`.
It discards all magnitude information. The **mutual information** between the weight and the true
local advantage is:

```
I( w_binary ; A_local )  =  H( sign(A_local) )  ≤  log 2      ← at most 1 bit retained

I( w_exp    ; A_local )  =  H( A_local )                       ← full information retained
```

A weight that ignores magnitude treats a marginally suboptimal action identically to a
catastrophically bad one. The exponential weight assigns monotonically increasing influence as
`A_local` increases, preserving the full ordinal and cardinal structure of the local advantage
signal.

---

## Why Adaptive β Is the Correct Normalisation

Dividing by `β = std(A_local)` standardises the exponent into a z-score:

```
A_local / β  →  dimensionless,  std = 1
```

This makes the weight distribution invariant to the absolute scale of `Q` (analogous to the
`normalize_q_loss` flag in the codebase). The batch normalisation `w / mean(w)` then ensures
the BC loss magnitude is preserved — only the *relative* emphasis across samples changes.

Limiting behaviour:
- `β → ∞` (low signal): `w → 1` uniformly — recovers standard, unweighted BC
- `β → 0` (strong signal): `w` concentrates on the highest-A_local sample — greedy selection
- Adaptive `β` self-calibrates to the current signal strength at each step

---

## Comparison Summary

| Property | Binary {w_low, w_high} | Exp( A_local / β ) |
|---|---|---|
| Magnitude-sensitive | No | Yes |
| Scale-invariant | No | Yes (adaptive β) |
| Grounding — optimisation | Heuristic | KL-constrained policy improvement |
| Grounding — probabilistic | Heuristic | Posterior local-selection probability |
| Gradient variance vs standard AWR | — | ≤ Var[A_standard] (Rao-Blackwell) |
| Information content | ≤ 1 bit | Full H(A_local) |
| Extra hyperparameters | 2 (w_high, w_low) | 0 |
