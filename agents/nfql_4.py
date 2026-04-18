"""NFQL₄: repaired version of NFQL₃.

See docs/nfql_4_proposal.md for rationale. Changes vs. NFQL₃:

  §3.1  Gate on the LOW-t portion of the Q_n residual (not the whole batch),
        so the gate no longer opens on the trivial high-t identity regime.
        Concretely: use L_martingale (see §3.3) as the gate signal.

  §3.2  Bias Q_n's training-t sampling:
            t_a ∈ [t_a_min, t_a_max]   (low, default [0.0, 0.5])
            t_b ∈ [t_b_min, t_b_max]   (high, default [0.5, 1.0])
        Concentrates capacity where Q_n actually needs to learn a
        non-trivial conditional expectation.

  §3.3  Two-t martingale bootstrap:
            x_{t_a}, x_{t_b} on the SAME flow path (same x_0, same a).
            L_anchor     = (Q_n(s, x_{t_b}, t_b) − Q(s, a))²        (MC)
            L_martingale = (Q_n(s, x_{t_a}, t_a)
                            − stop_grad Q_n(s, x_{t_b}, t_b))²      (TD-like)
            L_total      = L_anchor + λ_boot · L_martingale
        Justified by the tower/martingale property along the flow path.

  §3.4  Actor-side MAD / median computed within the LOW-t subset of the
        batch, not globally.  High-t samples are driven to bc_weight = 1
        (no-op) regardless of the global normalization.

  §3.5  Drop ESS-targeted temperature and its 14-step bisection.
        Use fixed τ + hard clip at w_max.

  §3.6  Drop the `trust` term (ensemble-disagreement suppression). Its
        polarity may fight the intended low-t signal.

  §3.7  Dual reliability gate retained, but the Q_n side now uses the §3.1
        low-t signal.

  §3.8  Final weight is the natural u_i = 1 + c_gate · (w_exp_i − 1).
        No hand-designed m(t) mask — the (x_t, t)-conditional baseline
        produces the taper structurally.

Diagnostics (§6):
  - Bucketed R² of the MC residual by t: `noised_critic/r2_t_*`.
  - L_anchor and L_martingale reported separately.
"""
import copy
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.encoders import encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import ActorVectorField, Value


class NFQL4Agent(flax.struct.PyTreeNode):
    """Noised Flow Q-learning v4."""

    rng: Any
    network: Any
    noised_network: Any
    # EMA state for the dual R²-gate.
    # noised_loss_ema is the EMA of L_martingale (the low-t gate signal).
    noised_loss_ema: jnp.ndarray
    var_q_ema: jnp.ndarray
    critic_loss_ema: jnp.ndarray
    var_tq_ema: jnp.ndarray
    config: Any = nonpytree_field()

    # ------------------------------------------------------------------
    # Critic loss — same as FQL / NFQL₃.
    # ------------------------------------------------------------------
    def critic_loss(self, batch, grad_params, rng):
        rng, sample_rng = jax.random.split(rng)
        next_actions = self.sample_actions(batch['next_observations'], seed=sample_rng)
        next_actions = jnp.clip(next_actions, -1, 1)

        next_qs = self.network.select('target_critic')(batch['next_observations'], actions=next_actions)
        if self.config['q_agg'] == 'min':
            next_q = next_qs.min(axis=0)
        else:
            next_q = next_qs.mean(axis=0)

        target_q = batch['rewards'] + self.config['discount'] * batch['masks'] * next_q

        q = self.network.select('critic')(batch['observations'], actions=batch['actions'], params=grad_params)
        critic_loss = jnp.square(q - target_q).mean()

        var_target_q = jnp.var(target_q) + 1e-8

        return critic_loss, {
            'critic_loss': critic_loss,
            'var_target_q': var_target_q,
            'q_mean': q.mean(),
            'q_max': q.max(),
            'q_min': q.min(),
        }

    # ------------------------------------------------------------------
    # §3.4 helper: median over a masked subset via sort-and-index.
    # ------------------------------------------------------------------
    @staticmethod
    def _masked_median(values, mask):
        """Median of `values[mask > 0]`. Non-masked positions are pushed to +inf
        so they sort to the tail and are ignored by the median index.

        Robust to NaN/Inf in `values` (replaced with 0 before sorting) and to
        an all-zero mask (returns 0 rather than the sort-artifact +inf).
        """
        safe_values = jnp.where(jnp.isfinite(values), values, jnp.asarray(0.0))
        pushed = jnp.where(mask > 0, safe_values, jnp.inf)
        sorted_vals = jnp.sort(pushed)
        n_valid = mask.sum().astype(jnp.int32)
        idx = jnp.clip(n_valid // 2, 0, values.shape[0] - 1)
        return jnp.where(n_valid > 0, sorted_vals[idx], jnp.asarray(0.0))

    # ------------------------------------------------------------------
    # Actor loss — §3.4 / §3.5 / §3.6 / §3.7 / §3.8.
    # ------------------------------------------------------------------
    def actor_loss(self, batch, grad_params, rng):
        batch_size, action_dim = batch['actions'].shape
        rng, x_rng, t_rng = jax.random.split(rng, 3)

        # Shared flow interpolation. The actor-side t stays U(0,1) because
        # the BC-flow needs coverage over all t.
        x_0 = jax.random.normal(x_rng, (batch_size, action_dim))
        x_1 = batch['actions']
        t = jax.random.uniform(t_rng, (batch_size, 1))                  # (B, 1)
        x_t = (1 - t) * x_0 + t * x_1                                   # (B, A)
        vel = x_1 - x_0

        # Q(s, a) at the dataset action.
        original_qs = self.network.select('critic')(batch['observations'], actions=batch['actions'])
        original_q = jax.lax.stop_gradient(original_qs.mean(axis=0))    # (B,)

        # Q_n(s, x_t, t) — timestep-matched.
        x_t_with_t = jnp.concatenate([x_t, t], axis=-1)                 # (B, A+1)
        noised_qs = self.noised_network.select('noised_critic')(
            batch['observations'], actions=jax.lax.stop_gradient(x_t_with_t)
        )                                                               # (2, B)
        noised_q = jax.lax.stop_gradient(noised_qs.mean(axis=0))        # (B,)

        # Timestep-matched local advantage.
        a_local = original_q - noised_q                                 # (B,)

        # -------- §3.4: MAD normalization over low-t subset only --------
        t_sq = t.squeeze(-1)                                            # (B,)
        low_th = self.config['low_t_threshold']
        mask_low = (t_sq < low_th).astype(jnp.float32)                  # (B,)
        mask_low_sum = jnp.maximum(mask_low.sum(), 1.0)

        a_med = self._masked_median(a_local, mask_low)
        β_MAD = 1.4826 * self._masked_median(jnp.abs(a_local - a_med), mask_low)
        # Floor raised from 1e-6 → 1e-3: the old floor permits a_norm to hit
        # O(1e6) when advantages are tightly clustered, which then overflows
        # exp() below. 1e-3 is still well below any physically meaningful
        # Q-scale and leaves the MAD normalization effectively intact when
        # real signal exists.
        β_MAD = jnp.maximum(β_MAD, 1e-3)
        a_norm = (a_local - a_med) / β_MAD
        # Clamp to a robust-statistics range. MAD-normalized samples are
        # essentially "sigmas"; anything past ±10 is a numerical artifact,
        # not signal. This hard-caps the exp input.
        a_norm = jnp.clip(a_norm, -10.0, 10.0)

        # -------- §3.5: fixed τ + hard clip, log-sum-exp stabilized --------
        tau_w = self.config['tau_weight']
        w_max = self.config['w_max']
        logits = a_norm / tau_w
        # Subtract the max over the LOW-t subset before exp. This is the
        # standard LSE stabilization that NFQL₃'s _ess_targeted_weights
        # had and NFQL₄ accidentally dropped along with the bisection.
        logits_low = jnp.where(mask_low > 0, logits, -jnp.inf)
        logit_max = jnp.max(logits_low)
        logit_max = jnp.where(jnp.isfinite(logit_max), logit_max, jnp.asarray(0.0))
        w_raw = jnp.exp(logits - logit_max)
        mean_low = jnp.sum(w_raw * mask_low) / mask_low_sum
        w_exp = w_raw / jnp.maximum(mean_low, 1e-8)
        w_exp = jnp.clip(w_exp, 0.0, w_max)
        # High-t samples: weight is exactly 1 (no-op) — structural taper.
        w_exp = jnp.where(mask_low > 0, w_exp, 1.0)

        # -------- §3.7: dual reliability gate --------
        # Q_n side: EMA of L_martingale vs Var[Q].
        ema_valid_qn = (self.noised_loss_ema >= 0) & (self.var_q_ema > 0)
        r2_qn = jnp.where(
            ema_valid_qn,
            1.0 - self.noised_loss_ema / jnp.maximum(self.var_q_ema, 1e-8),
            jnp.asarray(-1.0),
        )
        gate_qn = jax.nn.sigmoid(
            (r2_qn - self.config['r2_target']) / self.config['gate_kappa']
        )
        gate_qn = jnp.where(ema_valid_qn, gate_qn, jnp.asarray(0.0))

        # Critic side: EMA of L_critic vs Var[target_Q].
        critic_valid = (self.critic_loss_ema >= 0) & (self.var_tq_ema > 0)
        r2_critic = jnp.where(
            critic_valid,
            1.0 - self.critic_loss_ema / jnp.maximum(self.var_tq_ema, 1e-8),
            jnp.asarray(-1.0),
        )
        gate_critic = jax.nn.sigmoid(
            (r2_critic - self.config['r2_critic_target']) / self.config['gate_kappa_critic']
        )
        gate_critic = jnp.where(critic_valid, gate_critic, jnp.asarray(0.0))

        c_gate = gate_qn * gate_critic

        # -------- §3.8: final per-sample BC weight --------
        # Guard against 0 * inf = NaN when gate=0 (early training) and w_exp
        # happens to be non-finite from an earlier numerical hiccup.
        safe_w_exp = jnp.where(jnp.isfinite(w_exp), w_exp, 1.0)
        bc_weights = 1.0 + c_gate * (safe_w_exp - 1.0)
        bc_weights = jnp.where(jnp.isfinite(bc_weights), bc_weights, 1.0)
        bc_weights = jax.lax.stop_gradient(bc_weights)

        # -------- BC flow loss --------
        pred = self.network.select('actor_bc_flow')(
            batch['observations'], x_t, t, params=grad_params
        )
        per_sample_bc_loss = jnp.mean((pred - vel) ** 2, axis=-1)       # (B,)
        bc_flow_loss = jnp.mean(bc_weights * per_sample_bc_loss)

        # -------- Distillation loss (unchanged from FQL) --------
        rng, noise_rng = jax.random.split(rng)
        noises = jax.random.normal(noise_rng, (batch_size, action_dim))
        target_flow_actions = self.compute_flow_actions(batch['observations'], noises=noises)
        actor_actions = self.network.select('actor_onestep_flow')(
            batch['observations'], noises, params=grad_params
        )
        distill_loss = jnp.mean((actor_actions - target_flow_actions) ** 2)

        # -------- Q loss --------
        actor_actions = jnp.clip(actor_actions, -1, 1)
        qs = self.network.select('critic')(batch['observations'], actions=actor_actions)
        q = jnp.mean(qs, axis=0)

        q_loss = -q.mean()
        if self.config['normalize_q_loss']:
            lam = jax.lax.stop_gradient(1 / jnp.abs(q).mean())
            q_loss = lam * q_loss

        actor_loss = bc_flow_loss + self.config['alpha'] * distill_loss + q_loss

        # -------- Diagnostics --------
        rng, diag_rng = jax.random.split(rng)
        actions_sampled = self.sample_actions(batch['observations'], seed=diag_rng)
        mse = jnp.mean((actions_sampled - batch['actions']) ** 2)
        frac_suboptimal = jnp.mean(original_q < noised_q)
        bc_weight_low_mean = jnp.sum(bc_weights * mask_low) / mask_low_sum

        return actor_loss, {
            'actor_loss': actor_loss,
            'bc_flow_loss': bc_flow_loss,
            'distill_loss': distill_loss,
            'q_loss': q_loss,
            'q': q.mean(),
            'frac_suboptimal': frac_suboptimal,
            'mse': mse,
            'a_med': a_med,
            'beta_mad': β_MAD,
            'n_low_frac': mask_low.mean(),
            'gate_qn': gate_qn,
            'gate_critic': gate_critic,
            'gate_c': c_gate,
            'r2_qn': r2_qn,
            'r2_critic': r2_critic,
            'bc_weight_mean': bc_weights.mean(),
            'bc_weight_max': bc_weights.max(),
            'bc_weight_min': bc_weights.min(),
            'bc_weight_low_mean': bc_weight_low_mean,
            't_mean': t.mean(),
        }

    # ------------------------------------------------------------------
    # Noised critic loss — §3.2 + §3.3.
    # ------------------------------------------------------------------
    def noised_critic_loss(self, batch, grad_params, rng):
        batch_size, action_dim = batch['actions'].shape
        n = self.config['n_noised_actions']
        lambda_boot = self.config['lambda_boot']

        original_qs = self.network.select('critic')(batch['observations'], actions=batch['actions'])
        original_q = jax.lax.stop_gradient(original_qs.mean(axis=0))    # (B,)

        rng, x0_rng, ta_rng, tb_rng = jax.random.split(rng, 4)

        # Shared x_0 → t_a and t_b are on the SAME flow path per sample.
        x_0 = jax.random.normal(x0_rng, (batch_size, n, action_dim))    # (B, n, A)
        a_expanded = batch['actions'][:, None, :]                       # (B, 1, A)

        # §3.2: biased t sampling — t_a in the low band, t_b in the high band.
        t_a = jax.random.uniform(
            ta_rng, (batch_size, n, 1),
            minval=self.config['t_a_min'], maxval=self.config['t_a_max'],
        )
        t_b = jax.random.uniform(
            tb_rng, (batch_size, n, 1),
            minval=self.config['t_b_min'], maxval=self.config['t_b_max'],
        )

        x_ta = (1 - t_a) * x_0 + t_a * a_expanded                       # (B, n, A)
        x_tb = (1 - t_b) * x_0 + t_b * a_expanded                       # (B, n, A)

        x_ta_with_t = jnp.concatenate([x_ta, t_a], axis=-1)             # (B, n, A+1)
        x_tb_with_t = jnp.concatenate([x_tb, t_b], axis=-1)             # (B, n, A+1)

        # Stack and run a single forward covering both t_a and t_b — saves
        # one noised-critic forward vs. calling the module twice.
        x_stacked = jnp.concatenate([x_ta_with_t, x_tb_with_t], axis=1)  # (B, 2n, A+1)

        obs_expanded = jnp.broadcast_to(
            batch['observations'][:, None],
            (batch_size, 2 * n) + batch['observations'].shape[1:],
        )

        if self.config['encoder'] is not None:
            obs_flat = obs_expanded.reshape((batch_size * 2 * n,) + batch['observations'].shape[1:])
            acts_flat = x_stacked.reshape((batch_size * 2 * n, -1))
            qn_flat = self.noised_network.select('noised_critic')(
                obs_flat, actions=acts_flat, params=grad_params
            )
            qn = qn_flat.reshape((qn_flat.shape[0], batch_size, 2 * n))
        else:
            qn = self.noised_network.select('noised_critic')(
                obs_expanded, actions=x_stacked, params=grad_params
            )                                                           # (2, B, 2n)

        qn_ta = qn[:, :, :n]                                            # (2, B, n)
        qn_tb = qn[:, :, n:]                                            # (2, B, n)

        qn_ta_mean = qn_ta.mean(axis=0)                                 # (B, n)
        qn_tb_mean = qn_tb.mean(axis=0)                                 # (B, n)

        # -------- §3.3 anchor: MC regression at the easy end --------
        target_anchor = original_q[:, None]                             # (B, 1)
        anchor_sq = (qn_tb_mean - target_anchor) ** 2                   # (B, n)
        anchor_loss = anchor_sq.mean()

        # -------- §3.3 martingale: bootstrap from stop_grad(Q_n(t_b)) --------
        bootstrap_target = jax.lax.stop_gradient(qn_tb_mean)            # (B, n)
        martingale_sq = (qn_ta_mean - bootstrap_target) ** 2            # (B, n)
        martingale_loss = martingale_sq.mean()

        total = anchor_loss + lambda_boot * martingale_loss

        var_q = jnp.var(original_q) + 1e-8

        # -------- §6.1 diagnostic: bucketed MC R² --------
        # Use MC residual (vs Q(s,a)) at all 2n points, bucketed by t.
        all_t = jnp.concatenate([t_a.squeeze(-1), t_b.squeeze(-1)], axis=1)   # (B, 2n)
        all_mc_sq = jnp.concatenate([
            (qn_ta_mean - target_anchor) ** 2,
            (qn_tb_mean - target_anchor) ** 2,
        ], axis=1)                                                      # (B, 2n)

        def _bucket_r2(lo, hi):
            mask = ((all_t >= lo) & (all_t < hi)).astype(jnp.float32)
            mse_bin = (all_mc_sq * mask).sum() / jnp.maximum(mask.sum(), 1.0)
            return 1.0 - mse_bin / var_q

        info = {
            'var_q': var_q,
            'noised_critic_loss': total,
            'anchor_loss': anchor_loss,
            'martingale_loss': martingale_loss,
            # Gate signal: L_martingale (low-t quality proxy).
            'gate_signal': martingale_loss,
            'noised_q_ta_mean': qn_ta_mean.mean(),
            'noised_q_ta_std': qn_ta_mean.std(),
            'noised_q_tb_mean': qn_tb_mean.mean(),
            'noised_q_tb_std': qn_tb_mean.std(),
            't_a_mean': t_a.mean(),
            't_b_mean': t_b.mean(),
            'r2_t_0.00_0.25': _bucket_r2(0.00, 0.25),
            'r2_t_0.25_0.50': _bucket_r2(0.25, 0.50),
            'r2_t_0.50_0.75': _bucket_r2(0.50, 0.75),
            'r2_t_0.75_1.00': _bucket_r2(0.75, 1.00 + 1e-6),
        }
        return total, info

    # ------------------------------------------------------------------
    # Combined loss.
    # ------------------------------------------------------------------
    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        info = {}
        rng = rng if rng is not None else self.rng
        rng, actor_rng, critic_rng = jax.random.split(rng, 3)

        critic_loss, critic_info = self.critic_loss(batch, grad_params, critic_rng)
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v

        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        loss = critic_loss + actor_loss
        return loss, info

    def target_update(self, network, module_name):
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
            self.network.params[f'modules_{module_name}'],
            self.network.params[f'modules_target_{module_name}'],
        )
        network.params[f'modules_target_{module_name}'] = new_target_params

    @jax.jit
    def update(self, batch, online=False):
        del online
        new_rng, rng = jax.random.split(self.rng)
        rng, fql_rng, noised_rng = jax.random.split(rng, 3)

        def fql_loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=fql_rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=fql_loss_fn)
        self.target_update(new_network, 'critic')

        def noised_loss_fn(grad_params):
            return self.noised_critic_loss(batch, grad_params, rng=noised_rng)

        new_noised_network, noised_info = self.noised_network.apply_loss_fn(loss_fn=noised_loss_fn)
        for k, v in noised_info.items():
            info[f'noised_critic/{k}'] = v

        # ---- Update dual R²-gate EMAs ---------------------------------
        decay = self.config['gate_ema_decay']
        uninitialised = self.noised_loss_ema < 0

        # §3.1: Q_n gate uses the MARTINGALE loss, not the full loss.
        cur_gate_signal = noised_info['gate_signal']
        cur_var_q = noised_info['var_q']
        cur_critic_loss = info['critic/critic_loss']
        cur_var_tq = info['critic/var_target_q']

        def _ema_step(prev, cur, fallback_if_first):
            # On the first update, initialize to `cur` when finite, else to a
            # safe fallback. On later steps, skip NaN/Inf updates so one bad
            # batch can't poison the EMA permanently.
            init_val = jnp.where(jnp.isfinite(cur), cur, fallback_if_first)
            step_val = jnp.where(
                jnp.isfinite(cur),
                decay * prev + (1 - decay) * cur,
                prev,
            )
            return jnp.where(uninitialised, init_val, step_val)

        new_noised_ema = _ema_step(self.noised_loss_ema, cur_gate_signal, jnp.asarray(1.0))
        new_var_q_ema = _ema_step(self.var_q_ema, cur_var_q, jnp.asarray(1.0))
        new_critic_ema = _ema_step(self.critic_loss_ema, cur_critic_loss, jnp.asarray(1.0))
        new_var_tq_ema = _ema_step(self.var_tq_ema, cur_var_tq, jnp.asarray(1.0))

        info['gate/noised_loss_ema'] = new_noised_ema           # EMA of L_martingale
        info['gate/var_q_ema'] = new_var_q_ema
        info['gate/r2_qn_ema'] = 1.0 - new_noised_ema / jnp.maximum(new_var_q_ema, 1e-8)
        info['gate/critic_loss_ema'] = new_critic_ema
        info['gate/var_tq_ema'] = new_var_tq_ema
        info['gate/r2_critic_ema'] = 1.0 - new_critic_ema / jnp.maximum(new_var_tq_ema, 1e-8)

        return self.replace(
            network=new_network,
            noised_network=new_noised_network,
            noised_loss_ema=new_noised_ema,
            var_q_ema=new_var_q_ema,
            critic_loss_ema=new_critic_ema,
            var_tq_ema=new_var_tq_ema,
            rng=new_rng,
        ), info

    # ------------------------------------------------------------------
    # Inference — identical to FQL / NFQL / NFQL₂ / NFQL₃.
    # ------------------------------------------------------------------
    @jax.jit
    def sample_actions(self, observations, seed=None, temperature=1.0):
        action_seed, noise_seed = jax.random.split(seed)
        noises = jax.random.normal(
            action_seed,
            (*observations.shape[: -len(self.config['ob_dims'])],
             self.config['action_dim']),
        )
        actions = self.network.select('actor_onestep_flow')(observations, noises)
        actions = jnp.clip(actions, -1, 1)
        return actions

    @jax.jit
    def compute_flow_actions(self, observations, noises):
        if self.config['encoder'] is not None:
            observations = self.network.select('actor_bc_flow_encoder')(observations)
        actions = noises
        for i in range(self.config['flow_steps']):
            t = jnp.full((*observations.shape[:-1], 1), i / self.config['flow_steps'])
            vels = self.network.select('actor_bc_flow')(observations, actions, t, is_encoded=True)
            actions = actions + vels / self.config['flow_steps']
        actions = jnp.clip(actions, -1, 1)
        return actions

    # ------------------------------------------------------------------
    # Factory.
    # ------------------------------------------------------------------
    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        rng = jax.random.PRNGKey(seed)
        rng, init_rng, noised_init_rng = jax.random.split(rng, 3)

        ex_times = ex_actions[..., :1]
        ob_dims = ex_observations.shape[1:]
        action_dim = ex_actions.shape[-1]

        encoders = dict()
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            encoders['critic'] = encoder_module()
            encoders['actor_bc_flow'] = encoder_module()
            encoders['actor_onestep_flow'] = encoder_module()
            encoders['noised_critic'] = encoder_module()

        critic_def = Value(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            num_ensembles=2,
            encoder=encoders.get('critic'),
        )
        actor_bc_flow_def = ActorVectorField(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            layer_norm=config['actor_layer_norm'],
            encoder=encoders.get('actor_bc_flow'),
        )
        actor_onestep_flow_def = ActorVectorField(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            layer_norm=config['actor_layer_norm'],
            encoder=encoders.get('actor_onestep_flow'),
        )

        network_info = dict(
            critic=(critic_def, (ex_observations, ex_actions)),
            target_critic=(copy.deepcopy(critic_def), (ex_observations, ex_actions)),
            actor_bc_flow=(actor_bc_flow_def, (ex_observations, ex_actions, ex_times)),
            actor_onestep_flow=(actor_onestep_flow_def, (ex_observations, ex_actions)),
        )
        if encoders.get('actor_bc_flow') is not None:
            network_info['actor_bc_flow_encoder'] = (encoders.get('actor_bc_flow'), (ex_observations,))
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network.params
        params['modules_target_critic'] = params['modules_critic']

        ex_noised_actions_with_t = jnp.concatenate([ex_actions, ex_times], axis=-1)
        noised_critic_def = Value(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            num_ensembles=2,
            encoder=encoders.get('noised_critic'),
        )
        noised_network_def = ModuleDict({'noised_critic': noised_critic_def})
        noised_network_tx = optax.adam(learning_rate=config['lr'])
        noised_network_params = noised_network_def.init(
            noised_init_rng, noised_critic=(ex_observations, ex_noised_actions_with_t)
        )['params']
        noised_network = TrainState.create(noised_network_def, noised_network_params, tx=noised_network_tx)

        config['ob_dims'] = ob_dims
        config['action_dim'] = action_dim

        sentinel = jnp.asarray(-1.0, dtype=jnp.float32)
        return cls(
            rng,
            network=network,
            noised_network=noised_network,
            noised_loss_ema=sentinel,
            var_q_ema=sentinel,
            critic_loss_ema=sentinel,
            var_tq_ema=sentinel,
            config=flax.core.FrozenDict(**config),
        )


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name='nfql_4',
            ob_dims=ml_collections.config_dict.placeholder(list),
            action_dim=ml_collections.config_dict.placeholder(int),
            lr=3e-4,
            batch_size=256,
            actor_hidden_dims=(512, 512, 512, 512),
            value_hidden_dims=(512, 512, 512, 512),
            layer_norm=True,
            actor_layer_norm=False,
            discount=0.99,
            tau=0.005,
            q_agg='mean',
            alpha=10.0,
            flow_steps=10,
            normalize_q_loss=False,
            encoder=ml_collections.config_dict.placeholder(str),
            # -------- Noised critic (§3.2 + §3.3) --------
            n_noised_actions=4,          # interpolants per sample per side
            t_a_min=0.0,                 # low-t training band
            t_a_max=0.5,
            t_b_min=0.5,                 # high-t anchor band
            t_b_max=1.0,
            lambda_boot=0.5,             # weight on the martingale term
            # -------- Actor-side weighting (§3.4 / §3.5) --------
            low_t_threshold=0.5,         # t < this ⇒ "low-t subset"
            tau_weight=1.0,              # fixed exponential temperature
            w_max=10.0,                  # hard clip on bc weight
            # -------- Dual R²-gate (§3.7) --------
            r2_target=0.5,               # looser than NFQL₃: L_martingale has an
                                         # irreducible floor (Var[Q|x_ta,t_a]),
                                         # so R²_qn saturates below 1.
            gate_kappa=0.05,
            r2_critic_target=0.5,
            gate_kappa_critic=0.05,
            gate_ema_decay=0.999,
        )
    )
    return config
