import jax
import jax.numpy as jnp

from agents.nfql_7 import NFQL7Agent, get_config as _base_get_config


class NFQL8Agent(NFQL7Agent):
    """NFQL8: V-weighted offline BC, Q_n-weighted online BC.

    This keeps NFQL7's critic, value head, noised critic, and multi-time BC
    loss. The only behavioral change is the actor BC weighting source:

      * offline: use value advantage Q(s, a_data) - V(s)
      * online: use noised-critic residual Q(s, a_data) - Q_n(s, x_t, t)
    """

    def actor_loss(self, batch, grad_params, rng, online):
        batch_size, action_dim = batch['actions'].shape
        rng, x_rng, t_rng = jax.random.split(rng, 3)
        k = self.config['n_actor_time_samples']

        x_0 = jax.random.normal(x_rng, (batch_size, k, action_dim))
        x_1 = batch['actions']
        x_1_exp = x_1[:, None, :]
        t = jax.random.uniform(t_rng, (batch_size, k, 1))
        x_t = (1 - t) * x_0 + t * x_1_exp
        vel = x_1_exp - x_0

        original_qs = self.network.select('critic')(
            batch['observations'], actions=batch['actions']
        )
        original_q = jax.lax.stop_gradient(original_qs.mean(axis=0))

        # Offline weighting source: value advantage Q(s, a_data) - V(s).
        v = self.network.select('value')(batch['observations'])
        v = jax.lax.stop_gradient(v)
        a_v = original_q - v
        v_med = jnp.median(a_v)
        beta_mad_v = 1.4826 * jnp.median(jnp.abs(a_v - v_med))
        beta_mad_v = jnp.maximum(beta_mad_v, 1e-6)
        a_v_norm = (a_v - v_med) / beta_mad_v
        w_v, tau_star_v, ess_achieved_v = self._ess_targeted_weights(
            a_v_norm, jnp.asarray(self.config['ess_target'])
        )

        # Online weighting source: NFQL7 Q_n residual at each sampled flow time.
        obs_tail = batch['observations'].shape[1:]
        obs_exp = jnp.broadcast_to(
            batch['observations'][:, None], (batch_size, k) + obs_tail
        )
        x_t_with_t = jnp.concatenate([x_t, t], axis=-1)
        if self.config['encoder'] is not None:
            obs_flat = obs_exp.reshape((batch_size * k,) + obs_tail)
            actions_flat = x_t_with_t.reshape((batch_size * k, -1))
            noised_qs_flat = self.noised_network.select('noised_critic')(
                obs_flat,
                actions=jax.lax.stop_gradient(actions_flat),
            )
            noised_qs = noised_qs_flat.reshape(
                (noised_qs_flat.shape[0], batch_size, k)
            )
        else:
            noised_qs = self.noised_network.select('noised_critic')(
                obs_exp, actions=jax.lax.stop_gradient(x_t_with_t)
            )
        noised_q = jax.lax.stop_gradient(noised_qs.mean(axis=0))
        noised_q_disagree = jax.lax.stop_gradient(
            jnp.abs(noised_qs[0] - noised_qs[1])
        )

        a_qn = original_q[:, None] - noised_q
        a_qn_flat = a_qn.reshape(-1)
        qn_med = jnp.median(a_qn_flat)
        beta_mad_qn = 1.4826 * jnp.median(jnp.abs(a_qn_flat - qn_med))
        beta_mad_qn = jnp.maximum(beta_mad_qn, 1e-6)
        a_qn_norm = (a_qn_flat - qn_med) / beta_mad_qn
        w_qn_flat, tau_star_qn, ess_achieved_qn = self._ess_targeted_weights(
            a_qn_norm, jnp.asarray(self.config['ess_target'])
        )
        w_qn = w_qn_flat.reshape(batch_size, k)

        delta_med = jnp.maximum(jnp.median(noised_q_disagree.reshape(-1)), 1e-6)
        trust = jnp.exp(-noised_q_disagree / delta_med)

        ema_valid = (self.noised_loss_ema >= 0) & (self.var_q_ema > 0)
        r2_qn = jnp.where(
            ema_valid,
            1.0 - self.noised_loss_ema / jnp.maximum(self.var_q_ema, 1e-8),
            jnp.asarray(-1.0),
        )
        gate_qn = jax.nn.sigmoid(
            (r2_qn - self.config['r2_target']) / self.config['gate_kappa']
        )
        gate_qn = jnp.where(ema_valid, gate_qn, jnp.asarray(0.0))

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

        online_f = jnp.asarray(1.0 if online else 0.0, dtype=jnp.float32)
        v_gate = gate_critic * (1.0 - online_f)
        qn_gate = gate_qn * gate_critic * online_f
        bc_weights = (
            1.0
            + v_gate * (w_v[:, None] - 1.0)
            + qn_gate * trust * (w_qn - 1.0)
        )
        bc_weights = jax.lax.stop_gradient(bc_weights)

        pred = self.network.select('actor_bc_flow')(
            obs_exp, x_t, t, params=grad_params
        )
        per_time_bc_loss = jnp.mean((pred - vel) ** 2, axis=-1)
        bc_flow_loss = jnp.mean(bc_weights * per_time_bc_loss)

        rng, noise_rng = jax.random.split(rng)
        noises = jax.random.normal(noise_rng, (batch_size, action_dim))
        target_flow_actions = self.compute_flow_actions(
            batch['observations'], noises=noises
        )
        actor_actions = self.network.select('actor_onestep_flow')(
            batch['observations'], noises, params=grad_params
        )
        distill_loss = jnp.mean((actor_actions - target_flow_actions) ** 2)

        actor_actions = jnp.clip(actor_actions, -1, 1)
        qs = self.network.select('critic')(
            batch['observations'], actions=actor_actions
        )
        q = jnp.mean(qs, axis=0)

        q_loss = -q.mean()
        if self.config['normalize_q_loss']:
            lam = jax.lax.stop_gradient(1 / jnp.abs(q).mean())
            q_loss = lam * q_loss

        actor_loss = bc_flow_loss + self.config['alpha'] * distill_loss + q_loss

        actions = self.sample_actions(batch['observations'], seed=rng)
        mse = jnp.mean((actions - batch['actions']) ** 2)
        frac_suboptimal = jnp.mean(original_q[:, None] < noised_q)
        frac_below_v = jnp.mean(original_q < v)

        tau_star = (1.0 - online_f) * tau_star_v + online_f * tau_star_qn
        ess_achieved = (
            (1.0 - online_f) * ess_achieved_v + online_f * ess_achieved_qn
        )
        beta_mad = (1.0 - online_f) * beta_mad_v + online_f * beta_mad_qn

        return actor_loss, {
            'actor_loss': actor_loss,
            'bc_flow_loss': bc_flow_loss,
            'distill_loss': distill_loss,
            'q_loss': q_loss,
            'q': q.mean(),
            'v': v.mean(),
            'adv_v_mean': a_v.mean(),
            'adv_v_std': a_v.std(),
            'adv_qn_mean': a_qn.mean(),
            'adv_qn_std': a_qn.std(),
            'frac_suboptimal': frac_suboptimal,
            'frac_below_v': frac_below_v,
            'mse': mse,
            'beta_mad': beta_mad,
            'beta_mad_v': beta_mad_v,
            'beta_mad_qn': beta_mad_qn,
            'tau_star': tau_star,
            'tau_star_v': tau_star_v,
            'tau_star_qn': tau_star_qn,
            'ess_achieved': ess_achieved,
            'ess_achieved_v': ess_achieved_v,
            'ess_achieved_qn': ess_achieved_qn,
            'gate_v': v_gate,
            'gate_qn': gate_qn,
            'gate_critic': gate_critic,
            'gate_c': v_gate + qn_gate,
            'online_phase': online_f,
            'r2_qn': r2_qn,
            'r2_critic': r2_critic,
            'trust_mean': trust.mean(),
            'bc_weight_mean': bc_weights.mean(),
            'bc_weight_max': bc_weights.max(),
            'bc_weight_min': bc_weights.min(),
            'n_actor_time_samples': jnp.asarray(k, dtype=jnp.float32),
            't_mean': t.mean(),
        }


def get_config():
    config = _base_get_config()
    config.agent_name = 'nfql_8'
    return config
