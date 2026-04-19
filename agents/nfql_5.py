import copy
import functools
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.encoders import encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import ActorVectorField, Value


class NFQL5Agent(flax.struct.PyTreeNode):
    """Noised Flow Q-learning v5 (NFQL5).

    Restores NFQL3's actor recipe (ESS-targeted tau*, global MAD, trust term,
    dual R^2 gate) and adds a V(s) head that pins Q_n's t=0 boundary.

    Root-cause fix for NFQL4:
      NFQL4's two-t martingale bootstrap collapsed Q_n to Q(s, a_i) regardless
      of t, so the (x_t, t)-conditional baseline was never realized.  NFQL5
      pins BOTH boundaries of Q_n along the flow:
        t=1:  Q_n(s, a,   1) -> Q(s, a)        (NFQL3 anchor, kept)
        t=0:  Q_n(s, x_0, 0) -> V(s) = E_a Q   (NEW via auxiliary V head)
      With both endpoints anchored, the trivial fixed point Q_n equal to
      Q(s, a_i) for all t is excluded.

    V(s) is trained as a Monte-Carlo estimate of E_{a ~ pi}[Q(s, a)] with
    n_v_samples flow-policy samples per state and the target critic for
    stability.
    """

    rng: Any
    network: Any
    noised_network: Any
    noised_loss_ema: jnp.ndarray
    var_q_ema: jnp.ndarray
    critic_loss_ema: jnp.ndarray
    var_tq_ema: jnp.ndarray
    config: Any = nonpytree_field()

    # ------------------------------------------------------------------
    # Critic loss (identical to NFQL3).
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
    # Value loss — V(s) is trained as MC estimate of E_{a~pi}[Q_tgt(s, a)].
    # ------------------------------------------------------------------
    def value_loss(self, batch, grad_params, rng):
        batch_size, action_dim = batch['actions'].shape
        n = self.config['n_v_samples']

        # Sample n flow-policy actions per state using the distilled actor.
        noises = jax.random.normal(rng, (batch_size, n, action_dim))
        obs_tail = batch['observations'].shape[1:]
        obs_exp = jnp.broadcast_to(
            batch['observations'][:, None], (batch_size, n) + obs_tail
        )
        obs_flat = obs_exp.reshape((batch_size * n,) + obs_tail)
        noises_flat = noises.reshape(batch_size * n, action_dim)

        a_samples_flat = self.network.select('actor_onestep_flow')(obs_flat, noises_flat)
        a_samples_flat = jnp.clip(a_samples_flat, -1, 1)
        a_samples_flat = jax.lax.stop_gradient(a_samples_flat)

        q_samples_flat = self.network.select('target_critic')(
            obs_flat, actions=a_samples_flat
        )  # (ensemble, B*n)
        ensemble = q_samples_flat.shape[0]
        q_samples = q_samples_flat.reshape(ensemble, batch_size, n)
        v_target = jax.lax.stop_gradient(q_samples.mean(axis=(0, 2)))  # (B,)

        v_pred = self.network.select('value')(
            batch['observations'], params=grad_params
        )  # (B,)  (num_ensembles=1 → no ensemble axis)

        loss = jnp.mean((v_pred - v_target) ** 2)
        return loss, {
            'value_loss': loss,
            'v_pred_mean': v_pred.mean(),
            'v_target_mean': v_target.mean(),
            'v_target_std': v_target.std(),
        }

    # ------------------------------------------------------------------
    # ESS-targeted temperature (from NFQL3).
    # ------------------------------------------------------------------
    @staticmethod
    def _ess_targeted_weights(a_norm, ess_target, n_iters=14):
        b = a_norm.shape[0]

        def ess_over_b(log_tau):
            tau = jnp.exp(log_tau)
            z = a_norm / tau
            z = z - jnp.max(z)
            w = jnp.exp(z)
            s1 = jnp.sum(w)
            s2 = jnp.sum(w * w)
            return (s1 * s1) / (b * s2 + 1e-12)

        log_lo = jnp.log(1e-3)
        log_hi = jnp.log(1e3)

        def body(_, state):
            lo, hi = state
            mid = 0.5 * (lo + hi)
            ess_mid = ess_over_b(mid)
            new_lo = jnp.where(ess_mid < ess_target, mid, lo)
            new_hi = jnp.where(ess_mid < ess_target, hi, mid)
            return (new_lo, new_hi)

        lo, hi = jax.lax.fori_loop(0, n_iters, body, (log_lo, log_hi))
        log_tau = 0.5 * (lo + hi)
        tau = jnp.exp(log_tau)

        z = a_norm / tau
        z = z - jnp.max(z)
        w = jnp.exp(z)
        w_norm = w / (jnp.mean(w) + 1e-12)
        ess_achieved = ess_over_b(log_tau)
        return w_norm, tau, ess_achieved

    # ------------------------------------------------------------------
    # Actor loss (identical to NFQL3 — timestep-matched weighting).
    # ------------------------------------------------------------------
    def actor_loss(self, batch, grad_params, rng):
        batch_size, action_dim = batch['actions'].shape
        rng, x_rng, t_rng = jax.random.split(rng, 3)

        x_0 = jax.random.normal(x_rng, (batch_size, action_dim))
        x_1 = batch['actions']
        t = jax.random.uniform(t_rng, (batch_size, 1))
        x_t = (1 - t) * x_0 + t * x_1
        vel = x_1 - x_0

        original_qs = self.network.select('critic')(batch['observations'], actions=batch['actions'])
        original_q = jax.lax.stop_gradient(original_qs.mean(axis=0))

        x_t_with_t = jnp.concatenate([x_t, t], axis=-1)
        noised_qs = self.noised_network.select('noised_critic')(
            batch['observations'], actions=jax.lax.stop_gradient(x_t_with_t)
        )
        noised_q = jax.lax.stop_gradient(noised_qs.mean(axis=0))
        noised_q_disagree = jax.lax.stop_gradient(
            jnp.abs(noised_qs[0] - noised_qs[1])
        )

        a_local = original_q - noised_q

        a_med = jnp.median(a_local)
        beta_mad = 1.4826 * jnp.median(jnp.abs(a_local - a_med))
        beta_mad = jnp.maximum(beta_mad, 1e-6)
        a_norm = (a_local - a_med) / beta_mad

        w_exp, tau_star, ess_achieved = self._ess_targeted_weights(
            a_norm, jnp.asarray(self.config['ess_target'])
        )

        delta_med = jnp.maximum(jnp.median(noised_q_disagree), 1e-6)
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

        c_gate = gate_qn * gate_critic

        bc_weights = 1.0 + c_gate * trust * (w_exp - 1.0)
        bc_weights = jax.lax.stop_gradient(bc_weights)

        pred = self.network.select('actor_bc_flow')(
            batch['observations'], x_t, t, params=grad_params
        )
        per_sample_bc_loss = jnp.mean((pred - vel) ** 2, axis=-1)
        bc_flow_loss = jnp.mean(bc_weights * per_sample_bc_loss)

        rng, noise_rng = jax.random.split(rng)
        noises = jax.random.normal(noise_rng, (batch_size, action_dim))
        target_flow_actions = self.compute_flow_actions(batch['observations'], noises=noises)
        actor_actions = self.network.select('actor_onestep_flow')(
            batch['observations'], noises, params=grad_params
        )
        distill_loss = jnp.mean((actor_actions - target_flow_actions) ** 2)

        actor_actions = jnp.clip(actor_actions, -1, 1)
        qs = self.network.select('critic')(batch['observations'], actions=actor_actions)
        q = jnp.mean(qs, axis=0)

        q_loss = -q.mean()
        if self.config['normalize_q_loss']:
            lam = jax.lax.stop_gradient(1 / jnp.abs(q).mean())
            q_loss = lam * q_loss

        actor_loss = bc_flow_loss + self.config['alpha'] * distill_loss + q_loss

        actions = self.sample_actions(batch['observations'], seed=rng)
        mse = jnp.mean((actions - batch['actions']) ** 2)
        frac_suboptimal = jnp.mean(original_q < noised_q)

        return actor_loss, {
            'actor_loss': actor_loss,
            'bc_flow_loss': bc_flow_loss,
            'distill_loss': distill_loss,
            'q_loss': q_loss,
            'q': q.mean(),
            'frac_suboptimal': frac_suboptimal,
            'mse': mse,
            'beta_mad': beta_mad,
            'tau_star': tau_star,
            'ess_achieved': ess_achieved,
            'gate_qn': gate_qn,
            'gate_critic': gate_critic,
            'gate_c': c_gate,
            'r2_qn': r2_qn,
            'r2_critic': r2_critic,
            'trust_mean': trust.mean(),
            'bc_weight_mean': bc_weights.mean(),
            'bc_weight_max': bc_weights.max(),
            'bc_weight_min': bc_weights.min(),
            't_mean': t.mean(),
        }

    # ------------------------------------------------------------------
    # Noised critic loss — NFQL3 anchor plus a V-anchor at t=0, with
    # bucketed R^2_t diagnostics (ported from NFQL4).
    # ------------------------------------------------------------------
    def noised_critic_loss(self, batch, grad_params, rng):
        batch_size, action_dim = batch['actions'].shape
        n = self.config['n_noised_actions']

        original_qs = self.network.select('critic')(batch['observations'], actions=batch['actions'])
        original_q = jax.lax.stop_gradient(original_qs.mean(axis=0))

        rng, x0_rng, t_rng, xa_rng = jax.random.split(rng, 4)

        # ---- Uniform-t anchor (Q_n -> Q(s, a) along the flow) -------------
        x_0 = jax.random.normal(x0_rng, (batch_size, n, action_dim))
        t = jax.random.uniform(t_rng, (batch_size, n, 1))
        a_expanded = batch['actions'][:, None, :]
        x_t = (1 - t) * x_0 + t * a_expanded
        x_t_with_t = jnp.concatenate([x_t, t], axis=-1)

        obs_tail = batch['observations'].shape[1:]
        obs_expanded = jnp.broadcast_to(
            batch['observations'][:, None], (batch_size, n) + obs_tail
        )

        if self.config['encoder'] is not None:
            obs_flat = obs_expanded.reshape((batch_size * n,) + obs_tail)
            actions_flat = x_t_with_t.reshape((batch_size * n, -1))
            noised_qs_flat = self.noised_network.select('noised_critic')(
                obs_flat, actions=actions_flat, params=grad_params
            )
            noised_qs = noised_qs_flat.reshape((noised_qs_flat.shape[0], batch_size, n))
        else:
            noised_qs = self.noised_network.select('noised_critic')(
                obs_expanded, actions=x_t_with_t, params=grad_params
            )
        noised_q = noised_qs.mean(axis=0)  # (B, n)

        target_q = original_q[:, None]  # (B, 1)
        anchor_sq = (noised_q - target_q) ** 2  # (B, n)
        anchor_loss = anchor_sq.mean()

        # ---- V-anchor at t=0 (NEW) ---------------------------------------
        x_0_v = jax.random.normal(xa_rng, (batch_size, action_dim))
        t_zero = jnp.zeros((batch_size, 1))
        xt_with_t_zero = jnp.concatenate([x_0_v, t_zero], axis=-1)

        qn_t0_all = self.noised_network.select('noised_critic')(
            batch['observations'], actions=xt_with_t_zero, params=grad_params
        )  # (ensemble, B)
        qn_t0 = qn_t0_all.mean(axis=0)  # (B,)

        v_for_anchor = self.network.select('value')(batch['observations'])  # (B,)
        v_for_anchor = jax.lax.stop_gradient(v_for_anchor)

        v_anchor_loss = jnp.mean((qn_t0 - v_for_anchor) ** 2)

        total = anchor_loss + self.config['lambda_v_anchor'] * v_anchor_loss

        var_q = jnp.var(original_q) + 1e-8

        # ---- Bucketed R^2_t diagnostics (not gated) ----------------------
        flat_t = t.reshape(-1)                        # (B*n,)
        flat_sq = anchor_sq.reshape(-1)              # (B*n,)

        def _bucket_r2(lo, hi):
            mask = ((flat_t >= lo) & (flat_t < hi)).astype(jnp.float32)
            mse_bin = (flat_sq * mask).sum() / jnp.maximum(mask.sum(), 1.0)
            return 1.0 - mse_bin / var_q

        info = {
            'noised_critic_loss': total,
            'anchor_loss': anchor_loss,
            'v_anchor_loss': v_anchor_loss,
            'var_q': var_q,
            'noised_q_mean': noised_q.mean(),
            'noised_q_std': noised_q.std(),
            'qn_t0_mean': qn_t0.mean(),
            'v_at_anchor_mean': v_for_anchor.mean(),
            'r2_t_0.00_0.25': _bucket_r2(0.00, 0.25),
            'r2_t_0.25_0.50': _bucket_r2(0.25, 0.50),
            'r2_t_0.50_0.75': _bucket_r2(0.50, 0.75),
            'r2_t_0.75_1.00': _bucket_r2(0.75, 1.00 + 1e-6),
        }
        return total, info

    # ------------------------------------------------------------------
    # Combined main-network loss (critic + actor + value).
    # Noised-critic loss is optimized on a separate network.
    # ------------------------------------------------------------------
    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        info = {}
        rng = rng if rng is not None else self.rng
        rng, actor_rng, critic_rng, value_rng = jax.random.split(rng, 4)

        critic_loss, critic_info = self.critic_loss(batch, grad_params, critic_rng)
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v

        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        v_loss, v_info = self.value_loss(batch, grad_params, value_rng)
        for k, v in v_info.items():
            info[f'value/{k}'] = v

        loss = critic_loss + actor_loss + self.config['value_loss_weight'] * v_loss
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

        decay = self.config['gate_ema_decay']
        uninitialised = self.noised_loss_ema < 0

        cur_noised_loss = noised_info['noised_critic_loss']
        cur_var_q = noised_info['var_q']
        new_noised_ema = jnp.where(uninitialised, cur_noised_loss,
                                   decay * self.noised_loss_ema + (1 - decay) * cur_noised_loss)
        new_var_q_ema = jnp.where(uninitialised, cur_var_q,
                                  decay * self.var_q_ema + (1 - decay) * cur_var_q)

        cur_critic_loss = info['critic/critic_loss']
        cur_var_tq = info['critic/var_target_q']
        new_critic_ema = jnp.where(uninitialised, cur_critic_loss,
                                   decay * self.critic_loss_ema + (1 - decay) * cur_critic_loss)
        new_var_tq_ema = jnp.where(uninitialised, cur_var_tq,
                                   decay * self.var_tq_ema + (1 - decay) * cur_var_tq)

        info['gate/noised_loss_ema'] = new_noised_ema
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
    # Inference (identical to FQL / NFQL / NFQL2 / NFQL3).
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
            encoders['value'] = encoder_module()
            encoders['actor_bc_flow'] = encoder_module()
            encoders['actor_onestep_flow'] = encoder_module()
            encoders['noised_critic'] = encoder_module()

        critic_def = Value(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            num_ensembles=2,
            encoder=encoders.get('critic'),
        )
        value_def = Value(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            num_ensembles=1,
            encoder=encoders.get('value'),
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
            value=(value_def, (ex_observations,)),
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
            agent_name='nfql_5',
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
            # ---- Noised critic ----
            n_noised_actions=4,
            # ---- V head and V-anchor on Q_n at t=0 ----
            n_v_samples=8,            # flow-policy samples for V training target (MC E_a Q)
            lambda_v_anchor=1.0,      # weight on v_anchor_loss inside noised_critic_loss
            value_loss_weight=1.0,    # weight on value_loss inside total_loss
            # ---- ESS-targeted weighting ----
            ess_target=0.7,
            # ---- Dual R²-based reliability gate ----
            r2_target=0.75,
            gate_kappa=0.05,
            r2_critic_target=0.5,
            gate_kappa_critic=0.05,
            gate_ema_decay=0.999,
        )
    )
    return config
