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


class FQLARAgent(flax.struct.PyTreeNode):
    """FQL with Flow-Anchored Reweighted BC (FlowAR).

    The BC flow loss is reweighted by an ESS-targeted softmax of a flow-local
    advantage:

        a' = ODE(actor_bc_flow ; from (x_t = (1-t) eps + t a, t) to t = 1)
        Delta(s, a; t) = Q_target(s, a) - Q_target(s, a')

    Unlike fql_v which uses a global V(s) = E_{a' ~ pi}[Q] baseline, FlowAR's
    baseline is sample-anchored: a' is the BC flow's local completion of a
    after partial noising. The noise level t is a continuous knob between
    full self-consistency (t -> 0, a' = a) and the BC marginal (t -> 1,
    a' ~ BC(s)). The advantage signal vanishes when the BC flow already
    reproduces a (offline-safe), and turns systematically positive when the
    buffer contains best-of-n / improved actions that the BC flow would pull
    back toward its mode (online-aware).

    No V network and no Q_n network. Uses the existing actor_bc_flow itself
    as the baseline generator.
    """

    rng: Any
    network: Any
    critic_loss_ema: jnp.ndarray
    var_tq_ema: jnp.ndarray
    config: Any = nonpytree_field()

    def critic_loss(self, batch, grad_params, rng):
        """Standard FQL critic loss; also returns var(target_q) for the R^2 gate."""
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

        return critic_loss, {
            'critic_loss': critic_loss,
            'var_target_q': jnp.var(target_q) + 1e-8,
            'q_mean': q.mean(),
            'q_max': q.max(),
            'q_min': q.min(),
        }

    @staticmethod
    def _ess_targeted_weights(a_norm, ess_target, n_iters=14):
        """Bisect log(tau) so that ESS over the batch hits ess_target."""
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

    def _roll_flow_from(self, observations, x_t, t_start, n_steps):
        """Integrate actor_bc_flow from (x_t, t_start) to t = 1.

        Args:
            observations: (B, *obs_dims).
            x_t: (B, action_dim) flow state at t_start.
            t_start: (B, 1) starting flow time in [0, 1).
            n_steps: int Euler steps for the remaining (1 - t_start) interval.

        Returns:
            (B, action_dim) terminal flow state, clipped to [-1, 1].
        """
        if self.config['encoder'] is not None:
            obs_enc = self.network.select('actor_bc_flow_encoder')(observations)
        else:
            obs_enc = observations
        x = x_t
        # Per-sample uniform step size (1 - t_start) / n_steps.
        dt = (1.0 - t_start) / float(n_steps)
        for i in range(n_steps):
            tau = t_start + i * dt
            vels = self.network.select('actor_bc_flow')(obs_enc, x, tau, is_encoded=True)
            x = x + vels * dt
        return jnp.clip(x, -1, 1)

    def _sample_adv_t(self, rng, shape):
        """Sample FlowAR advantage-path t according to the configured schedule."""
        lo = self.config['adv_t_lo']
        hi = self.config['adv_t_hi']
        dist = self.config.get('adv_t_dist', 'uniform')
        if dist == 'uniform':
            t_unit = jax.random.uniform(rng, shape)
        elif dist == 'triangular':
            u = jax.random.uniform(rng, shape)
            t_unit = jnp.where(u < 0.5, jnp.sqrt(u / 2.0), 1.0 - jnp.sqrt((1.0 - u) / 2.0))
        elif dist == 'beta22':
            t_unit = jax.random.beta(rng, 2.0, 2.0, shape=shape)
        else:
            raise ValueError(f"Unknown adv_t_dist: {dist}")
        return lo + (hi - lo) * t_unit

    def actor_loss(self, batch, grad_params, rng, online):
        """FlowAR actor loss = Delta-weighted BC flow + distill + Q."""
        batch_size, action_dim = batch['actions'].shape
        a = batch['actions']

        # ----- Branch A: advantage via self-denoise -------------------------
        rng, eps_rng, t_rng = jax.random.split(rng, 3)
        eps_adv = jax.random.normal(eps_rng, (batch_size, action_dim))
        t_adv = self._sample_adv_t(t_rng, (batch_size, 1))
        x_t_adv = (1.0 - t_adv) * eps_adv + t_adv * a
        a_prime = self._roll_flow_from(
            batch['observations'], x_t_adv, t_adv,
            n_steps=self.config['adv_flow_steps'],
        )
        a_prime = jax.lax.stop_gradient(a_prime)

        # Use target_critic for advantage (stable baseline).
        q_a_all = self.network.select('target_critic')(
            batch['observations'], actions=a
        )
        q_aprime_all = self.network.select('target_critic')(
            batch['observations'], actions=a_prime
        )
        if self.config['q_agg'] == 'min':
            q_a = q_a_all.min(axis=0)
            q_aprime = q_aprime_all.min(axis=0)
        else:
            q_a = q_a_all.mean(axis=0)
            q_aprime = q_aprime_all.mean(axis=0)

        delta = jax.lax.stop_gradient(q_a - q_aprime)

        # MAD-normalize across batch.
        d_med = jnp.median(delta)
        beta_mad = 1.4826 * jnp.median(jnp.abs(delta - d_med))
        beta_mad = jnp.maximum(beta_mad, 1e-6)
        d_norm = (delta - d_med) / beta_mad

        # ESS-targeted softmax.
        w_exp, tau_star, ess_achieved = self._ess_targeted_weights(
            d_norm, jnp.asarray(self.config['ess_target'])
        )

        # Critic R^2 reliability gate.
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

        apply_weights = (not self.config['weighted_bc_online_only']) or online
        weighting_active = jnp.asarray(1.0 if apply_weights else 0.0, dtype=jnp.float32)
        gate_c = gate_critic * weighting_active
        bc_weights = 1.0 + gate_c * (w_exp - 1.0)
        bc_weights = jax.lax.stop_gradient(bc_weights)

        # ----- Branch B: independently-sampled CFM BC loss, reweighted -----
        rng, eps2_rng, t2_rng = jax.random.split(rng, 3)
        k = self.config['n_actor_time_samples']
        eps2 = jax.random.normal(eps2_rng, (batch_size, k, action_dim))
        t2 = jax.random.uniform(t2_rng, (batch_size, k, 1))
        a_exp = a[:, None, :]
        x_t2 = (1.0 - t2) * eps2 + t2 * a_exp
        target_vel = a_exp - eps2

        obs_tail = batch['observations'].shape[1:]
        obs_exp = jnp.broadcast_to(
            batch['observations'][:, None], (batch_size, k) + obs_tail
        )
        obs_flat = obs_exp.reshape((batch_size * k,) + obs_tail)
        x_t2_flat = x_t2.reshape(batch_size * k, action_dim)
        t2_flat = t2.reshape(batch_size * k, 1)
        pred_flat = self.network.select('actor_bc_flow')(
            obs_flat, x_t2_flat, t2_flat, params=grad_params
        )
        pred = pred_flat.reshape(batch_size, k, action_dim)
        per_time_bc = jnp.mean((pred - target_vel) ** 2, axis=-1)
        bc_flow_loss = jnp.mean(bc_weights[:, None] * per_time_bc)

        if self.config['use_distill_head']:
            # ----- Distillation + Q loss (unchanged from FQL) --------------
            rng, noise_rng = jax.random.split(rng)
            noises = jax.random.normal(noise_rng, (batch_size, action_dim))
            target_flow_actions = self.compute_flow_actions(
                batch['observations'], noises=noises
            )
            actor_actions = self.network.select('actor_onestep_flow')(
                batch['observations'], noises, params=grad_params
            )
            distill_loss = jnp.mean((actor_actions - target_flow_actions) ** 2)

            actor_actions_clip = jnp.clip(actor_actions, -1, 1)
            qs = self.network.select('critic')(batch['observations'], actions=actor_actions_clip)
            q = jnp.mean(qs, axis=0)

            q_loss = -q.mean()
            if self.config['normalize_q_loss']:
                lam = jax.lax.stop_gradient(1 / jnp.abs(q).mean())
                q_loss = lam * q_loss
        else:
            distill_loss = jnp.asarray(0.0, dtype=jnp.float32)
            q_loss = jnp.asarray(0.0, dtype=jnp.float32)
            q = jnp.asarray(0.0, dtype=jnp.float32)

        actor_loss = bc_flow_loss + self.config['alpha'] * distill_loss + q_loss

        # ----- Diagnostics --------------------------------------------------
        if self.config['use_distill_head']:
            actions = self.sample_actions(batch['observations'], seed=rng)
            mse = jnp.mean((actions - batch['actions']) ** 2)
        else:
            mse = jnp.asarray(0.0, dtype=jnp.float32)
        dist_a_aprime = jnp.linalg.norm(a - a_prime, axis=-1)
        frac_a_better = jnp.mean((delta > 0).astype(jnp.float32))

        return actor_loss, {
            'actor_loss': actor_loss,
            'bc_flow_loss': bc_flow_loss,
            'distill_loss': distill_loss,
            'q_loss': q_loss,
            'q': q.mean(),
            'mse': mse,
            'flowar/delta_mean': delta.mean(),
            'flowar/delta_std': delta.std(),
            'flowar/dist_a_aprime_mean': dist_a_aprime.mean(),
            'flowar/dist_a_aprime_p50': jnp.median(dist_a_aprime),
            'flowar/frac_a_better': frac_a_better,
            'flowar/q_a_mean': q_a.mean(),
            'flowar/q_aprime_mean': q_aprime.mean(),
            'flowar/t_adv_mean': t_adv.mean(),
            'beta_mad': beta_mad,
            'tau_star': tau_star,
            'ess_achieved': ess_achieved,
            'gate_critic': gate_critic,
            'gate_c': gate_c,
            'online_phase': jnp.asarray(1.0 if online else 0.0, dtype=jnp.float32),
            'weighting_active': weighting_active,
            'r2_critic': r2_critic,
            'bc_weight_mean': bc_weights.mean(),
            'bc_weight_std': bc_weights.std(),
            'bc_weight_max': bc_weights.max(),
            'bc_weight_min': bc_weights.min(),
        }

    @functools.partial(jax.jit, static_argnames=('online',))
    def total_loss(self, batch, grad_params, rng=None, online=False):
        info = {}
        rng = rng if rng is not None else self.rng

        rng, actor_rng, critic_rng = jax.random.split(rng, 3)

        critic_loss, critic_info = self.critic_loss(batch, grad_params, critic_rng)
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v

        actor_loss, actor_info = self.actor_loss(
            batch, grad_params, actor_rng, online=online
        )
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

    @functools.partial(jax.jit, static_argnames=('online',))
    def update(self, batch, online=False):
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng, online=online)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        self.target_update(new_network, 'critic')

        decay = self.config['gate_ema_decay']
        uninitialised = self.critic_loss_ema < 0

        cur_critic_loss = info['critic/critic_loss']
        cur_var_tq = info['critic/var_target_q']
        new_critic_ema = jnp.where(
            uninitialised, cur_critic_loss,
            decay * self.critic_loss_ema + (1 - decay) * cur_critic_loss,
        )
        new_var_tq_ema = jnp.where(
            uninitialised, cur_var_tq,
            decay * self.var_tq_ema + (1 - decay) * cur_var_tq,
        )

        info['gate/critic_loss_ema'] = new_critic_ema
        info['gate/var_tq_ema'] = new_var_tq_ema
        info['gate/r2_critic_ema'] = 1.0 - new_critic_ema / jnp.maximum(new_var_tq_ema, 1e-8)

        return self.replace(
            network=new_network,
            critic_loss_ema=new_critic_ema,
            var_tq_ema=new_var_tq_ema,
            rng=new_rng,
        ), info

    @jax.jit
    def sample_actions(self, observations, seed=None, temperature=1.0):
        action_seed, _ = jax.random.split(seed)
        noises = jax.random.normal(
            action_seed,
            (
                *observations.shape[: -len(self.config['ob_dims'])],
                self.config['action_dim'],
            ),
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

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ex_times = ex_actions[..., :1]
        ob_dims = ex_observations.shape[1:]
        action_dim = ex_actions.shape[-1]

        encoders = dict()
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            encoders['critic'] = encoder_module()
            encoders['actor_bc_flow'] = encoder_module()
            encoders['actor_onestep_flow'] = encoder_module()

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
            network_info['actor_bc_flow_encoder'] = (
                encoders.get('actor_bc_flow'), (ex_observations,)
            )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network.params
        params['modules_target_critic'] = params['modules_critic']

        config['ob_dims'] = ob_dims
        config['action_dim'] = action_dim
        sentinel = jnp.asarray(-1.0, dtype=jnp.float32)
        return cls(
            rng,
            network=network,
            critic_loss_ema=sentinel,
            var_tq_ema=sentinel,
            config=flax.core.FrozenDict(**config),
        )


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name='fql_ar',
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
            use_distill_head=True,
            encoder=ml_collections.config_dict.placeholder(str),
            # ---- BC flow CFM samples per state (independent of advantage path) ----
            n_actor_time_samples=1,
            # ---- FlowAR advantage path ----
            adv_t_lo=0.4,           # lower bound on noise level for advantage path
            adv_t_hi=0.7,           # upper bound on noise level for advantage path
            adv_t_dist='uniform',   # uniform, triangular, or beta22
            adv_flow_steps=3,       # Euler steps for partial denoising back to t=1
            # ---- ESS-targeted reweighting ----
            ess_target=0.7,
            weighted_bc_online_only=True,
            # ---- Critic R^2 reliability gate ----
            r2_critic_target=0.5,
            gate_kappa_critic=0.05,
            gate_ema_decay=0.999,
        )
    )
    return config
