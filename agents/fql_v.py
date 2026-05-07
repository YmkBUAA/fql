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


class FQLVAgent(flax.struct.PyTreeNode):
    """FQL with value-advantage weighted BC.

    The BC flow loss is reweighted from the standard critic/value advantage
    A(s, a) = Q(s, a) - V(s). There is no noised critic and no Q_n training.
    """

    rng: Any
    network: Any
    critic_loss_ema: jnp.ndarray
    var_tq_ema: jnp.ndarray
    config: Any = nonpytree_field()

    def critic_loss(self, batch, grad_params, rng):
        """Compute the FQL critic loss."""
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

    def value_loss(self, batch, grad_params, rng):
        """Train V(s) toward E_a[Q(s, a)] under the configured actor sampler."""
        batch_size, action_dim = batch['actions'].shape
        n = self.config['n_v_samples']

        noises = jax.random.normal(rng, (batch_size, n, action_dim))
        obs_tail = batch['observations'].shape[1:]
        obs_exp = jnp.broadcast_to(
            batch['observations'][:, None], (batch_size, n) + obs_tail
        )
        obs_flat = obs_exp.reshape((batch_size * n,) + obs_tail)
        noises_flat = noises.reshape(batch_size * n, action_dim)

        if self.config['v_baseline_source'] == 'bc_flow':
            a_samples_flat = self.compute_flow_actions(obs_flat, noises_flat)
        else:
            a_samples_flat = self.network.select('actor_onestep_flow')(obs_flat, noises_flat)
        a_samples_flat = jnp.clip(a_samples_flat, -1, 1)
        a_samples_flat = jax.lax.stop_gradient(a_samples_flat)

        q_samples_flat = self.network.select('target_critic')(
            obs_flat, actions=a_samples_flat
        )
        ensemble = q_samples_flat.shape[0]
        q_samples = q_samples_flat.reshape(ensemble, batch_size, n)
        v_target = jax.lax.stop_gradient(q_samples.mean(axis=(0, 2)))

        v_pred = self.network.select('value')(
            batch['observations'], params=grad_params
        )

        loss = jnp.mean((v_pred - v_target) ** 2)
        return loss, {
            'value_loss': loss,
            'v_pred_mean': v_pred.mean(),
            'v_target_mean': v_target.mean(),
            'v_target_std': v_target.std(),
            'v_baseline_source_bc_flow': jnp.asarray(
                1.0 if self.config['v_baseline_source'] == 'bc_flow' else 0.0,
                dtype=jnp.float32,
            ),
        }

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

    def actor_loss(self, batch, grad_params, rng, online):
        """Compute the FQL actor loss."""
        batch_size, action_dim = batch['actions'].shape
        rng, x_rng, t_rng = jax.random.split(rng, 3)
        k = self.config['n_actor_time_samples']

        # BC flow loss.
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
        v = self.network.select('value')(batch['observations'])
        v = jax.lax.stop_gradient(v)

        a_local = original_q - v
        a_med = jnp.median(a_local)
        beta_mad = 1.4826 * jnp.median(jnp.abs(a_local - a_med))
        beta_mad = jnp.maximum(beta_mad, 1e-6)
        a_norm = (a_local - a_med) / beta_mad

        w_exp, tau_star, ess_achieved = self._ess_targeted_weights(
            a_norm, jnp.asarray(self.config['ess_target'])
        )

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

        obs_tail = batch['observations'].shape[1:]
        obs_exp = jnp.broadcast_to(
            batch['observations'][:, None], (batch_size, k) + obs_tail
        )
        obs_flat = obs_exp.reshape((batch_size * k,) + obs_tail)
        x_t_flat = x_t.reshape(batch_size * k, action_dim)
        t_flat = t.reshape(batch_size * k, 1)
        pred_flat = self.network.select('actor_bc_flow')(
            obs_flat, x_t_flat, t_flat, params=grad_params
        )
        pred = pred_flat.reshape(batch_size, k, action_dim)
        per_time_bc = jnp.mean((pred - vel) ** 2, axis=-1)
        bc_flow_loss = jnp.mean(bc_weights[:, None] * per_time_bc)

        if self.config['use_distill_head']:
            # Distillation loss.
            rng, noise_rng = jax.random.split(rng)
            noises = jax.random.normal(noise_rng, (batch_size, action_dim))
            target_flow_actions = self.compute_flow_actions(batch['observations'], noises=noises)
            actor_actions = self.network.select('actor_onestep_flow')(batch['observations'], noises, params=grad_params)
            distill_loss = jnp.mean((actor_actions - target_flow_actions) ** 2)

            # Q loss.
            actor_actions = jnp.clip(actor_actions, -1, 1)
            qs = self.network.select('critic')(batch['observations'], actions=actor_actions)
            q = jnp.mean(qs, axis=0)

            q_loss = -q.mean()
            if self.config['normalize_q_loss']:
                lam = jax.lax.stop_gradient(1 / jnp.abs(q).mean())
                q_loss = lam * q_loss
        else:
            distill_loss = jnp.asarray(0.0, dtype=jnp.float32)
            q_loss = jnp.asarray(0.0, dtype=jnp.float32)
            q = jnp.asarray(0.0, dtype=jnp.float32)

        # Total loss.
        actor_loss = bc_flow_loss + self.config['alpha'] * distill_loss + q_loss

        # Additional metrics for logging.
        if self.config['use_distill_head']:
            actions = self.sample_actions(batch['observations'], seed=rng)
            mse = jnp.mean((actions - batch['actions']) ** 2)
        else:
            mse = jnp.asarray(0.0, dtype=jnp.float32)

        return actor_loss, {
            'actor_loss': actor_loss,
            'bc_flow_loss': bc_flow_loss,
            'distill_loss': distill_loss,
            'q_loss': q_loss,
            'q': q.mean(),
            'mse': mse,
            'v': v.mean(),
            'adv_mean': a_local.mean(),
            'adv_std': a_local.std(),
            'beta_mad': beta_mad,
            'tau_star': tau_star,
            'ess_achieved': ess_achieved,
            'gate_critic': gate_critic,
            'gate_c': gate_c,
            'online_phase': jnp.asarray(1.0 if online else 0.0, dtype=jnp.float32),
            'weighting_active': weighting_active,
            'r2_critic': r2_critic,
            'bc_weight_mean': bc_weights.mean(),
            'bc_weight_max': bc_weights.max(),
            'bc_weight_min': bc_weights.min(),
            'n_actor_time_samples': jnp.asarray(k, dtype=jnp.float32),
            't_mean': t.mean(),
        }

    @functools.partial(jax.jit, static_argnames=('online',))
    def total_loss(self, batch, grad_params, rng=None, online=False):
        """Compute the total loss."""
        info = {}
        rng = rng if rng is not None else self.rng

        rng, actor_rng, critic_rng, value_rng = jax.random.split(rng, 4)

        critic_loss, critic_info = self.critic_loss(batch, grad_params, critic_rng)
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v

        actor_loss, actor_info = self.actor_loss(
            batch, grad_params, actor_rng, online=online
        )
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        v_loss, v_info = self.value_loss(batch, grad_params, value_rng)
        for k, v in v_info.items():
            info[f'value/{k}'] = v

        loss = critic_loss + actor_loss + self.config['value_loss_weight'] * v_loss
        return loss, info

    def target_update(self, network, module_name):
        """Update the target network."""
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
            self.network.params[f'modules_{module_name}'],
            self.network.params[f'modules_target_{module_name}'],
        )
        network.params[f'modules_target_{module_name}'] = new_target_params

    @functools.partial(jax.jit, static_argnames=('online',))
    def update(self, batch, online=False):
        """Update the agent and return a new agent with information dictionary."""
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
    def sample_actions(
        self,
        observations,
        seed=None,
        temperature=1.0,
    ):
        """Sample actions from the one-step policy."""
        action_seed, noise_seed = jax.random.split(seed)
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
    def compute_flow_actions(
        self,
        observations,
        noises,
    ):
        """Compute actions from the BC flow model using the Euler method."""
        if self.config['encoder'] is not None:
            observations = self.network.select('actor_bc_flow_encoder')(observations)
        actions = noises
        # Euler method.
        for i in range(self.config['flow_steps']):
            t = jnp.full((*observations.shape[:-1], 1), i / self.config['flow_steps'])
            vels = self.network.select('actor_bc_flow')(observations, actions, t, is_encoded=True)
            actions = actions + vels / self.config['flow_steps']
        actions = jnp.clip(actions, -1, 1)
        return actions

    @classmethod
    def create(
        cls,
        seed,
        ex_observations,
        ex_actions,
        config,
    ):
        """Create a new agent.

        Args:
            seed: Random seed.
            ex_observations: Example batch of observations.
            ex_actions: Example batch of actions.
            config: Configuration dictionary.
        """
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ex_times = ex_actions[..., :1]
        ob_dims = ex_observations.shape[1:]
        action_dim = ex_actions.shape[-1]

        # Define encoders.
        encoders = dict()
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            encoders['critic'] = encoder_module()
            encoders['value'] = encoder_module()
            encoders['actor_bc_flow'] = encoder_module()
            encoders['actor_onestep_flow'] = encoder_module()

        # Define networks.
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
            # Add actor_bc_flow_encoder to ModuleDict to make it separately callable.
            network_info['actor_bc_flow_encoder'] = (encoders.get('actor_bc_flow'), (ex_observations,))
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
            agent_name='fql_v',  # Agent name.
            ob_dims=ml_collections.config_dict.placeholder(list),  # Observation dimensions (will be set automatically).
            action_dim=ml_collections.config_dict.placeholder(int),  # Action dimension (will be set automatically).
            lr=3e-4,  # Learning rate.
            batch_size=256,  # Batch size.
            actor_hidden_dims=(512, 512, 512, 512),  # Actor network hidden dimensions.
            value_hidden_dims=(512, 512, 512, 512),  # Value network hidden dimensions.
            layer_norm=True,  # Whether to use layer normalization.
            actor_layer_norm=False,  # Whether to use layer normalization for the actor.
            discount=0.99,  # Discount factor.
            tau=0.005,  # Target network update rate.
            q_agg='mean',  # Aggregation method for target Q values.
            alpha=10.0,  # BC coefficient (need to be tuned for each environment).
            flow_steps=10,  # Number of flow steps.
            normalize_q_loss=False,  # Whether to normalize the Q loss.
            use_distill_head=True,  # Whether to train/use the distilled one-step head in the actor loss.
            encoder=ml_collections.config_dict.placeholder(str),  # Visual encoder name (None, 'impala_small', etc.).
            n_actor_time_samples=1,
            n_v_samples=1,
            v_baseline_source='onestep',  # 'onestep' (legacy) or 'bc_flow'.
            value_loss_weight=1.0,
            ess_target=0.7,
            weighted_bc_online_only=True,  # If True, value weighting is active only during the online stage.
            r2_critic_target=0.5,
            gate_kappa_critic=0.05,
            gate_ema_decay=0.999,
        )
    )
    return config
