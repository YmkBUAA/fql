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


class NFQLAgent(flax.struct.PyTreeNode):
    """Noised Flow Q-learning (NFQL) agent.

    Extends FQL with a noised-action value network that learns to predict Q-values
    for noise-perturbed actions by matching the original critic's output.
    """

    rng: Any
    network: Any
    noised_network: Any  # Separate TrainState for the noised critic.
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
            'q_mean': q.mean(),
            'q_max': q.max(),
            'q_min': q.min(),
        }

    def actor_loss(self, batch, grad_params, rng, online):
        """Compute the FQL actor loss with noised-critic-based BC weighting.

        Per-sample BC weights are derived by comparing Q(s, a) against the noised
        critic's local neighborhood mean Q_n(s, a+ε, t).  If Q(s, a) < Q_n (the
        dataset action is below its local average → suboptimal), the BC loss for
        that transition is down-weighted; otherwise it is up-weighted.

        Whether weighting is active depends on `weighted_bc_online_only`:
          False (default) — weights applied in both offline and online stages.
          True            — weights applied only during the online stage.
        The `online` argument (static at JIT time) signals the current stage.
        """
        batch_size, action_dim = batch['actions'].shape

        # Keep the same 3-way split as the original FQL so that x_rng, t_rng,
        # and the downstream distillation noise_rng are identical to FQL's when
        # bc_weights are not active.  The extra keys for the noised-critic
        # comparison are derived from a separate fold of `rng` so they cannot
        # contaminate the main-loss RNG stream.
        rng, x_rng, t_rng = jax.random.split(rng, 3)
        weight_rng = jax.random.fold_in(rng, 0)   # isolated sub-stream for weighting
        noise_rng, t_noised_rng = jax.random.split(weight_rng)

        # ------------------------------------------------------------------
        # Compute per-sample BC weights from the noised critic comparison.
        # Both Q values are stop_gradient'd — they only steer the weights,
        # not the critic or noised-critic gradients.
        #
        # `online` and `weighted_bc_online_only` are both Python bools at
        # JIT trace time, so `apply_weights` is a compile-time constant and
        # the inactive branch is eliminated by XLA.
        # ------------------------------------------------------------------
        apply_weights = (not self.config['weighted_bc_online_only']) or online

        original_qs = self.network.select('critic')(batch['observations'], actions=batch['actions'])
        original_q = jax.lax.stop_gradient(original_qs.mean(axis=0))  # (batch_size,)

        # One noised sample per state is enough for the rough weight signal.
        noise = jax.random.normal(noise_rng, (batch_size, action_dim)) * self.config['noise_scale']
        noised_actions = jnp.clip(batch['actions'] + noise, -1, 1)
        t_noised = jax.random.uniform(t_noised_rng, (batch_size, 1))
        noised_actions_with_t = jnp.concatenate([noised_actions, t_noised], axis=-1)
        noised_qs = self.noised_network.select('noised_critic')(batch['observations'], actions=noised_actions_with_t)
        noised_q = jax.lax.stop_gradient(noised_qs.mean(axis=0))  # (batch_size,)

        # Up-weight locally optimal actions; down-weight suboptimal ones.
        # When `apply_weights` is False (offline stage + online_only mode),
        # all weights are 1.0 — equivalent to the unweighted FQL BC loss.
        bc_weights = jnp.where(
            apply_weights,
            jnp.where(
                original_q >= noised_q,
                self.config['bc_weight_high'],   # action is at or above neighborhood mean
                self.config['bc_weight_low'],    # action is below neighborhood mean
            ),
            jnp.ones(batch_size),
        )  # (batch_size,)

        # ------------------------------------------------------------------
        # BC flow loss (weighted per sample).
        # ------------------------------------------------------------------
        x_0 = jax.random.normal(x_rng, (batch_size, action_dim))
        x_1 = batch['actions']
        t = jax.random.uniform(t_rng, (batch_size, 1))
        x_t = (1 - t) * x_0 + t * x_1
        vel = x_1 - x_0

        pred = self.network.select('actor_bc_flow')(batch['observations'], x_t, t, params=grad_params)
        per_sample_bc_loss = jnp.mean((pred - vel) ** 2, axis=-1)  # (batch_size,)
        bc_flow_loss = jnp.mean(bc_weights * per_sample_bc_loss)

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

        # Total loss.
        actor_loss = bc_flow_loss + self.config['alpha'] * distill_loss + q_loss

        # Additional metrics for logging.
        actions = self.sample_actions(batch['observations'], seed=rng)
        mse = jnp.mean((actions - batch['actions']) ** 2)
        frac_suboptimal = jnp.mean(original_q < noised_q)

        return actor_loss, {
            'actor_loss': actor_loss,
            'bc_flow_loss': bc_flow_loss,
            'distill_loss': distill_loss,
            'q_loss': q_loss,
            'q': q.mean(),
            'frac_suboptimal': frac_suboptimal,  # fraction of transitions down-weighted
            'mse': mse,
        }

    def noised_critic_loss(self, batch, grad_params, rng):
        """Compute the noised critic loss.

        Generates n noised actions per state, evaluates them with the noised-action
        value network, and minimizes MSE against the original (frozen) Q-values.

        Args:
            batch: Training batch.
            grad_params: Parameters of the noised_network (gradients flow through these).
            rng: Random key.
        """
        batch_size, action_dim = batch['actions'].shape
        n = self.config['n_noised_actions']

        # Compute original Q-values with the frozen original critic (stop_gradient ensures
        # no gradients flow into network.params from this loss).
        original_qs = self.network.select('critic')(batch['observations'], actions=batch['actions'])
        original_q = jax.lax.stop_gradient(original_qs.mean(axis=0))  # (batch_size,)

        # Generate n noised actions by adding Gaussian noise to original actions.
        rng, noise_rng, t_rng = jax.random.split(rng, 3)
        noise = jax.random.normal(noise_rng, (batch_size, n, action_dim)) * self.config['noise_scale']
        noised_actions = batch['actions'][:, None, :] + noise  # (batch_size, n, action_dim)
        noised_actions = jnp.clip(noised_actions, -1, 1)

        # Sample a noise timestep t for each noised action.
        t = jax.random.uniform(t_rng, (batch_size, n, 1))

        # Concatenate noised actions with timestep t as additional input.
        noised_actions_with_t = jnp.concatenate([noised_actions, t], axis=-1)  # (batch_size, n, action_dim+1)

        # Expand observations to (batch_size, n, *ob_dims).
        obs_expanded = jnp.broadcast_to(
            batch['observations'][:, None],
            (batch_size, n) + batch['observations'].shape[1:],
        )

        # Evaluate the noised-action value network.
        # For visual environments the encoder inside Value expects a leading batch
        # dimension only (no extra n-axis), so we flatten (B, n, H, W, C) →
        # (B*n, H, W, C) before the call and restore the shape afterwards.
        if self.config['encoder'] is not None:
            obs_flat     = obs_expanded.reshape((batch_size * n,) + batch['observations'].shape[1:])
            actions_flat = noised_actions_with_t.reshape((batch_size * n, -1))
            noised_qs_flat = self.noised_network.select('noised_critic')(
                obs_flat, actions=actions_flat, params=grad_params
            )  # (num_ensembles, batch_size * n)
            noised_qs = noised_qs_flat.reshape((noised_qs_flat.shape[0], batch_size, n))
        else:
            noised_qs = self.noised_network.select('noised_critic')(
                obs_expanded, actions=noised_actions_with_t, params=grad_params
            )  # (num_ensembles, batch_size, n)
        noised_q = noised_qs.mean(axis=0)  # (batch_size, n)

        # MSE against the frozen original Q-values (broadcast over n noised samples).
        target_q = original_q[:, None]  # (batch_size, 1)
        noised_critic_loss = jnp.mean((noised_q - target_q) ** 2)

        return noised_critic_loss, {
            'noised_critic_loss': noised_critic_loss,
            'noised_q_mean': noised_q.mean(),
            'noised_q_std': noised_q.std(),
        }

    @functools.partial(jax.jit, static_argnames=('online',))
    def total_loss(self, batch, grad_params, rng=None, online=False):
        """Compute the total FQL loss (critic + actor).

        Args:
            online: Whether the agent is in the online fine-tuning stage.
                    Static at JIT time — triggers a retrace only once when
                    the stage switches from offline to online.
        """
        info = {}
        rng = rng if rng is not None else self.rng

        rng, actor_rng, critic_rng = jax.random.split(rng, 3)

        critic_loss, critic_info = self.critic_loss(batch, grad_params, critic_rng)
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v

        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng, online=online)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        loss = critic_loss + actor_loss
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
        """Update the agent and return a new agent with information dictionary.

        Args:
            online: Whether the agent is in the online fine-tuning stage.
                    Controls BC weighting when `weighted_bc_online_only=True`.
                    Static at JIT time — triggers a retrace only once when
                    the stage switches.

        Performs two separate gradient updates:
        1. Standard FQL update for the critic and actor networks.
        2. Noised critic update — the original critic is structurally frozen because
           it lives in a separate TrainState with its own optimizer, and its outputs
           are additionally stop_gradient'd inside noised_critic_loss.
        """
        new_rng, rng = jax.random.split(self.rng)
        rng, fql_rng, noised_rng = jax.random.split(rng, 3)

        # Standard FQL update (critic + actor).
        def fql_loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=fql_rng, online=online)

        new_network, info = self.network.apply_loss_fn(loss_fn=fql_loss_fn)
        self.target_update(new_network, 'critic')

        # Noised critic update — only noised_network.params receive gradients.
        def noised_loss_fn(grad_params):
            return self.noised_critic_loss(batch, grad_params, rng=noised_rng)

        new_noised_network, noised_info = self.noised_network.apply_loss_fn(loss_fn=noised_loss_fn)
        for k, v in noised_info.items():
            info[f'noised_critic/{k}'] = v

        return self.replace(network=new_network, noised_network=new_noised_network, rng=new_rng), info

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
        rng, init_rng, noised_init_rng = jax.random.split(rng, 3)

        ex_times = ex_actions[..., :1]
        ob_dims = ex_observations.shape[1:]
        action_dim = ex_actions.shape[-1]

        # Define encoders.
        encoders = dict()
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            encoders['critic'] = encoder_module()
            encoders['actor_bc_flow'] = encoder_module()
            encoders['actor_onestep_flow'] = encoder_module()
            encoders['noised_critic'] = encoder_module()

        # Define FQL networks.
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

        # Define the noised-action value network.
        # Input: (observations, concat(noised_action, t)) where t is the noise timestep.
        # Architecture is identical to the original critic (same hidden dims, layer norm,
        # num_ensembles=2); the only difference is the action input includes t as an extra dim.
        # For visual environments the same encoder type as the critic is used so that pixel
        # observations are encoded before concatenation with the action.
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
        return cls(rng, network=network, noised_network=noised_network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name='nfql',  # Agent name.
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
            encoder=ml_collections.config_dict.placeholder(str),  # Visual encoder name (None, 'impala_small', etc.).
            n_noised_actions=4,  # Number of noised actions to sample per state.
            noise_scale=0.1,  # Standard deviation of Gaussian noise added to actions.
            bc_weight_high=1.2,  # BC loss weight when Q(s,a) >= local neighborhood mean (optimal).
            bc_weight_low=0.9,   # BC loss weight when Q(s,a) <  local neighborhood mean (suboptimal).
            weighted_bc_online_only=False,  # If True, weighting is active only during the online stage.
        )
    )
    return config
