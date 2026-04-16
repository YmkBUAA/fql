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


class NFQL2Agent(flax.struct.PyTreeNode):
    """Noised Flow Q-learning v2 (NFQL2).

    Improvements over NFQL:
      1. Robust scale via MAD instead of std.
      2. ESS-targeted exponential temperature (bounded weight concentration).
      3. Per-sample trust from noised-critic ensemble disagreement.
      4. Data-driven reliability gate from the noised-critic loss EMA
         (replaces the step-based curriculum from recommended_solution.md).

    See `docs/improved_solution.md` for derivations and proofs.
    """

    rng: Any
    network: Any
    noised_network: Any  # Separate TrainState for the noised critic.
    # Scalar EMA state for the R²-based reliability gate.
    # Both EMAs use -1.0 as an "uninitialised" sentinel; they are populated
    # from the first update step.
    noised_loss_ema: jnp.ndarray
    var_q_ema: jnp.ndarray
    config: Any = nonpytree_field()

    # ------------------------------------------------------------------
    # Critic loss (identical to FQL / NFQL).
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

        return critic_loss, {
            'critic_loss': critic_loss,
            'q_mean': q.mean(),
            'q_max': q.max(),
            'q_min': q.min(),
        }

    # ------------------------------------------------------------------
    # Helper: ESS-targeted temperature via fixed-iteration bisection.
    # ------------------------------------------------------------------
    @staticmethod
    def _ess_targeted_weights(a_norm, ess_target, n_iters=14):
        """Return per-sample weights w_i = exp(a_norm_i / tau*) where tau* is
        chosen so that ESS / B = ess_target.

        Args:
            a_norm:     (B,) robust z-scored advantage (mean ~ 0, MAD ~ 1).
            ess_target: scalar in (1/B, 1).  ESS_target = 0.5 by default.
            n_iters:    number of bisection iterations (14 → tau resolved
                        to ~6e-5 over [1e-3, 1e3]).

        Returns:
            w_norm: (B,) weights normalised so that mean(w_norm) = 1.
            tau:    scalar temperature found.
            ess:    achieved ESS / B (for logging).
        """
        b = a_norm.shape[0]

        def ess_over_b(log_tau):
            tau = jnp.exp(log_tau)
            # Numerically stable softmax-like normalisation.
            z = a_norm / tau
            z = z - jnp.max(z)
            w = jnp.exp(z)
            s1 = jnp.sum(w)
            s2 = jnp.sum(w * w)
            return (s1 * s1) / (b * s2 + 1e-12)

        # Bisection in log-tau space over [log(1e-3), log(1e3)].
        log_lo = jnp.log(1e-3)
        log_hi = jnp.log(1e3)

        def body(_, state):
            lo, hi = state
            mid = 0.5 * (lo + hi)
            ess_mid = ess_over_b(mid)
            # ess(tau) is increasing in tau; if ess_mid < target, raise lo.
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
    # Actor loss with ESS-targeted robust weighting and data-driven gate.
    # ------------------------------------------------------------------
    def actor_loss(self, batch, grad_params, rng):
        batch_size, action_dim = batch['actions'].shape

        # RNG split mirrors NFQL so that x_rng/t_rng/distill noise are
        # identical to FQL when the gate is closed.
        rng, x_rng, t_rng = jax.random.split(rng, 3)
        weight_rng = jax.random.fold_in(rng, 0)
        noise_rng, t_noised_rng = jax.random.split(weight_rng)

        # ---- Original critic Q(s, a) ------------------------------------
        original_qs = self.network.select('critic')(batch['observations'], actions=batch['actions'])
        original_q = jax.lax.stop_gradient(original_qs.mean(axis=0))  # (B,)

        # ---- Noised critic ensemble Q_n(s, a+ε, t) ----------------------
        noise = jax.random.normal(noise_rng, (batch_size, action_dim)) * self.config['noise_scale']
        noised_actions = jnp.clip(batch['actions'] + noise, -1, 1)
        t_noised = jax.random.uniform(t_noised_rng, (batch_size, 1))
        noised_actions_with_t = jnp.concatenate([noised_actions, t_noised], axis=-1)
        noised_qs = self.noised_network.select('noised_critic')(
            batch['observations'], actions=noised_actions_with_t
        )  # (num_ensembles=2, B)
        noised_q = jax.lax.stop_gradient(noised_qs.mean(axis=0))            # (B,)
        noised_q_disagree = jax.lax.stop_gradient(
            jnp.abs(noised_qs[0] - noised_qs[1])
        )                                                                    # (B,)

        # ---- 1. Local advantage ----------------------------------------
        a_local = original_q - noised_q                                     # (B,)

        # ---- 2. Robust scale via MAD -----------------------------------
        a_med = jnp.median(a_local)
        beta_mad = 1.4826 * jnp.median(jnp.abs(a_local - a_med))
        beta_mad = jnp.maximum(beta_mad, 1e-6)                              # avoid /0
        a_norm = (a_local - a_med) / beta_mad                                # robust z-score

        # ---- 3. ESS-targeted exponential weights -----------------------
        w_exp_normed, tau_star, ess_achieved = self._ess_targeted_weights(
            a_norm, jnp.asarray(self.config['ess_target'])
        )                                                                    # (B,) mean ~ 1

        # ---- 4. Per-sample trust from ensemble disagreement ------------
        delta_med = jnp.maximum(jnp.median(noised_q_disagree), 1e-6)
        trust = jnp.exp(-noised_q_disagree / delta_med)                     # ∈ (0, 1]

        # ---- 5. Global reliability gate from R²(Q_n → Q) ----------------
        # R² = 1 − L_noised / Var[Q].  Scale-free: insensitive to Q's absolute
        # magnitude, which is the failure mode of the ratio L_noised/L_anchor.
        # Both EMAs use -1.0 as an uninitialised sentinel.
        ema_valid = (self.noised_loss_ema >= 0) & (self.var_q_ema > 0)
        r2 = jnp.where(
            ema_valid,
            1.0 - self.noised_loss_ema / jnp.maximum(self.var_q_ema, 1e-8),
            jnp.asarray(-1.0),
        )
        gate_arg = (r2 - self.config['r2_target']) / self.config['gate_kappa']
        c_gate = jax.nn.sigmoid(gate_arg)
        c_gate = jnp.where(ema_valid, c_gate, jnp.asarray(0.0))

        # ---- 6. Final per-sample BC weight -----------------------------
        bc_weights = 1.0 + c_gate * trust * (w_exp_normed - 1.0)            # (B,)
        bc_weights = jax.lax.stop_gradient(bc_weights)

        # ---- BC flow loss ----------------------------------------------
        x_0 = jax.random.normal(x_rng, (batch_size, action_dim))
        x_1 = batch['actions']
        t = jax.random.uniform(t_rng, (batch_size, 1))
        x_t = (1 - t) * x_0 + t * x_1
        vel = x_1 - x_0

        pred = self.network.select('actor_bc_flow')(batch['observations'], x_t, t, params=grad_params)
        per_sample_bc_loss = jnp.mean((pred - vel) ** 2, axis=-1)            # (B,)
        bc_flow_loss = jnp.mean(bc_weights * per_sample_bc_loss)

        # ---- Distillation loss -----------------------------------------
        rng, noise_rng = jax.random.split(rng)
        noises = jax.random.normal(noise_rng, (batch_size, action_dim))
        target_flow_actions = self.compute_flow_actions(batch['observations'], noises=noises)
        actor_actions = self.network.select('actor_onestep_flow')(batch['observations'], noises, params=grad_params)
        distill_loss = jnp.mean((actor_actions - target_flow_actions) ** 2)

        # ---- Q loss ----------------------------------------------------
        actor_actions = jnp.clip(actor_actions, -1, 1)
        qs = self.network.select('critic')(batch['observations'], actions=actor_actions)
        q = jnp.mean(qs, axis=0)

        q_loss = -q.mean()
        if self.config['normalize_q_loss']:
            lam = jax.lax.stop_gradient(1 / jnp.abs(q).mean())
            q_loss = lam * q_loss

        actor_loss = bc_flow_loss + self.config['alpha'] * distill_loss + q_loss

        # ---- Diagnostics ------------------------------------------------
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
            # Weighting diagnostics.
            'beta_mad': beta_mad,
            'tau_star': tau_star,
            'ess_achieved': ess_achieved,
            'gate_c': c_gate,
            'r2': r2,
            'trust_mean': trust.mean(),
            'bc_weight_mean': bc_weights.mean(),
            'bc_weight_max': bc_weights.max(),
            'bc_weight_min': bc_weights.min(),
        }

    # ------------------------------------------------------------------
    # Noised critic loss (identical to NFQL).
    # ------------------------------------------------------------------
    def noised_critic_loss(self, batch, grad_params, rng):
        batch_size, action_dim = batch['actions'].shape
        n = self.config['n_noised_actions']

        original_qs = self.network.select('critic')(batch['observations'], actions=batch['actions'])
        original_q = jax.lax.stop_gradient(original_qs.mean(axis=0))         # (B,)

        rng, noise_rng, t_rng = jax.random.split(rng, 3)
        noise = jax.random.normal(noise_rng, (batch_size, n, action_dim)) * self.config['noise_scale']
        noised_actions = batch['actions'][:, None, :] + noise                # (B, n, A)
        noised_actions = jnp.clip(noised_actions, -1, 1)

        t = jax.random.uniform(t_rng, (batch_size, n, 1))
        noised_actions_with_t = jnp.concatenate([noised_actions, t], axis=-1)

        obs_expanded = jnp.broadcast_to(
            batch['observations'][:, None],
            (batch_size, n) + batch['observations'].shape[1:],
        )

        if self.config['encoder'] is not None:
            obs_flat = obs_expanded.reshape((batch_size * n,) + batch['observations'].shape[1:])
            actions_flat = noised_actions_with_t.reshape((batch_size * n, -1))
            noised_qs_flat = self.noised_network.select('noised_critic')(
                obs_flat, actions=actions_flat, params=grad_params
            )
            noised_qs = noised_qs_flat.reshape((noised_qs_flat.shape[0], batch_size, n))
        else:
            noised_qs = self.noised_network.select('noised_critic')(
                obs_expanded, actions=noised_actions_with_t, params=grad_params
            )
        noised_q = noised_qs.mean(axis=0)                                    # (B, n)

        target_q = original_q[:, None]
        noised_critic_loss = jnp.mean((noised_q - target_q) ** 2)

        # Batch variance of the *original* Q — used by the R²-based gate to
        # normalise the noised-critic loss into a scale-free reliability signal.
        var_q = jnp.var(original_q) + 1e-8

        return noised_critic_loss, {
            'var_q': var_q,
            'noised_critic_loss': noised_critic_loss,
            'noised_q_mean': noised_q.mean(),
            'noised_q_std': noised_q.std(),
        }

    # ------------------------------------------------------------------
    # Combined loss (single jit'd entry point).
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
        """Update the agent and refresh the noised-loss EMA / anchor.

        The `online` flag is accepted for interface compatibility with NFQL but
        is unused: the data-driven gate replaces stage-based switching.
        """
        del online  # interface compatibility
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

        # ---- Update R²-gate EMAs (L_noised and Var[Q]) -----------------
        cur_loss = noised_info['noised_critic_loss']
        cur_var_q = noised_info['var_q']
        decay = self.config['gate_ema_decay']
        uninitialised = self.noised_loss_ema < 0
        new_noised_ema = jnp.where(
            uninitialised,
            cur_loss,
            decay * self.noised_loss_ema + (1.0 - decay) * cur_loss,
        )
        new_var_q_ema = jnp.where(
            uninitialised,
            cur_var_q,
            decay * self.var_q_ema + (1.0 - decay) * cur_var_q,
        )
        info['gate/noised_loss_ema'] = new_noised_ema
        info['gate/var_q_ema'] = new_var_q_ema
        info['gate/r2_ema'] = 1.0 - new_noised_ema / jnp.maximum(new_var_q_ema, 1e-8)

        return self.replace(
            network=new_network,
            noised_network=new_noised_network,
            noised_loss_ema=new_noised_ema,
            var_q_ema=new_var_q_ema,
            rng=new_rng,
        ), info

    # ------------------------------------------------------------------
    # Inference helpers (identical to NFQL).
    # ------------------------------------------------------------------
    @jax.jit
    def sample_actions(self, observations, seed=None, temperature=1.0):
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

        # Sentinels: EMAs < 0 mean "not yet captured" → gate stays closed.
        noised_loss_ema = jnp.asarray(-1.0, dtype=jnp.float32)
        var_q_ema = jnp.asarray(-1.0, dtype=jnp.float32)

        return cls(
            rng,
            network=network,
            noised_network=noised_network,
            noised_loss_ema=noised_loss_ema,
            var_q_ema=var_q_ema,
            config=flax.core.FrozenDict(**config),
        )


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name='nfql_2',
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
            noise_scale=0.1,
            # ---- ESS-targeted weighting ----
            ess_target=0.5,        # target ESS / B for bisection
            # ---- R²-based reliability gate ----
            # R² = 1 − EMA(L_noised) / EMA(Var[Q]).  Scale-free reliability
            # signal; the ratio-based gate of the previous version was broken
            # because L_noised grows as Q grows in absolute magnitude.
            r2_target=0.75,        # gate half-open when R² reaches 0.75
            gate_kappa=0.05,       # sigmoid scale in R² space (5%-wide transition)
            gate_ema_decay=0.999,  # EMA half-life ≈ 693 steps
        )
    )
    return config
