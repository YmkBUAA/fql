import json
import os
from collections.abc import Mapping

from envs.gymnasium_utils import patch_gymnasium_plugin_autoload

import h5py
import numpy as np

patch_gymnasium_plugin_autoload()

import gymnasium
from gymnasium.spaces import Box


PUSHT_ENV_KEYS = {
    'maniskill-pusht-state-v1': dict(asymmetric=False),
    'maniskill-pusht-asym-state-v1': dict(asymmetric=True),
}


def _to_numpy(x):
    """Convert numpy/torch/scalar leaves to numpy arrays."""
    if hasattr(x, 'detach'):
        x = x.detach().cpu().numpy()
    elif hasattr(x, 'cpu') and hasattr(x, 'numpy'):
        x = x.cpu().numpy()
    return np.asarray(x)


def _squeeze_single_env(x):
    x = _to_numpy(x)
    if x.ndim > 0 and x.shape[0] == 1:
        return x[0]
    return x


def _to_scalar_or_array(x):
    if isinstance(x, Mapping):
        result = {}
        for k, v in x.items():
            clean = _to_scalar_or_array(v)
            if clean is not None:
                result[k] = clean
        return result
    arr = _squeeze_single_env(x)
    if arr.dtype.kind in ('O', 'U', 'S'):
        return None
    if arr.shape == ():
        return float(arr)
    if arr.size == 1:
        return float(arr.reshape(-1)[0])
    return arr


def flatten_observation(obs):
    """Flatten ManiSkill state observations into one float32 vector."""
    leaves = []

    def visit(value):
        if isinstance(value, Mapping):
            for key in sorted(value):
                visit(value[key])
        else:
            arr = _squeeze_single_env(value)
            if arr.dtype == np.dtype('O'):
                return
            leaves.append(arr.astype(np.float32).reshape(-1))

    visit(obs)
    if not leaves:
        raise ValueError('No numeric leaves found in ManiSkill observation.')
    return np.concatenate(leaves, axis=0).astype(np.float32)


class ManiSkillFlattenStateWrapper(gymnasium.Wrapper):
    """Flatten ManiSkill state observations and present a non-vector action space."""

    def __init__(self, env):
        super().__init__(env)
        obs, _ = env.reset()
        flat_obs = flatten_observation(obs)
        high = np.full_like(flat_obs, np.inf, dtype=np.float32)
        self.observation_space = Box(low=-high, high=high, dtype=np.float32)

        low = _squeeze_single_env(env.action_space.low).astype(np.float32)
        high = _squeeze_single_env(env.action_space.high).astype(np.float32)
        self._batched_action_space = env.action_space.low.ndim > low.ndim
        self.action_space = Box(low=low, high=high, dtype=np.float32)

    def _env_action(self, action):
        action = np.asarray(action, dtype=np.float32)
        if self._batched_action_space:
            return action[None]
        return action

    def reset(self, *args, **kwargs):
        obs, info = self.env.reset(*args, **kwargs)
        info = _to_scalar_or_array(info) if isinstance(info, Mapping) else {}
        return flatten_observation(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(self._env_action(action))
        reward = float(np.asarray(reward).reshape(-1)[0])
        terminated = bool(np.asarray(terminated).reshape(-1)[0])
        truncated = bool(np.asarray(truncated).reshape(-1)[0])
        info = _to_scalar_or_array(info) if isinstance(info, Mapping) else {}
        return flatten_observation(obs), reward, terminated, truncated, info


class PushTAsymmetricActionWrapper(gymnasium.Wrapper):
    """Action/reward-level stress test that makes one push side unreliable.

    `blocked_side=left` suppresses negative x actions; `right` suppresses
    positive x actions. This keeps the base PushT task intact while testing
    whether a policy can fall back to the other action mode.
    """

    def __init__(self, env, blocked_side='left', penalty=0.05, zero_blocked_axis=True):
        super().__init__(env)
        if blocked_side not in ('left', 'right'):
            raise ValueError("blocked_side must be 'left' or 'right'.")
        self.blocked_side = blocked_side
        self.penalty = penalty
        self.zero_blocked_axis = zero_blocked_axis

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).copy()
        blocked = False
        if action.shape[-1] > 0:
            blocked = (self.blocked_side == 'left' and action[..., 0] < 0) or (
                self.blocked_side == 'right' and action[..., 0] > 0
            )
            if bool(np.asarray(blocked).any()) and self.zero_blocked_axis:
                action[..., 0] = 0.0

        obs, reward, terminated, truncated, info = self.env.step(action)
        if bool(np.asarray(blocked).any()):
            reward = float(reward) - self.penalty
        info = _to_scalar_or_array(info) if isinstance(info, Mapping) else {}
        info['pusht_asym'] = {
            'blocked_side_id': float(-1.0 if self.blocked_side == 'left' else 1.0),
            'blocked_action': float(bool(np.asarray(blocked).any())),
            'fallback_side_used': float(not bool(np.asarray(blocked).any())),
        }
        return obs, reward, terminated, truncated, info


def make_pusht_env(env_name, seed=None):
    if env_name not in PUSHT_ENV_KEYS:
        raise ValueError(f'Unsupported ManiSkill PushT env key: {env_name}')
    try:
        import mani_skill.envs  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            'ManiSkill is required for PushT experiments. Install requirements.txt '
            'or run `pip install mani_skill torch` in the fql environment.'
        ) from exc

    env_kwargs = dict(
        obs_mode='state',
        control_mode='pd_ee_delta_pos',
        reward_mode='normalized_dense',
        render_mode='rgb_array',
        sim_backend=os.environ.get('MANISKILL_SIM_BACKEND', 'physx_cpu'),
    )
    try:
        env = gymnasium.make('PushT-v1', num_envs=1, **env_kwargs)
    except TypeError:
        env = gymnasium.make('PushT-v1', **env_kwargs)
    if seed is not None:
        env.reset(seed=seed)
    env = ManiSkillFlattenStateWrapper(env)
    if PUSHT_ENV_KEYS[env_name]['asymmetric']:
        env = PushTAsymmetricActionWrapper(
            env,
            blocked_side=os.environ.get('PUSHT_ASYM_BLOCKED_SIDE', 'left'),
            penalty=float(os.environ.get('PUSHT_ASYM_PENALTY', '0.05')),
            zero_blocked_axis=os.environ.get('PUSHT_ASYM_ZERO_BLOCKED_AXIS', '1') != '0',
        )
    return env


def _read_metadata(json_path):
    if json_path is None or not os.path.exists(json_path):
        return {}
    with open(json_path) as f:
        metadata = json.load(f)
    episodes = metadata.get('episodes', metadata.get('env_info', {}).get('episodes', []))
    by_name = {}
    for idx, ep in enumerate(episodes):
        name = ep.get('episode_id', ep.get('traj_id', ep.get('id', idx)))
        by_name[f'traj_{name}'] = ep
        by_name[str(name)] = ep
        by_name[f'traj_{idx}'] = ep
    return by_name


def _episode_success(meta):
    for key in ('success', 'is_success', 'episode_success'):
        if key in meta:
            return bool(meta[key])
    return True


def _flatten_h5_group(group):
    leaves = []

    def visit(node):
        if isinstance(node, h5py.Dataset):
            arr = np.asarray(node)
            if arr.dtype == np.dtype('O') or arr.ndim == 0:
                return
            leaves.append(arr.astype(np.float32).reshape(arr.shape[0], -1))
        elif isinstance(node, h5py.Group):
            for key in sorted(node.keys()):
                visit(node[key])

    visit(group)
    if not leaves:
        raise ValueError(f'No numeric datasets found under HDF5 group {group.name}.')
    n = min(arr.shape[0] for arr in leaves)
    return np.concatenate([arr[:n] for arr in leaves], axis=-1).astype(np.float32)


def _first_existing(group, names):
    for name in names:
        if name in group:
            return group[name]
    return None


def _trajectory_keys(h5):
    keys = [k for k in h5.keys() if isinstance(h5[k], h5py.Group) and k.startswith('traj')]
    if keys:
        return sorted(keys, key=lambda k: int(k.split('_')[-1]) if k.split('_')[-1].isdigit() else k)
    if 'traj_0' not in h5 and 'actions' in h5:
        return [None]
    return sorted(k for k in h5.keys() if isinstance(h5[k], h5py.Group))


def load_pusht_dataset(h5_path, json_path=None, only_success=True):
    """Load a replayed ManiSkill PushT trajectory file into FQL Dataset fields."""
    if not os.path.exists(h5_path):
        raise FileNotFoundError(f'ManiSkill PushT trajectory not found: {h5_path}')
    metadata = _read_metadata(json_path)
    parts = {k: [] for k in ('observations', 'actions', 'rewards', 'terminals', 'masks', 'next_observations')}

    with h5py.File(h5_path, 'r') as h5:
        for traj_key in _trajectory_keys(h5):
            group = h5 if traj_key is None else h5[traj_key]
            meta = metadata.get(traj_key or 'traj_0', {})
            if only_success and not _episode_success(meta):
                continue

            actions_node = _first_existing(group, ('actions', 'action'))
            if actions_node is None:
                raise KeyError(f'Missing actions in trajectory group {group.name}.')
            actions = np.asarray(actions_node).astype(np.float32)
            actions = actions.reshape(actions.shape[0], -1)

            obs_node = _first_existing(group, ('obs', 'observations'))
            next_obs_node = _first_existing(group, ('next_obs', 'next_observations'))
            if obs_node is None:
                raise KeyError(f'Missing obs/observations in trajectory group {group.name}.')
            obs_all = _flatten_h5_group(obs_node) if isinstance(obs_node, h5py.Group) else np.asarray(obs_node).astype(np.float32)
            obs_all = obs_all.reshape(obs_all.shape[0], -1)
            if next_obs_node is not None:
                next_obs = (
                    _flatten_h5_group(next_obs_node)
                    if isinstance(next_obs_node, h5py.Group)
                    else np.asarray(next_obs_node).astype(np.float32)
                )
                next_obs = next_obs.reshape(next_obs.shape[0], -1)
                obs = obs_all
            else:
                obs = obs_all[:-1]
                next_obs = obs_all[1:]

            rewards_node = _first_existing(group, ('rewards', 'reward'))
            if rewards_node is None:
                rewards = np.zeros((actions.shape[0],), dtype=np.float32)
            else:
                rewards = np.asarray(rewards_node).astype(np.float32).reshape(-1)

            terminated_node = _first_existing(group, ('terminated', 'terminations', 'terminals', 'dones'))
            truncated_node = _first_existing(group, ('truncated', 'timeouts'))
            if terminated_node is None:
                terminated = np.zeros((actions.shape[0],), dtype=np.float32)
                terminated[-1] = 1.0
            else:
                terminated = np.asarray(terminated_node).astype(np.float32).reshape(-1)
            if truncated_node is None:
                truncated = np.zeros_like(terminated)
            else:
                truncated = np.asarray(truncated_node).astype(np.float32).reshape(-1)

            n = min(obs.shape[0], next_obs.shape[0], actions.shape[0], rewards.shape[0], terminated.shape[0], truncated.shape[0])
            terminals = np.maximum(terminated[:n], truncated[:n]).astype(np.float32)
            masks = (1.0 - terminated[:n]).astype(np.float32)

            parts['observations'].append(obs[:n].astype(np.float32))
            parts['next_observations'].append(next_obs[:n].astype(np.float32))
            parts['actions'].append(actions[:n].astype(np.float32))
            parts['rewards'].append(rewards[:n].astype(np.float32))
            parts['terminals'].append(terminals)
            parts['masks'].append(masks)

    if not parts['observations']:
        raise ValueError(f'No trajectories loaded from {h5_path}; check only_success={only_success}.')
    dataset = {key: np.concatenate(value, axis=0) for key, value in parts.items()}
    for key, value in dataset.items():
        if not np.all(np.isfinite(value)):
            raise ValueError(f'Non-finite values found in loaded PushT dataset field {key}.')
    return dataset


def get_pusht_dataset_paths():
    data_dir = os.environ.get('MANISKILL_PUSHT_DATA_DIR', 'data/maniskill/PushT-v1')
    h5_path = os.environ.get('MANISKILL_PUSHT_H5', os.path.join(data_dir, 'trajectory.state.pd_ee_delta_pos.physx_cuda.h5'))
    json_path = os.environ.get('MANISKILL_PUSHT_JSON', os.path.splitext(h5_path)[0] + '.json')
    return h5_path, json_path
