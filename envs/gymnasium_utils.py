import importlib.metadata as importlib_metadata


def patch_gymnasium_plugin_autoload():
    """Filter Gymnasium entry-point plugins that import fragile legacy stacks.

    Gymnasium 0.29 auto-loads third-party plugins on import. In this repo,
    shimmy is installed for D4RL compatibility, but its Gymnasium plugin imports
    mujoco_py eagerly. If mujoco_py was compiled against a different NumPy ABI,
    even unrelated OGBench/ManiSkill imports fail before training starts.
    """
    if getattr(importlib_metadata, '_fql_gym_plugin_patch', False):
        return

    original_entry_points = importlib_metadata.entry_points

    def entry_points_without_shimmy(*args, **kwargs):
        eps = original_entry_points(*args, **kwargs)
        if kwargs.get('group') == 'gymnasium.envs':
            return [ep for ep in eps if not str(getattr(ep, 'value', '')).startswith('shimmy')]
        return eps

    importlib_metadata.entry_points = entry_points_without_shimmy
    importlib_metadata._fql_gym_plugin_patch = True
