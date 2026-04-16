# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Flow Q-Learning (FQL) is a JAX/Flax-based offline and offline-to-online reinforcement learning implementation. It supports OGBench and D4RL environments.

## Installation

```bash
pip install -r requirements.txt
```

D4RL environments additionally require MuJoCo 2.1.0.

## Running Experiments

```bash
# Basic offline RL run
python main.py --env_name=cube-double-play-singletask-v0 --agent.alpha=300

# Switch agent (fql is default)
python main.py --env_name=... --agent=agents/iql.py --agent.alpha=10

# Offline-to-online RL
python main.py --env_name=scene-play-singletask-v0 --online_steps=1000000 --agent.alpha=300

# Pixel-based environments
python main.py --env_name=visual-cube-single-play-singletask-task1-v0 \
  --offline_steps=500000 --agent.alpha=300 \
  --agent.encoder=impala_small --p_aug=0.5 --frame_stack=3

# Restore from checkpoint
python main.py --env_name=... --restore_path='exp/fql/Debug/sd000_*' --restore_epoch=1000000

# Quick debug run (fewer steps, no real eval)
python main.py --env_name=cube-double-play-singletask-v0 --offline_steps=1000 --eval_interval=0 --log_interval=100
```

Experiments are saved to `exp/<project>/<run_group>/<exp_name>/` with `flags.json`, `train.csv`, and `eval.csv`. Metrics are also logged to wandb (project `fql`).

## Architecture

### Agent Pattern

All agents (`agents/fql.py`, `iql.py`, `rebrac.py`, `ifql.py`, `sac.py`) follow the same interface:
- Subclass `flax.struct.PyTreeNode` — agents are **immutable**; `update()` returns a new agent
- `create(seed, ex_observations, ex_actions, config)` — static factory method
- `update(batch)` — returns `(new_agent, info_dict)`
- `sample_actions(observations, seed, temperature)` — JIT-compiled inference
- `total_loss(batch, grad_params)` — used for validation loss

The agent config is loaded from each agent file's `get_config()` function via `ml_collections.ConfigDict`. The `--agent` flag points to a config file (e.g. `--agent=agents/iql.py`); per-field overrides use `--agent.<field>=<value>`.

### Network Infrastructure (`utils/flax_utils.py`)

- **`ModuleDict`**: wraps multiple `nn.Module`s under one Flax module, allowing joint initialization. Access sub-modules via `network.select('module_name')`.
- **`TrainState`**: custom train state holding model def, params, optimizer. Calling `state(...)` runs a forward pass (stopping gradients by default); pass `params=grad_params` to flow gradients. `apply_loss_fn(loss_fn)` computes gradients and logs grad stats automatically.
- **`save_agent` / `restore_agent`**: pickle-based checkpoint I/O to `params_{epoch}.pkl`.

### Network Modules (`utils/networks.py`)

- `MLP`: base building block with optional layer norm
- `Value`: ensemble critic Q(s,a) or value V(s); uses `nn.vmap` for ensembles
- `Actor`: Gaussian policy (used by IQL, SAC)
- `ActorVectorField`: flow matching vector field u(s, a, t) — the FQL actor

### FQL-Specific Design

FQL maintains three actor components in a single `ModuleDict`:
1. `actor_bc_flow` — BC flow model trained with flow matching loss
2. `actor_onestep_flow` — one-step distilled policy (used for inference)
3. `target_critic` — EMA copy of critic

The actor loss combines: BC flow loss + distillation loss (weighted by `alpha`) + Q loss.
`compute_flow_actions` runs Euler integration (10 steps by default) of the BC flow to generate target actions for distillation.

### Environment & Data (`envs/`, `utils/datasets.py`)

- `make_env_and_datasets(env_name)` in `envs/env_utils.py` handles both OGBench and D4RL environments, returning `(env, eval_env, train_dataset, val_dataset)`
- `Dataset`: static offline dataset with sampling and optional image augmentation
- `ReplayBuffer`: extends `Dataset` with `add_transition()` for online data; also wraps offline data when `balanced_sampling=0`

### Key Hyperparameters

- `--agent.alpha`: BC coefficient — **most important**, must be tuned per environment
- `--agent.normalize_q_loss=True`: makes `alpha` scale-invariant; recommended for new environments
- `--agent.q_agg=min`: enables clipped double Q-learning (vs default `mean`)
- `--agent.discount`: discount factor (0.995 for harder navigation tasks)
