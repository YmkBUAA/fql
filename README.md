<div align="center">

<div id="user-content-toc" style="margin-bottom: 50px">
  <ul align="center" style="list-style: none;">
    <summary>
      <h1>Flow Q-Learning</h1>
      <br>
      <h2><a href="https://arxiv.org/abs/2502.02538">Paper</a> &emsp; <a href="https://seohong.me/projects/fql/">Project page</a></h2>
    </summary>
  </ul>
</div>

<img src="assets/fql.png" width="80%">

</div>

## Overview

Flow Q-learning (FQL) is a simple and performance data-driven RL algorithm
that leverages an expressive *flow-matching* policy
to model complex action distributions in data.

## Installation

FQL requires Python 3.9+ and is based on JAX. The main dependencies are
`jax >= 0.4.26`, `ogbench == 1.1.0`, and `gymnasium == 0.29.1`.
To install the full dependencies, simply run:
```bash
pip install -r requirements.txt
```

> [!NOTE]
> To use D4RL environments, you need to additionally set up MuJoCo 2.1.0.

## Usage

The main implementation of FQL is in [agents/fql.py](agents/fql.py),
and our implementations of four baselines (IQL, ReBRAC, IFQL, and RLPD)
can also be found in the same directory.
Here are some example commands (see [the section below](#reproducing-the-main-results) for the complete list):
```bash
# FQL on OGBench antsoccer-arena (offline RL)
python main.py --env_name=antsoccer-arena-navigate-singletask-v0 --agent.discount=0.995 --agent.alpha=10
# FQL on OGBench visual-cube-single (offline RL)
python main.py --env_name=visual-cube-single-play-singletask-task1-v0 --offline_steps=500000 --agent.alpha=300 --agent.encoder=impala_small --p_aug=0.5 --frame_stack=3
# FQL on OGBench scene (offline-to-online RL)
python main.py --env_name=scene-play-singletask-v0 --online_steps=1000000 --agent.alpha=300
```

## ManiSkill PushT offline-to-online experiments

This fork includes an experimental ManiSkill PushT pipeline for testing
whether flow policies preserve multi-modal action distributions during
offline-to-online fine-tuning. The first version uses state observations
from ManiSkill `PushT-v1` and focuses on action modes such as left-push vs
right-push, not RGB/state sensor fusion.

Install the extra dependencies from `requirements.txt`:
```bash
pip install -r requirements.txt
```

Prepare the PushT demonstrations from
`haosulab/ManiSkill_Demonstrations`:
```bash
python scripts/prepare_maniskill_pusht_data.py
```

The data preparation script downloads
`demos/PushT-v1/rl/trajectory.none.pd_ee_delta_pos.physx_cuda.h5` and replays
it with ManiSkill state observations and dense normalized rewards. It defaults
to `--sim-backend=physx_cpu` so the replay does not require a CUDA-compatible
PyTorch/driver pair. Use `--sim-backend=physx_cuda` only if your driver and
PyTorch build support it. By default the training code expects the converted
trajectory at:
```bash
data/maniskill/PushT-v1/trajectory.state.pd_ee_delta_pos.physx_cuda.h5
```

If ManiSkill writes a different filename, point the loader to it explicitly:
```bash
export MANISKILL_PUSHT_H5=/path/to/converted_trajectory.h5
export MANISKILL_PUSHT_JSON=/path/to/converted_trajectory.json
```

Run a quick launch check:
```bash
bash scripts/run_exp_maniskill_pusht_o2o_multimodal.sh dryrun
```

Launch the default sweep:
```bash
bash scripts/run_exp_maniskill_pusht_o2o_multimodal.sh
```

The default sweep trains on `maniskill-pusht-state-v1`, evaluates on both
standard PushT and `maniskill-pusht-asym-state-v1`, and runs:
`fql`, `fql_v_k1`, `fql_v_k4`, `fql_ar`, and `fql_pi` across seeds `0 1 2`.
The asymmetric environment is an action/reward-level stress test that blocks
one action side by default. Override it with:
```bash
PUSHT_ASYM_BLOCKED_SIDE=right PUSHT_ASYM_PENALTY=0.1 \
    bash scripts/run_exp_maniskill_pusht_o2o_multimodal.sh
```

PushT runs log additional action-mode diagnostics at evaluation checkpoints:
`action_modes/left_ratio`, `action_modes/right_ratio`,
`action_modes/mode_entropy`, `action_modes/mode_coverage`, and
`action_modes/lr_balance`. The sampled actions are also saved under each run's
`action_samples/` directory unless `--save_action_samples=False` is passed.

## Tips for hyperparameter tuning

Here are some general tips for FQL's hyperparameter tuning for new tasks:

* The most important hyperparameter of FQL is the BC coefficient (`--agent.alpha`).
  This needs to be individually tuned for each environment.
* Although this was not used in the original paper,
  setting `--agent.normalize_q_loss=True` makes `alpha` invariant to the scale of the Q-values.
  **For new environments, we highly recommend turning on this flag** (`--agent.normalize_q_loss=True`)
  and tuning `alpha` starting from `[0.03, 0.1, 0.3, 1, 3, 10]`.
* For other hyperparameters, you may use the default values in `agents/fql.py`.
  For some tasks, setting `--agent.q_agg=min` (to enable clipped double Q-learning) may slightly improve performance.
  See the ablation study in the paper for more details.
* For pixel-based environments, don't forget to set `--agent.encoder=impala_small` (or larger encoders),
  `--p_aug=0.5`, and `--frame_stack=3`.

## Reproducing the main results

We provide the complete list of the **exact command-line flags**
used to produce the main results of FQL in the paper.

> [!NOTE]
> In OGBench, each environment provides five tasks, one of which is the default task.
> This task corresponds to the environment ID without any task suffixes.
> For example, the default task of `antmaze-large-navigate` is `task1`,
> and `antmaze-large-navigate-singletask-v0` is the same environment as `antmaze-large-navigate-singletask-task1-v0`.

<details>
<summary><b>Click to expand the full list of commands</b></summary>

### Offline RL

#### FQL on state-based OGBench (default tasks)

```bash
# FQL on OGBench antmaze-large-navigate-singletask-v0 (=antmaze-large-navigate-singletask-task1-v0)
python main.py --env_name=antmaze-large-navigate-singletask-v0 --agent.q_agg=min --agent.alpha=10
# FQL on OGBench antmaze-giant-navigate-singletask-v0 (=antmaze-giant-navigate-singletask-task1-v0)
python main.py --env_name=antmaze-giant-navigate-singletask-v0 --agent.discount=0.995 --agent.q_agg=min --agent.alpha=10
# FQL on OGBench humanoidmaze-medium-navigate-singletask-v0 (=humanoidmaze-medium-navigate-singletask-task1-v0)
python main.py --env_name=humanoidmaze-medium-navigate-singletask-v0 --agent.discount=0.995 --agent.alpha=30
# FQL on OGBench humanoidmaze-large-navigate-singletask-v0 (=humanoidmaze-large-navigate-singletask-task1-v0)
python main.py --env_name=humanoidmaze-large-navigate-singletask-v0 --agent.discount=0.995 --agent.alpha=30
# FQL on OGBench antsoccer-arena-navigate-singletask-v0 (=antsoccer-arena-navigate-singletask-task4-v0)
python main.py --env_name=antsoccer-arena-navigate-singletask-v0 --agent.discount=0.995 --agent.alpha=10
# FQL on OGBench cube-single-play-singletask-v0 (=cube-single-play-singletask-task2-v0)
python main.py --env_name=cube-single-play-singletask-v0 --agent.alpha=300
# FQL on OGBench cube-double-play-singletask-v0 (=cube-double-play-singletask-task2-v0)
python main.py --env_name=cube-double-play-singletask-v0 --agent.alpha=300
# FQL on OGBench scene-play-singletask-v0 (=scene-play-singletask-task2-v0)
python main.py --env_name=scene-play-singletask-v0 --agent.alpha=300
# FQL on OGBench puzzle-3x3-play-singletask-v0 (=puzzle-3x3-play-singletask-task4-v0)
python main.py --env_name=puzzle-3x3-play-singletask-v0 --agent.alpha=1000
# FQL on OGBench puzzle-4x4-play-singletask-v0 (=puzzle-4x4-play-singletask-task4-v0)
python main.py --env_name=puzzle-4x4-play-singletask-v0 --agent.alpha=1000
```

#### FQL on state-based OGBench (all tasks)

```bash
# FQL on OGBench antmaze-large-navigate-singletask-{task1, task2, task3, task4, task5}-v0 (default: task1)
python main.py --env_name=antmaze-large-navigate-singletask-task1-v0 --agent.q_agg=min --agent.alpha=10
python main.py --env_name=antmaze-large-navigate-singletask-task2-v0 --agent.q_agg=min --agent.alpha=10
python main.py --env_name=antmaze-large-navigate-singletask-task3-v0 --agent.q_agg=min --agent.alpha=10
python main.py --env_name=antmaze-large-navigate-singletask-task4-v0 --agent.q_agg=min --agent.alpha=10
python main.py --env_name=antmaze-large-navigate-singletask-task5-v0 --agent.q_agg=min --agent.alpha=10
# FQL on OGBench antmaze-giant-navigate-singletask-{task1, task2, task3, task4, task5}-v0 (default: task1)
python main.py --env_name=antmaze-giant-navigate-singletask-task1-v0 --agent.discount=0.995 --agent.q_agg=min --agent.alpha=10
python main.py --env_name=antmaze-giant-navigate-singletask-task2-v0 --agent.discount=0.995 --agent.q_agg=min --agent.alpha=10
python main.py --env_name=antmaze-giant-navigate-singletask-task3-v0 --agent.discount=0.995 --agent.q_agg=min --agent.alpha=10
python main.py --env_name=antmaze-giant-navigate-singletask-task4-v0 --agent.discount=0.995 --agent.q_agg=min --agent.alpha=10
python main.py --env_name=antmaze-giant-navigate-singletask-task5-v0 --agent.discount=0.995 --agent.q_agg=min --agent.alpha=10
# FQL on OGBench humanoidmaze-medium-navigate-singletask-{task1, task2, task3, task4, task5}-v0 (default: task1)
python main.py --env_name=humanoidmaze-medium-navigate-singletask-task1-v0 --agent.discount=0.995 --agent.alpha=30
python main.py --env_name=humanoidmaze-medium-navigate-singletask-task2-v0 --agent.discount=0.995 --agent.alpha=30
python main.py --env_name=humanoidmaze-medium-navigate-singletask-task3-v0 --agent.discount=0.995 --agent.alpha=30
python main.py --env_name=humanoidmaze-medium-navigate-singletask-task4-v0 --agent.discount=0.995 --agent.alpha=30
python main.py --env_name=humanoidmaze-medium-navigate-singletask-task5-v0 --agent.discount=0.995 --agent.alpha=30
# FQL on OGBench humanoidmaze-large-navigate-singletask-{task1, task2, task3, task4, task5}-v0 (default: task1)
python main.py --env_name=humanoidmaze-large-navigate-singletask-task1-v0 --agent.discount=0.995 --agent.alpha=30
python main.py --env_name=humanoidmaze-large-navigate-singletask-task2-v0 --agent.discount=0.995 --agent.alpha=30
python main.py --env_name=humanoidmaze-large-navigate-singletask-task3-v0 --agent.discount=0.995 --agent.alpha=30
python main.py --env_name=humanoidmaze-large-navigate-singletask-task4-v0 --agent.discount=0.995 --agent.alpha=30
python main.py --env_name=humanoidmaze-large-navigate-singletask-task5-v0 --agent.discount=0.995 --agent.alpha=30
# FQL on OGBench antsoccer-arena-navigate-singletask-{task1, task2, task3, task4, task5}-v0 (default: task4)
python main.py --env_name=antsoccer-arena-navigate-singletask-task1-v0 --agent.discount=0.995 --agent.alpha=10
python main.py --env_name=antsoccer-arena-navigate-singletask-task2-v0 --agent.discount=0.995 --agent.alpha=10
python main.py --env_name=antsoccer-arena-navigate-singletask-task3-v0 --agent.discount=0.995 --agent.alpha=10
python main.py --env_name=antsoccer-arena-navigate-singletask-task4-v0 --agent.discount=0.995 --agent.alpha=10
python main.py --env_name=antsoccer-arena-navigate-singletask-task5-v0 --agent.discount=0.995 --agent.alpha=10
# FQL on OGBench cube-single-play-singletask-{task1, task2, task3, task4, task5}-v0 (default: task2)
python main.py --env_name=cube-single-play-singletask-task1-v0 --agent.alpha=300
python main.py --env_name=cube-single-play-singletask-task2-v0 --agent.alpha=300
python main.py --env_name=cube-single-play-singletask-task3-v0 --agent.alpha=300
python main.py --env_name=cube-single-play-singletask-task4-v0 --agent.alpha=300
python main.py --env_name=cube-single-play-singletask-task5-v0 --agent.alpha=300
# FQL on OGBench cube-double-play-singletask-{task1, task2, task3, task4, task5}-v0 (default: task2)
python main.py --env_name=cube-double-play-singletask-task1-v0 --agent.alpha=300
python main.py --env_name=cube-double-play-singletask-task2-v0 --agent.alpha=300
python main.py --env_name=cube-double-play-singletask-task3-v0 --agent.alpha=300
python main.py --env_name=cube-double-play-singletask-task4-v0 --agent.alpha=300
python main.py --env_name=cube-double-play-singletask-task5-v0 --agent.alpha=300
# FQL on OGBench scene-play-singletask-{task1, task2, task3, task4, task5}-v0 (default: task2)
python main.py --env_name=scene-play-singletask-task1-v0 --agent.alpha=300
python main.py --env_name=scene-play-singletask-task2-v0 --agent.alpha=300
python main.py --env_name=scene-play-singletask-task3-v0 --agent.alpha=300
python main.py --env_name=scene-play-singletask-task4-v0 --agent.alpha=300
python main.py --env_name=scene-play-singletask-task5-v0 --agent.alpha=300
# FQL on OGBench puzzle-3x3-play-singletask-{task1, task2, task3, task4, task5}-v0 (default: task4)
python main.py --env_name=puzzle-3x3-play-singletask-task1-v0 --agent.alpha=1000
python main.py --env_name=puzzle-3x3-play-singletask-task2-v0 --agent.alpha=1000
python main.py --env_name=puzzle-3x3-play-singletask-task3-v0 --agent.alpha=1000
python main.py --env_name=puzzle-3x3-play-singletask-task4-v0 --agent.alpha=1000
python main.py --env_name=puzzle-3x3-play-singletask-task5-v0 --agent.alpha=1000
# FQL on OGBench puzzle-4x4-play-singletask-{task1, task2, task3, task4, task5}-v0 (default: task4)
python main.py --env_name=puzzle-4x4-play-singletask-task1-v0 --agent.alpha=1000
python main.py --env_name=puzzle-4x4-play-singletask-task2-v0 --agent.alpha=1000
python main.py --env_name=puzzle-4x4-play-singletask-task3-v0 --agent.alpha=1000
python main.py --env_name=puzzle-4x4-play-singletask-task4-v0 --agent.alpha=1000
python main.py --env_name=puzzle-4x4-play-singletask-task5-v0 --agent.alpha=1000
```

#### FQL on pixel-based OGBench

```bash
# FQL on OGBench visual-cube-single-play-singletask-task1-v0
python main.py --env_name=visual-cube-single-play-singletask-task1-v0 --offline_steps=500000 --agent.alpha=300 --agent.encoder=impala_small --p_aug=0.5 --frame_stack=3
# FQL on OGBench visual-cube-double-play-singletask-task1-v0
python main.py --env_name=visual-cube-double-play-singletask-task1-v0 --offline_steps=500000 --agent.alpha=100 --agent.encoder=impala_small --p_aug=0.5 --frame_stack=3
# FQL on OGBench visual-scene-play-singletask-task1-v0
python main.py --env_name=visual-scene-play-singletask-task1-v0 --offline_steps=500000 --agent.alpha=100 --agent.encoder=impala_small --p_aug=0.5 --frame_stack=3
# FQL on OGBench visual-puzzle-3x3-play-singletask-task1-v0
python main.py --env_name=visual-puzzle-3x3-play-singletask-task1-v0 --offline_steps=500000 --agent.alpha=300 --agent.encoder=impala_small --p_aug=0.5 --frame_stack=3
# FQL on OGBench visual-puzzle-4x4-play-singletask-task1-v0
python main.py --env_name=visual-puzzle-4x4-play-singletask-task1-v0 --offline_steps=500000 --agent.alpha=300 --agent.encoder=impala_small --p_aug=0.5 --frame_stack=3
```

#### FQL on D4RL

```bash
# FQL on D4RL antmaze-umaze-v2
python main.py --env_name=antmaze-umaze-v2 --offline_steps=500000 --agent.alpha=10
# FQL on D4RL antmaze-umaze-diverse-v2
python main.py --env_name=antmaze-umaze-diverse-v2 --offline_steps=500000 --agent.alpha=10
# FQL on D4RL antmaze-medium-play-v2
python main.py --env_name=antmaze-medium-play-v2 --offline_steps=500000 --agent.alpha=10
# FQL on D4RL antmaze-medium-diverse-v2
python main.py --env_name=antmaze-medium-diverse-v2 --offline_steps=500000 --agent.alpha=10
# FQL on D4RL antmaze-large-play-v2
python main.py --env_name=antmaze-large-play-v2 --offline_steps=500000 --agent.alpha=3
# FQL on D4RL antmaze-large-diverse-v2
python main.py --env_name=antmaze-large-diverse-v2 --offline_steps=500000 --agent.alpha=3
# FQL on D4RL pen-human-v1
python main.py --env_name=pen-human-v1 --offline_steps=500000 --agent.q_agg=min --agent.alpha=10000
# FQL on D4RL pen-cloned-v1
python main.py --env_name=pen-cloned-v1 --offline_steps=500000 --agent.q_agg=min --agent.alpha=10000
# FQL on D4RL pen-expert-v1
python main.py --env_name=pen-expert-v1 --offline_steps=500000 --agent.q_agg=min --agent.alpha=3000
# FQL on D4RL door-human-v1
python main.py --env_name=door-human-v1 --offline_steps=500000 --agent.q_agg=min --agent.alpha=30000
# FQL on D4RL door-cloned-v1
python main.py --env_name=door-cloned-v1 --offline_steps=500000 --agent.q_agg=min --agent.alpha=30000
# FQL on D4RL door-expert-v1
python main.py --env_name=door-expert-v1 --offline_steps=500000 --agent.q_agg=min --agent.alpha=30000
# FQL on D4RL hammer-human-v1
python main.py --env_name=hammer-human-v1 --offline_steps=500000 --agent.q_agg=min --agent.alpha=30000
# FQL on D4RL hammer-cloned-v1
python main.py --env_name=hammer-cloned-v1 --offline_steps=500000 --agent.q_agg=min --agent.alpha=10000
# FQL on D4RL hammer-expert-v1
python main.py --env_name=hammer-expert-v1 --offline_steps=500000 --agent.q_agg=min --agent.alpha=30000
# FQL on D4RL relocate-human-v1
python main.py --env_name=relocate-human-v1 --offline_steps=500000 --agent.q_agg=min --agent.alpha=10000
# FQL on D4RL relocate-cloned-v1
python main.py --env_name=relocate-cloned-v1 --offline_steps=500000 --agent.q_agg=min --agent.alpha=30000
# FQL on D4RL relocate-expert-v1
python main.py --env_name=relocate-expert-v1 --offline_steps=500000 --agent.q_agg=min --agent.alpha=30000
```

#### IQL, ReBRAC, and IFQL (examples)

```bash
# IQL on OGBench humanoidmaze-medium-navigate-singletask-v0
python main.py --env_name=humanoidmaze-medium-navigate-singletask-v0 --agent=agents/iql.py --agent.discount=0.995 --agent.alpha=10
# ReBRAC on OGBench humanoidmaze-medium-navigate-singletask-v0 
python main.py --env_name=humanoidmaze-medium-navigate-singletask-v0 --agent=agents/rebrac.py --agent.discount=0.995 --agent.alpha_actor=0.01 --agent.alpha_critic=0.01
# IFQL on OGBench humanoidmaze-medium-navigate-singletask-v0
python main.py --env_name=humanoidmaze-medium-navigate-singletask-v0 --agent=agents/ifql.py --agent.discount=0.995 --agent.num_samples=32
# IQL on OGBench visual-cube-single-play-singletask-task1-v0
python main.py --env_name=visual-cube-single-play-singletask-task1-v0 --offline_steps=500000 --agent=agents/iql.py --agent.alpha=1 --agent.encoder=impala_small --p_aug=0.5 --frame_stack=3
# ReBRAC on OGBench visual-cube-single-play-singletask-task1-v0
python main.py --env_name=visual-cube-single-play-singletask-task1-v0 --offline_steps=500000 --agent=agents/rebrac.py --agent.alpha_actor=1 --agent.alpha_critic=0 --agent.encoder=impala_small --p_aug=0.5 --frame_stack=3
# IFQL on OGBench visual-cube-single-play-singletask-task1-v0
python main.py --env_name=visual-cube-single-play-singletask-task1-v0 --offline_steps=500000 --agent=agents/ifql.py --agent.num_samples=32 --agent.encoder=impala_small --p_aug=0.5 --frame_stack=3
```

### Offline-to-online RL

#### FQL

```bash
# FQL on OGBench humanoidmaze-medium-navigate-singletask-v0
python main.py --env_name=humanoidmaze-medium-navigate-singletask-v0 --online_steps=1000000 --agent.discount=0.995 --agent.alpha=100
# FQL on OGBench antsoccer-arena-navigate-singletask-v0
python main.py --env_name=antsoccer-arena-navigate-singletask-v0 --online_steps=1000000 --agent.discount=0.995 --agent.alpha=30
# FQL on OGBench cube-double-play-singletask-v0
python main.py --env_name=cube-double-play-singletask-v0 --online_steps=1000000 --agent.alpha=300
# FQL on OGBench scene-play-singletask-v0
python main.py --env_name=scene-play-singletask-v0 --online_steps=1000000 --agent.alpha=300
# FQL on OGBench puzzle-4x4-play-singletask-v0
python main.py --env_name=puzzle-4x4-play-singletask-v0 --online_steps=1000000 --agent.alpha=1000
# FQL on D4RL antmaze-umaze-v2
python main.py --env_name=antmaze-umaze-v2 --online_steps=1000000 --agent.alpha=10
# FQL on D4RL antmaze-umaze-diverse-v2
python main.py --env_name=antmaze-umaze-diverse-v2 --online_steps=1000000 --agent.alpha=10
# FQL on D4RL antmaze-medium-play-v2
python main.py --env_name=antmaze-medium-play-v2 --online_steps=1000000 --agent.alpha=10
# FQL on D4RL antmaze-medium-diverse-v2
python main.py --env_name=antmaze-medium-diverse-v2 --online_steps=1000000 --agent.alpha=10
# FQL on D4RL antmaze-large-play-v2
python main.py --env_name=antmaze-large-play-v2 --online_steps=1000000 --agent.alpha=3
# FQL on D4RL antmaze-large-diverse-v2
python main.py --env_name=antmaze-large-diverse-v2 --online_steps=1000000 --agent.alpha=3
# FQL on D4RL pen-cloned-v1
python main.py --env_name=pen-cloned-v1 --online_steps=1000000 --agent.q_agg=min --agent.alpha=1000
# FQL on D4RL door-cloned-v1
python main.py --env_name=door-cloned-v1 --online_steps=1000000 --agent.q_agg=min --agent.alpha=1000
# FQL on D4RL hammer-cloned-v1
python main.py --env_name=hammer-cloned-v1 --online_steps=1000000 --agent.q_agg=min --agent.alpha=1000
# FQL on D4RL relocate-cloned-v1
python main.py --env_name=relocate-cloned-v1 --online_steps=1000000 --agent.q_agg=min --agent.alpha=10000
```

#### IQL, ReBRAC, IFQL, and RLPD (examples)

```bash
# IQL on OGBench humanoidmaze-medium-navigate-singletask-v0
python main.py --env_name=humanoidmaze-medium-navigate-singletask-v0 --online_steps=1000000 --agent=agents/iql.py --agent.discount=0.995 --agent.alpha=10
# ReBRAC on OGBench humanoidmaze-medium-navigate-singletask-v0 
python main.py --env_name=humanoidmaze-medium-navigate-singletask-v0 --online_steps=1000000 --agent=agents/rebrac.py --agent.discount=0.995 --agent.alpha_actor=0.01 --agent.alpha_critic=0.01
# IFQL on OGBench humanoidmaze-medium-navigate-singletask-v0
python main.py --env_name=humanoidmaze-medium-navigate-singletask-v0 --online_steps=1000000 --agent=agents/ifql.py --agent.discount=0.995 --agent.num_samples=32
# RLPD on OGBench humanoidmaze-medium-navigate-singletask-v0
python main.py --env_name=humanoidmaze-medium-navigate-singletask-v0 --offline_steps=0 --online_steps=1000000 --agent=agents/sac.py --agent.discount=0.995 --balanced_sampling=1
```
</details>

## Acknowledgments

This codebase is built on top of [OGBench](https://github.com/seohongpark/ogbench)'s reference implementations.
