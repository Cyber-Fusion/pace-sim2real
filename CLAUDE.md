# CLAUDE.md

## Project Overview

PACE (Precise Adaptation through Continuous Evolution) is a sim-to-real transfer framework for legged robots built on **NVIDIA Isaac Lab**. It uses CMA-ES evolutionary optimization to fit actuator/joint dynamics parameters from measured data, improving simulation fidelity for locomotion tasks.

## Key Commands

```bash
# Data collection (chirp excitation signals)
python scripts/pace/data_collection.py --task Isaac-Pace-Ayg-v0

# Parameter fitting via CMA-ES
python scripts/pace/fit.py

# Plot fitted trajectories
python scripts/pace/plot_trajectory.py

# List registered environments
python scripts/list_envs.py

# Pre-commit (formatting + linting)
pre-commit run --all-files
```

Scripts use Isaac Lab's `AppLauncher` and support `--task`, `--num_envs`, `--device` flags.

## Architecture

### Package: `source/pace_sim2real/pace_sim2real/`

- **`optim/cma_es.py`** — `CMAESOptimizer`: evolutionary parameter fitting with TensorBoard logging
- **`utils/pace_actuator.py`** + **`pace_actuator_cfg.py`** — `PaceDCMotor` / `PaceDCMotorCfg`: custom actuator model
- **`tasks/manager_based/`** — Gymnasium environment registration and configs:
  - `pace/pace_sim2real_env_cfg.py` — Base environment config (`PaceSim2realEnvCfg`)
  - `pace/anymal_pace_env_cfg.py` — ANYmal D robot config
  - `pace/ayg_pace_env_cfg.py` — Ayg robot config
  - `pace/mdp/rewards.py` — Reward functions
  - `pace/agents/rsl_rl_ppo_cfg.py` — PPO training config

### Adding a New Robot

1. Create a new `<robot>_pace_env_cfg.py` in `tasks/manager_based/pace/` inheriting from `PaceSim2realEnvCfg`
2. Register a Gymnasium environment in `tasks/manager_based/__init__.py`

### Registered Environments

- `Isaac-Pace-Anymal-D-v0` → `AnymalDPaceEnvCfg`
- `Isaac-Pace-Ayg-v0` → `AygPaceEnvCfg`

### Data Flow

1. `data_collection.py` → collects chirp data → saves to `data/<robot>/chirp_data.pt`
2. `fit.py` → loads data → runs CMA-ES optimization → logs to `logs/pace/<robot>/`

## Entry Points

- **Scripts**: `scripts/pace/` (data collection, fitting, plotting)
- **RL training**: `scripts/rsl_rl/train.py` and `play.py`
- **Public API**: `from pace_sim2real import PaceCfg, PaceSim2realEnvCfg, CMAESOptimizer`
