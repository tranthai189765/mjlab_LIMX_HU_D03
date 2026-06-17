# mjlab architecture (deliverable 1 reference)

A grounded explanation of the mjlab codebase: structure, modules, how it works,
and the training algorithm. Verified against the repo at the time of writing.

## What mjlab is

mjlab = **Isaac Lab's manager-based API** running on **MuJoCo Warp** (the
GPU-accelerated MuJoCo). It simulates thousands of robot copies in parallel on an
NVIDIA GPU and trains control policies with reinforcement learning.

- Physics backend: `mujoco-warp` (CUDA) — **NVIDIA GPU required for training**;
  macOS/CPU is evaluation-only.
- RL algorithm: **PPO** via `rsl-rl-lib==5.4.0`.
- Config language: plain Python dataclasses (no YAML). Run with `uv run`.

## Directory map (`src/mjlab/`)

| Module | Role |
| --- | --- |
| `sim/` | Wraps MuJoCo Warp. `Simulation` + `SimulationCfg` (timestep, contacts, GPU). The physics engine. |
| `entity/` | `Entity` = one body group in the scene. `EntityCfg` points at the MJCF, actuators, init state, collisions. |
| `actuator/` | Motor models, e.g. `BuiltinPositionActuatorCfg` (stiffness, damping, effort/velocity limits, armature). |
| `scene/` | `Scene` composes entities + terrain + sensors. |
| `terrains/` | Flat/rough terrain generation with difficulty curriculum. |
| `sensor/` | Contact sensors (feet, self-collision), raycast terrain-height sensors. |
| `managers/` | Heart of the manager-based API. One manager per MDP concern: Observation, Reward, Termination, Event (randomization), Command, Curriculum, Action. |
| `envs/` | `ManagerBasedRlEnvCfg` glues managers into a Gym env. `envs/mdp/` holds shared MDP terms. |
| `rl/` | Bridge to rsl_rl: PPO cfg (`RslRlPpoAlgorithmCfg`), actor/critic nets, runner, ONNX export for deployment. |
| `asset_zoo/robots/` | Robot library. Currently `unitree_g1`, `unitree_go1`, `i2rt_yam`. Each robot = an MJCF (`.xml`) + a `*_constants.py`. **New robots go here.** |
| `tasks/` | Concrete problems. Key ones: `velocity` (locomotion) and `tracking` (mimic). Also `cartpole`, `manipulation`. |
| `scripts/` | CLI entry points: `train`, `play`, `demo`, `list-envs`, and `csv_to_npz.py` (motion preprocessing for mimic). |

## How it works (control flow)

```
robot constants (MJCF + actuators)            -> EntityCfg
env_cfgs.py (robot + scene + rewards + obs + commands + events) -> ManagerBasedRlEnvCfg
register_mjlab_task("Mjlab-...-G1", env_cfg, rl_cfg, runner_cls)  -> a task id
uv run train <task-id>  -> N parallel envs on GPU -> PPO -> checkpoints + ONNX
uv run play  <task-id>  -> roll out / visualize the policy
```

Each robot+task pair is registered as a **task id** (e.g.
`Mjlab-Velocity-Flat-Unitree-G1`) inside `tasks/<task>/config/<robot>/__init__.py`
via `register_mjlab_task(...)`. `train`/`play` only need that id. `uv run list-envs`
lists everything registered.

## Training algorithm

From `tasks/velocity/config/g1/rl_cfg.py`:

- **PPO**, on-policy, actor-critic.
- Networks: MLP `(512, 256, 128)`, `elu` activation, observation normalization.
- Key hyperparameters: `clip_param=0.2`, `entropy_coef=0.01`, `gamma=0.99`,
  `lam=0.95` (GAE), `learning_rate=1e-3` with **adaptive KL schedule**
  (`desired_kl=0.01`), `max_grad_norm=1.0`.
- Loop: collect `num_steps_per_env=24` steps × `num_envs` (e.g. 4096) -> one
  batch -> update for `num_learning_epochs=5` × `num_mini_batches=4`. Run to
  `max_iterations=30_000`.
- The PPO core is identical across tasks. What differs between locomotion and
  mimic is only the **rewards, observations, and commands**.

## The two tasks relevant to HU_D03

**Locomotion = `velocity`** (`tasks/velocity/`): the robot tracks randomly
sampled velocity commands (vx, vy, yaw-rate). Rewards favor command tracking,
balance, natural gait; penalize falling/collisions. **No external data.**

**Mimic = `tracking`** (`tasks/tracking/`): the robot imitates a reference
motion. Requires a preprocessing step — `scripts/csv_to_npz.py` turns a
retargeted motion CSV into an NPZ (plays it through MuJoCo, runs forward
kinematics, stores per-body pos/quat/vel indexed by MuJoCo body order), uploaded
to a WandB `motions` registry. Training pulls it via `--registry-name`.

### Motion data convention (mimic)

- Reference CSV = "Unitree generalized coordinate convention", one row per frame:
  `[base_pos(3)] [base_quat xyzw(4)] [joint angles in a fixed order]`.
- **You must use mjlab's `csv_to_npz`.** NPZ stores bodies by MuJoCo's depth-first
  index; converters from other engines (PhysX/IsaacLab BFS) map bodies wrongly and
  training won't converge.
- mjlab provides no retargeter and no motion data. Producing a CSV retargeted to a
  given robot's skeleton is an external step (e.g. LAFAN1 -> GMR).

## Third-party / dependencies of note

- `mujoco~=3.8.0`, `mujoco-warp~=3.9.0`, `warp-lang`, `torch>=2.7`,
  `rsl-rl-lib==5.4.0`, `torchrunx` (multi-GPU).
- `utils/lab_api/` is forked from NVIDIA Isaac Lab (BSD-3-Clause).
