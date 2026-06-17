---
name: hu-d03-mjlab
description: >
  End-to-end plan to bring the LimX HU_D03 humanoid into mjlab and train two
  policies: a locomotion (velocity-tracking) policy and a motion-imitation
  (mimic/tracking) policy. Use this when working in the mjlab repo on a
  GPU server to (1) port the HU_D03 robot from URDF to MuJoCo MJCF, (2) wire
  up velocity and tracking tasks for it, and (3) build the motion-data pipeline
  (LAFAN1 -> GMR retarget -> csv_to_npz -> WandB). Read references/ for the
  codebase architecture and the verified HU_D03 robot facts before editing.
---

# Port & train LimX HU_D03 in mjlab

This skill is a self-contained execution plan. The goal is two deliverables:

1. **Explanation** of the mjlab codebase (structure, modules, how it works, the
   training algorithm). The full writeup is in
   [references/architecture.md](references/architecture.md) — reuse/extend it.
2. **Train a locomotion policy AND a mimic policy** for the **LimX HU_D03**
   humanoid (`limxdynamics/humanoid-description`, `HU_D03_description`),
   following mjlab's existing Unitree G1 tasks as the template.

> Read [references/architecture.md](references/architecture.md) and
> [references/hu_d03_facts.md](references/hu_d03_facts.md) first. They contain
> verified facts (joint list, base link, parallel linkages) and the mental model
> you need. Do not skip them.

## Ground rules

- **Always use `uv run`, never bare `python`.** See the repo's `CLAUDE.md`.
- Run `make check` (format + lint + type) before committing. Don't commit code
  that fails type checking.
- **Training requires an NVIDIA GPU + CUDA** (MuJoCo Warp). macOS/CPU is
  eval-only. Confirm `nvidia-smi` works and `uv run python -c "import torch; print(torch.cuda.is_available())"` prints `True` before training.
- Mirror the **Unitree G1** implementation everywhere. G1 is the reference robot;
  copy its file layout and adapt names/numbers. Never invent APIs — grep G1 usage.
- The hard part is **NOT the RL**. It is (a) a correct URDF->MJCF conversion and
  actuator tuning so the robot stands, and (b) motion retargeting for mimic.
  Budget effort accordingly and verify at every gate.

## HU_D03 quick facts (see references for detail)

- **31 revolute (actuated) joints**: legs 12 (hip pitch/roll/yaw, knee, ankle
  pitch/roll ×2), waist 3 (yaw/roll/pitch), arms 14 (shoulder pitch/roll/yaw,
  elbow, wrist pitch/yaw, hand_yaw ×2), head 2 (yaw/pitch).
- Floating base body = **`base_link`** (URDF `root_link` is a virtual world
  anchor; `floating_base_joint` connects root_link -> base_link).
- Feet = **`left_ankle_roll_link` / `right_ankle_roll_link`**.
- **Parallel linkages** (achilles A/B at the ankles, waist A/B) appear as meshes
  but NOT as revolute joints in the URDF. Treat them the way G1 approximates its
  4-bar ankle/waist linkages in `g1_constants.py` (lump the armature). Do not try
  to model closed loops first; get a serial-chain model standing first.
- Source robot assets: `humanoid-description/HU_D03_description/` (URDF + meshes
  STL + USD). mjlab needs **MJCF**, so conversion is mandatory.

---

## Phase 0 — Environment sanity check

```bash
cd mjlab
uv run demo                       # confirms install + GPU rendering
uv run list-envs                  # see registered task ids
uv run train Mjlab-Velocity-Flat-Unitree-G1 --env.scene.num-envs 4096  # ~smoke test, can Ctrl-C early
```

If the G1 baseline does not start training, fix the environment before touching
HU_D03. Verify CUDA as noted in ground rules.

---

## Phase 1 — Port the HU_D03 robot into the asset zoo

Target layout (mirror `src/mjlab/asset_zoo/robots/unitree_g1/`):

```
src/mjlab/asset_zoo/robots/limx_hu_d03/
  __init__.py
  hu_d03_constants.py
  xmls/
    hu_d03.xml          # MJCF
    assets/             # copied STL meshes
```

### 1.1 Convert URDF -> MJCF

- Copy meshes from `humanoid-description/HU_D03_description/meshes/HU_D03_03/`
  into `xmls/assets/`.
- Convert `HU_D03_03.urdf` to MJCF. Options, in order of preference:
  - Load URDF with `mujoco.MjSpec` / `mujoco.MjModel.from_xml_path` after adding a
    `<mujoco><compiler ... /></mujoco>` block, then save MJCF; or
  - Use MuJoCo's URDF importer and hand-edit.
- Then edit the MJCF to add what URDF can't express (study `g1.xml` side by side):
  - `<compiler angle="radian" meshdir="assets" autolimits="true"/>`.
  - A `freejoint` on `base_link`.
  - **Default classes** `visual` (group 2, no collision) and `collision`
    (group 3) like G1.
  - **Collision geoms**: at minimum capsule foot collisions named
    `left_foot[1-7]_collision` / `right_foot[1-7]_collision` (G1 uses this naming
    and the task configs depend on it). Add condim=3 for feet.
  - **Sites** `left_foot` and `right_foot` under the ankle-roll bodies (tasks
    reference these site names).
  - A **home keyframe** (standing, knees slightly bent).
  - Sensible inertias (the URDF should carry them; verify none are degenerate).

### 1.2 Write `hu_d03_constants.py`

Copy `unitree_g1/g1_constants.py` and adapt. Required exports (match G1's names
so task code ports cleanly):

- `get_spec()` -> loads the MJCF.
- Actuator configs (`BuiltinPositionActuatorCfg`) grouped by motor type, with
  `target_names_expr` regexes over HU_D03 joint names, plus `stiffness`,
  `damping`, `effort_limit`, `armature`.
  - **Motor specs**: get HU_D03 actuator gear ratios / rotor inertias / torque
    limits from LimX datasheets if available. If unavailable, start from
    physically reasonable values scaled from G1 by joint, and **flag them as
    assumptions in a comment**. Bad actuator params are the #1 reason the robot
    won't stand.
  - For the **achilles (ankle) and waist parallel linkages**, lump armature like
    G1's `G1_ACTUATOR_ANKLE` / `G1_ACTUATOR_WAIST` (sum the two actuators'
    armature; nominal 1:1 ratio).
  - Add actuator groups for the **extra DOFs G1 lacks**: `head_yaw/pitch` and
    `.*_hand_yaw` (and wrist_pitch/yaw). Low stiffness is fine for head/hands.
- Keyframes: `HOME_KEYFRAME` and a `KNEES_BENT_KEYFRAME` (`EntityCfg.InitialStateCfg`).
- Collision configs: `FULL_COLLISION`, `FULL_COLLISION_WITHOUT_SELF`,
  `FEET_ONLY_COLLISION` (copy structure, fix geom-name regexes).
- `HU_D03_ARTICULATION = EntityArticulationInfoCfg(actuators=(...), soft_joint_pos_limit_factor=0.9)`.
- `get_hu_d03_robot_cfg() -> EntityCfg`.
- `HU_D03_ACTION_SCALE: dict[str, float]` built from the actuators (copy G1's loop:
  `0.25 * effort_limit / stiffness` per joint).
- Keep the `if __name__ == "__main__":` viewer block to eyeball the model.

### 1.3 Register the robot

Add exports in `src/mjlab/asset_zoo/robots/__init__.py` (mirror the G1 lines:
`get_hu_d03_robot_cfg`, `HU_D03_ACTION_SCALE`).

### 1.4 GATE — verify the robot before any training

```bash
uv run python -m mjlab.asset_zoo.robots.limx_hu_d03.hu_d03_constants   # opens viewer
```

Checklist: model compiles; meshes load; robot stands under gravity with **zero
action** from the keyframe without exploding, sinking, or jittering; joint limits
look sane; feet collide with the floor; no NaNs. Iterate on inertia / actuator /
keyframe until this holds. **Do not proceed until the robot stands.**

---

## Phase 2 — Locomotion policy (velocity task) — NO external data

Target layout (mirror `src/mjlab/tasks/velocity/config/g1/`):

```
src/mjlab/tasks/velocity/config/limx_hu_d03/
  __init__.py        # register_mjlab_task(...)
  env_cfgs.py        # adapt body/site/regex names to HU_D03
  rl_cfg.py          # PPO cfg (start from G1's, tweak experiment_name)
```

### 2.1 `env_cfgs.py`

Copy G1's `env_cfgs.py` and replace every G1-specific reference:
- `get_g1_robot_cfg` / `G1_ACTION_SCALE` -> HU_D03 equivalents.
- Body names: `torso_link` -> HU_D03 torso body (likely a waist/torso link;
  pick the upper-torso body; `cfg.viewer.body_name`, `base_com`, `upright`,
  `body_ang_vel` all reference it).
- `pelvis` raycast frame -> HU_D03 base body (`base_link`).
- Foot sites `left_foot`/`right_foot` and foot collision geom regex
  `(left|right)_foot[1-7]_collision` must match the names you created in 1.1.
- Self-collision sensor `primary/secondary` subtree pattern -> HU_D03 base body.
- `rewards["pose"]` std dicts: keep the same regex keys (`.*hip_pitch.*`, etc.)
  and **add entries for head and hand joints** so they're regularized.

### 2.2 `rl_cfg.py` + `__init__.py`

- `rl_cfg.py`: copy G1's, set `experiment_name="hu_d03_velocity"`.
- `__init__.py`: `register_mjlab_task` for
  `Mjlab-Velocity-Flat-LimX-HU-D03` (flat) and optionally `-Rough-`.
  Make sure this `__init__` is imported so registration runs (check how G1's
  config package is imported under `tasks/velocity/config/__init__.py`).

### 2.3 Sanity-check the MDP, then train

```bash
uv run play Mjlab-Velocity-Flat-LimX-HU-D03 --agent zero     # robot should just stand
uv run play Mjlab-Velocity-Flat-LimX-HU-D03 --agent random   # should flail but not NaN
uv run train Mjlab-Velocity-Flat-LimX-HU-D03 --env.scene.num-envs 4096
uv run play Mjlab-Velocity-Flat-LimX-HU-D03 --wandb-run-path <org>/mjlab/<run-id>
```

GATE: reward should climb and the robot should learn to track commanded
velocities. If it never stands, revisit Phase 1 (actuators/inertia) — not the RL.

---

## Phase 3 — Mimic policy (tracking task) — needs motion DATA

mjlab ships NO motion data and NO retargeter. You must produce reference motion
**retargeted to HU_D03's exact 31-joint skeleton**. WandB is the documented host
(you have an account).

### 3.1 Build the motion data (the long pole)

Pipeline: **LAFAN1 (.bvh) -> GMR retarget to HU_D03 -> CSV (Unitree convention)**.

- **GMR** = General Motion Retargeting (`github.com/YanjieZe/GMR`). Retargets
  AMASS/LAFAN1/OMOMO -> humanoid robots from URDF, on CPU.
- GMR supports Unitree G1/H1 out of the box but **NOT HU_D03** -> you must add a
  HU_D03 robot config to GMR (its URDF + a human->robot link/joint correspondence
  map). This is the real work of mimic data prep.
- Start with **1-2 short LAFAN1 clips** (e.g. a walk) end-to-end before scaling up.
- Output CSV must be **Unitree generalized-coordinate convention**, one row per
  frame: `[base_pos(3)] [base_quat xyzw(4)] [31 joint angles in HU_D03 order]`.
  The joint order must match the list you pass to `csv_to_npz` (Phase 3.2).
- Validate retargeting visually: no foot sliding, no ground penetration, no
  self-intersection.

### 3.2 Adapt `csv_to_npz.py` for HU_D03

`src/mjlab/scripts/csv_to_npz.py` is hardcoded to G1 (it builds the G1 tracking
scene and lists 29 G1 joint names). Create a HU_D03 variant (or parametrize):
- Use `hu_d03_flat_tracking_env_cfg()` for the scene (from Phase 3.3).
- Replace the `joint_names=[...]` list with HU_D03's 31 joints **in the exact
  order your retargeted CSV uses**.
- Keep mjlab's converter — it stores body pos/quat indexed by **MuJoCo body order
  (depth-first)**. Do not substitute an IsaacLab/other NPZ or bodies will be
  mismapped and training won't converge.

```bash
MUJOCO_GL=egl uv run -m mjlab.scripts.csv_to_npz \
  --input-file <retargeted.csv> --output-name <motion_name> \
  --input-fps 30 --output-fps 50 --render True
```

This plays the motion, computes FK for all bodies, writes the NPZ, and uploads it
to your WandB `motions` registry.

### 3.3 Tracking task config

Target layout (mirror `src/mjlab/tasks/tracking/config/g1/`):

```
src/mjlab/tasks/tracking/config/limx_hu_d03/
  __init__.py   # register Mjlab-Tracking-Flat-LimX-HU-D03 (+ -No-State-Estimation)
  env_cfgs.py
  rl_cfg.py
```

In `env_cfgs.py` (copy G1's tracking cfg) set:
- `motion_cmd.anchor_body_name` -> HU_D03 torso body.
- `motion_cmd.body_names` -> the HU_D03 bodies to track (base, hips, knees,
  ankles, torso, shoulders, elbows, wrists — pick the HU_D03 analogues of the G1
  list).
- `terminations["ee_body_pos"].body_names` -> HU_D03 ankle/wrist links.
- foot friction geom regex + self-collision subtree -> HU_D03 names.
- viewer body -> HU_D03 torso.

### 3.4 Train + eval

```bash
uv run train Mjlab-Tracking-Flat-LimX-HU-D03 \
  --registry-name <org>/motions/<motion_name> --env.scene.num-envs 4096
uv run play Mjlab-Tracking-Flat-LimX-HU-D03 --wandb-run-path <org>/mjlab/<run-id>
```

GATE: the robot should follow the reference motion. If it diverges immediately,
suspect (a) joint-order mismatch between CSV and `csv_to_npz`, or (b) body-name
mismatch in `motion_cmd.body_names`.

---

## Common pitfalls

- **Joint order mismatch** (CSV vs `csv_to_npz` vs MJCF) — silent, deadly for
  mimic. Pin one canonical order and reuse it everywhere.
- **Foot geom / site names** not matching task-config regexes — locomotion
  rewards/sensors silently do nothing. Keep G1's naming scheme.
- **Wrong/zero inertia or bad actuator gains** — robot can't stand; no amount of
  RL fixes it. Fix in Phase 1.
- **Using a non-mjlab NPZ** — body index convention differs (MuJoCo DFS vs PhysX
  BFS); tracking won't converge.
- **Registration not imported** — task id won't show in `uv run list-envs`.
  Ensure the new `config/<robot>/__init__.py` is imported by its package.
- **Parallel linkages** — don't model closed loops up front; serial approximation
  with lumped armature (G1-style) first.

## Definition of done

- `uv run list-envs` shows `Mjlab-Velocity-Flat-LimX-HU-D03` and
  `Mjlab-Tracking-Flat-LimX-HU-D03`.
- A trained locomotion checkpoint that tracks velocity commands (play looks good).
- A trained mimic checkpoint that imitates at least one retargeted LAFAN1 clip.
- `make check` passes. Short writeup of the architecture (deliverable 1) is ready
  — see [references/architecture.md](references/architecture.md).
