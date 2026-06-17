# LimX HU_D03 robot facts (verified from URDF)

Source: `humanoid-description/HU_D03_description/` (from
`github.com/limxdynamics/humanoid-description`). The variant is **HU_D03_03**.
Assets provided: URDF (`urdf/HU_D03_03.urdf`), STL meshes
(`meshes/HU_D03_03/`), USD (`usd/`). **No MJCF** — conversion to MuJoCo MJCF is
required for mjlab.

## Kinematic structure

- `root_link` = virtual world anchor. `floating_base_joint` (type `floating`)
  connects `root_link` -> **`base_link`**. In MJCF, put the `freejoint` on
  `base_link`; that is the floating base body.
- **31 revolute (actuated) joints**, **12 fixed**, **1 floating**.

### The 31 actuated joints

Legs (12):
- `left_hip_pitch_joint`, `left_hip_roll_joint`, `left_hip_yaw_joint`,
  `left_knee_joint`, `left_ankle_pitch_joint`, `left_ankle_roll_joint`
- `right_*` mirror (same six on the right)

Waist (3): `waist_yaw_joint`, `waist_roll_joint`, `waist_pitch_joint`

Arms (14):
- `left_shoulder_pitch_joint`, `left_shoulder_roll_joint`,
  `left_shoulder_yaw_joint`, `left_elbow_joint`, `left_wrist_pitch_joint`,
  `left_wrist_yaw_joint`, `left_hand_yaw_joint`
- `right_*` mirror (same seven on the right)

Head (2): `head_yaw_joint`, `head_pitch_joint`

> Note vs Unitree G1: HU_D03 **adds** head (yaw/pitch) and `*_hand_yaw`; its wrist
> has pitch+yaw (no wrist_roll). Account for these when copying G1 actuator
> groups and pose-reward regexes.

## Feet

- Foot bodies: **`left_ankle_roll_link`**, **`right_ankle_roll_link`** (the link
  the foot mesh attaches to). Put foot collision geoms + the `left_foot`/
  `right_foot` sites on these.

## Parallel linkages (important for MJCF + actuators)

The URDF/mesh set includes closed-chain mechanisms that are NOT exposed as
revolute joints:
- **Achilles rods** at each ankle: `left/right_A_achilles*`, `left/right_B_achilles*`.
- **Waist rods**: `waist_A_*`, `waist_B_*`.

URDF cannot represent kinematic loops, so these show up as fixed/mesh only. For a
first working model, **do not model the closed loop** — use a serial chain and
approximate the effective actuator armature at the ankle/waist joints by lumping
the two driving actuators (exactly how `g1_constants.py` handles G1's 4-bar
ankle/waist linkages with `G1_ACTUATOR_ANKLE` / `G1_ACTUATOR_WAIST`).

## What is NOT in the URDF (you must supply / decide)

- Motor specs (gear ratios, rotor inertia, torque/velocity limits) — get from LimX
  datasheets; otherwise start from G1-scaled values and mark as assumptions.
- A standing keyframe (joint angles + base height). Derive from a stable pose;
  start near zero with slightly bent knees and tune until the robot stands under
  zero action.
- Collision geometry suitable for sim (capsules for feet, simplified self-collision
  set) — mirror G1's `*_collision` naming so task configs match.

## Official LimX control data (USE THESE — not guesses)

LimX publishes per-robot controllers in `github.com/limxdynamics/humanoid-rl-deploy-python`
under `controllers/HU_D03_03/`. These are authoritative:

- `stand_controller/joint_params.yaml`: `stand_pos` (the standing pose) and
  `stand_kp` / `stand_kd` (PD gains). Used directly as the mjlab actuator
  stiffness/damping and the home keyframe. Joint order in the yaml is
  legs L(6) R(6), waist(3), head(2), arms L(7) R(7) — arm order is
  shoulder_pitch, _roll, _yaw, elbow, wrist_yaw, wrist_pitch, hand_yaw.
  Official gains (kp/kd): hip_pitch 580/8, hip_roll&yaw 500/6, knee 660/8,
  ankle 400/2, waist 800/8, head 10/1, shoulder_pitch&roll 80/3,
  shoulder_yaw 80/2, elbow 50/3, wrist_yaw 40/2, wrist_pitch 20/1, hand 10/1.
  Stand pose: hip_pitch -0.15, hip_yaw -+0.05, knee 0.30, ankle_pitch -0.16,
  shoulder_pitch 0.1, shoulder_roll -+0.1, shoulder_yaw +-0.2, elbow -0.2.
- `mimic_controller/mimic_param.yaml`: `control.action_scale` (per-joint),
  `decimation` (10), `user_torque_limit` (peak torque, higher than URDF
  nameplate), and `policy/default/*.onnx` (a pre-trained mimic policy +
  encoders — a useful reference baseline for the mimic task).
- Torque (effort) limits: URDF `<limit effort>` is the conservative nameplate
  (hip/knee 120, ankle/waist 45, shoulder/elbow 30, wrist/hand/head 18). The
  mimic yaml's `user_torque_limit` is higher (160/120/80/3/80/10) for dynamic
  motion. Joint POSITION limits are in the URDF and survive URDF->MJCF.
- `humanoid-mujoco-sim` has NO standalone MJCF (a prebuilt binary loads the
  URDF), so URDF->MJCF conversion is still required; there is no official MJCF
  to copy.

The converted model is faithful: mass 51.66 kg and all 31 joint position limits
come straight from the URDF. Only two bodies are massless (head/waist camera
frames). Remaining approximations: box foot-collision geometry, small estimated
`armature` (no published rotor inertia), and parallel linkages modeled as serial.

## Reference files to mirror (G1 is the template)

- Robot: `src/mjlab/asset_zoo/robots/unitree_g1/{g1_constants.py, xmls/g1.xml}`
- Velocity task: `src/mjlab/tasks/velocity/config/g1/{env_cfgs.py, rl_cfg.py, __init__.py}`
- Tracking task: `src/mjlab/tasks/tracking/config/g1/{env_cfgs.py, rl_cfg.py, __init__.py}`
- Motion preprocessing (hardcoded to G1 — needs a HU_D03 variant):
  `src/mjlab/scripts/csv_to_npz.py`
