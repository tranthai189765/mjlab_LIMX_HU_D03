# HU_D03 — locomotion notes + mimic data plan

Working notes for the LimX HU_D03 task in mjlab. Part 1 records what was learned
and done to get the locomotion (velocity) policy training. Part 2 is the concrete
plan + reading list for the mimic (tracking) dataset, which is NOT done yet.

---

# Part 1 — Locomotion: what was learned / done

End state: robot ported, validated standing, velocity task registered and
training. The non-obvious things that mattered:

## 1.1 URDF -> MJCF conversion (the source has only URDF + USD, no MJCF)
- HU_D03 mesh paths use `package://HU_D03_description/...`; MuJoCo doesn't
  understand `package://`. Strip that prefix and set `meshdir` to the assets dir.
- The URDF has no `<mujoco>` block — inject a `<mujoco><compiler .../></mujoco>`
  with `balanceinertia=true`. Compiles cleanly; all 31 hinge joints come through.
- **Mass/inertia and joint position limits are REAL** (from the URDF): total mass
  51.66 kg; every joint keeps its `<limit lower/upper>`. Joint torque limits are
  in the URDF too (`<limit effort/velocity>`): hip/knee 120, ankle/waist 45,
  shoulder/elbow 30, wrist/hand/head 18 Nm.
- The URDF's `floating_base_joint` does NOT auto-convert to a MuJoCo free joint.
  Add a `freejoint` to `base_link` manually.
- Parallel linkages (achilles A/B at ankles, waist A/B rods) appear only as
  meshes, not joints — modeled as serial (same simplification G1 uses).
- Only 2 bodies are massless: `head_camera_link`, `waist_camera_link` (camera
  frames) — balanceinertia gives them trivial inertia, no real link is distorted.

## 1.2 Collision + standing
- Feet-only collision: all URDF geoms set visual-only; add BOX foot-collision
  geoms `left/right_foot1_collision` on the `*_ankle_roll_link` bodies, plus
  `left/right_foot` sites. (Foot box is an approximation; could be sized from the
  ankle-roll mesh AABB later, or use the URDF's heel/center/tip contact frames.)
- Standing base height is computed by forward kinematics so the feet rest on the
  ground (0.911 m for the official bent-knee pose).

## 1.3 OFFICIAL LimX control data (this replaced earlier guessed gains)
Source repo: `github.com/limxdynamics/humanoid-rl-deploy-python`, folder
`controllers/HU_D03_03/`. Authoritative values used in `hu_d03_constants.py`:
- `stand_controller/joint_params.yaml`: `stand_kp` / `stand_kd` -> actuator
  stiffness/damping; `stand_pos` -> home keyframe (knees bent 0.30).
  Gains (kp/kd): hip_pitch 580/8, hip_roll&yaw 500/6, knee 660/8, ankle 400/2,
  waist 800/8, head 10/1, shoulder_pitch&roll 80/3, shoulder_yaw 80/2,
  elbow 50/3, wrist_yaw 40/2, wrist_pitch 20/1, hand 10/1. (Earlier guesses were
  3-4x too soft — the robot only stands properly with these.)
- `mimic_controller/mimic_param.yaml`: per-joint `action_scale`, `decimation` 10,
  and a higher peak `user_torque_limit` (160/120/80/... Nm) for dynamic motion.
- Only remaining ESTIMATE: `armature` (LimX has not published rotor inertia).
- `humanoid-mujoco-sim` has NO standalone MJCF (prebuilt binary loads the URDF),
  so URDF->MJCF conversion is required; there is no official MJCF to copy.

## 1.4 Velocity task config (mirror of G1) — name remap
File: `tasks/velocity/config/limx_hu_d03/{env_cfgs,rl_cfg,__init__}.py`.
G1 -> HU_D03 name mapping:
- floating base ("pelvis") -> `base_link`
- upper torso ("torso_link") -> `waist_pitch_link` (the body arms/head attach to)
- foot collision geoms -> `left/right_foot1_collision`; foot sites -> `left/right_foot`
- pose-reward std dicts MUST cover every joint -> added `.*head.*` and `.*hand.*`
Tasks auto-register via `import_packages` (no edit to `config/__init__.py` needed).

## 1.5 IMU sensors had to be ADDED to the MJCF
The velocity observations require native MuJoCo sensors `robot/imu_lin_vel`,
`robot/imu_ang_vel`, `robot/root_angmom` (G1 defines these in its XML; the
converted HU_D03 XML did not). Added an `imu_in_base` site on `base_link` plus
`gyro/velocimeter/accelerometer/framezaxis/subtreeangmom` sensors. Done in the
builder so it's reproducible.

## 1.5b PITFALL: action_scale must follow mjlab convention, NOT LimX raw values
First training run collapsed — the robot fell within ~23 control steps and
`track_linear_velocity` stayed ~0 for thousands of iterations. Cause: the
constants used LimX's raw `mimic_param` action_scale (legs 0.25-0.5). mjlab's
action is `target = default_pos + action * scale`; LimX's raw scale is calibrated
to their own deploy pipeline, so it made the leg action authority ~5-10x too large
(per-step target jumps up to ~0.75 rad) and the robot flailed and fell.
FIX: compute `action_scale = 0.25 * effort_limit / stiffness` per joint (the G1
formula). Legs become ~0.03-0.06. Use LimX's official kp/kd/pose/effort, but NOT
its action_scale. Symptom to watch in the log: `Mean episode length` tiny and
`track_linear_velocity` not rising.

## 1.5c PITFALL: use LimX MIMIC gains for locomotion, NOT STAND gains
Second collapse mode: the robot stayed upright but would NOT walk — with a forced
0.6 m/s forward command the base moved 0.004 m/s. Root cause: the constants used
LimX's stand_controller gains (kp knee 660, waist 800) which are a stiff standing
hold. Because action_scale = 0.25*effort/stiffness, stiff gains shrink the action
scale to ~10x below G1 (knee 0.045 vs G1 0.351, ankle 0.028 vs 0.439) -> the policy
can only move the knee ~±5deg, far too little to step.
FIX: use LimX's mimic_controller gains (mimic_param.yaml control.kp/kd — the
DYNAMIC-motion gains): hip_pitch/roll 280, hip_yaw 100, knee 280, ankle 20,
waist 80, head 5, shoulder/elbow 80, wrist/hand 40 (kd ~2-5). Action scale then
becomes G1-comparable (knee 0.107, ankle 0.562, hip_yaw 0.30). Rule: STAND gains
for a static hold; MIMIC gains for any RL locomotion. Diagnose objectively by
rendering with a forced forward command and printing base dx + mean forward vel,
not by trusting the average track_linear_velocity reward (it is inflated by the
standing-commanded envs).

## 1.5d PITFALL: pkill self-match when restarting training
`pkill -f "Mjlab-Velocity-..."` or `pkill -f "uv run train"` matches the shell that
is running the command (its own cmdline contains the pattern), so it kills itself,
ssh returns 255, and the restart never launches. Kill the real process with
`pkill -f "bin/train"` (the python is `.venv/bin/train ...`) or by PID from
`ps -eo pid,cmd | grep "[b]in/train"`. Confirm with nvidia-smi (the training python
shows up as a GPU compute app using ~1.5 GB; exactly one should be running).

## 1.6 Headless server gotchas
- `MUJOCO_GL=disable` is REQUIRED for every command (no GL libs on the box;
  mjlab forces `MUJOCO_GL=egl` which crashes). Rendering (play/csv_to_npz
  --render) needs EGL system libs (sudo) — not installed yet.
- Use `--agent.logger tensorboard` (or `WANDB_MODE=disabled`) to avoid the wandb
  login prompt during training.

## 1.7 Rebuild recipe (server)
```bash
cd ~/ml/mjlab && export PATH="$HOME/.local/bin:$PATH"
MUJOCO_GL=disable uv run python ~/ml/stand_hu_d03.py     # build curated xml (+IMU) + stand test
uv run python ~/ml/package_hu_d03.py                     # copy meshes + write package xml
MUJOCO_GL=disable uv run python ~/ml/validate_hu_d03.py  # mjlab Entity stands check
MUJOCO_GL=disable uv run train Mjlab-Velocity-Flat-LimX-HU-D03 --env.scene.num-envs 4096 --agent.logger tensorboard
```

---

# Part 2 — Mimic (tracking): dataset plan — NOT done yet

## 2.0 Scope — what the mimic policy reproduces
Mimic (tracking) is NOT the same as locomotion. Locomotion follows a velocity
*command* (reactive, any direction/speed). Mimic reproduces a specific recorded
*reference motion* — the whole-body pose over time (legs, torso, ARMS, timing).

Convention (mjlab / BeyondMimic / LimX): one tracking policy tracks ONE reference
motion clip. task.txt requires "1 mimic policy", so the minimal deliverable is a
single policy that imitates one clip. The "tasks" the policy performs = whatever
motion is in that clip.

Candidate motions (from LAFAN1): walk, run, sprint; turn / sidestep / spin;
jump / hop; dance (rich arm+torso motion); crawl; get-up after a fall; kick /
fight. Picking a clip decides the skill (e.g. a dance clip -> learns that dance
including arm swing and turns).

Scope options:
  - PLAN A (recommended): one moderately dynamic clip (e.g. walk+turn+stop, or a
    short dance) — enough to demonstrate whole-body imitation, high chance of
    converging. Matches LimX's own setup (their mimic_controller policy.onnx
    imitates a single 764-frame clip).
  - PLAN B (more ambitious): several clips -> several policies (walk, dance,
    jump). More work, more impressive.
Decision still pending: which clip(s) / skill the user wants.

## 2.1 Why we still need to build data
LimX ships NO reusable reference motion. Their `mimic_controller.py` feeds the
policy a scalar `motion_phase = motion_iter / 764` — the reference motion is baked
into `policy.onnx`, not stored as data. There are no csv/npz/bvh motion files in
their repos. So we must produce HU_D03 reference motion ourselves.

mjlab's tracking task needs a reference motion CSV in "Unitree generalized
coordinate convention" (per frame: base_pos(3) + base_quat xyzw(4) + 31 joint
angles in a fixed order), converted to NPZ with `mjlab.scripts.csv_to_npz` and
hosted on a WandB registry.

## 2.2 Canonical joint order (from LimX yaml — reuse everywhere)
legs L(6): hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll;
legs R(6): same; waist(3): yaw, roll, pitch; head(2): yaw, pitch;
arms L(7): shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_yaw,
wrist_pitch, hand_yaw; arms R(7): same. (Total 31.)

## 2.3 Pipeline: LAFAN1 -> GMR -> CSV -> csv_to_npz -> WandB -> train
1. **Source motion**: download LAFAN1 (Ubisoft mocap, BVH). Clean, agile,
   the dataset BeyondMimic/the csv_to_npz lineage uses.
2. **Retarget with GMR** (General Motion Retargeting): retargets AMASS/LAFAN1 ->
   humanoid robots from URDF on CPU. Supports G1/H1 out of the box; **HU_D03 must
   be added as a new robot config** (its URDF + a human->robot link/joint
   correspondence map). This is the real work of mimic data prep.
3. **Export CSV** in Unitree convention with the joint order in 2.2.
4. **Adapt `src/mjlab/scripts/csv_to_npz.py`** for HU_D03: it is hardcoded to G1
   (builds the G1 tracking scene + lists 29 G1 joints). Make a HU_D03 variant that
   uses a HU_D03 tracking env cfg and the 31-joint order above. Keep mjlab's
   converter (NPZ stores body pose by MuJoCo depth-first body index; a non-mjlab
   NPZ will mismap bodies and training won't converge).
   ```bash
   MUJOCO_GL=egl uv run -m mjlab.scripts.csv_to_npz \
     --input-file <retargeted.csv> --output-name <name> \
     --input-fps 30 --output-fps 50 --render True
   ```
   (csv_to_npz with --render needs EGL — install GL libs or run --render False.)
5. **WandB**: the script uploads the NPZ to your `motions` registry.
6. **Tracking task config** for HU_D03: mirror `tasks/tracking/config/g1/`,
   remap `motion_cmd.anchor_body_name` -> `waist_pitch_link`, `body_names` ->
   HU_D03 bodies (base, hips, knees, ankles, torso, shoulders, elbows, wrists),
   `ee_body_pos` -> ankle/wrist links. Register `Mjlab-Tracking-Flat-LimX-HU-D03`.
7. **Train**:
   ```bash
   uv run train Mjlab-Tracking-Flat-LimX-HU-D03 --registry-name <org>/motions/<name> --env.scene.num-envs 4096
   ```

## 2.4 Alternative (shortcut, optional)
Roll out LimX's pre-trained `mimic_controller/policy/default/policy.onnx` in
MuJoCo and RECORD the joint-angle + base-pose trajectory (764 frames) as the
reference CSV. Gives a HU_D03-native motion without GMR or external datasets, but
only one clip and it "imitates a policy's output." Needs replicating their obs
(lin_encoder/priv_encoder + phase). Use only if GMR setup is too costly.

## 2.5 Reading list / links
- GMR (General Motion Retargeting), the retargeter: https://github.com/YanjieZe/GMR
- GMR paper "Retargeting Matters": https://arxiv.org/abs/2510.02252
- BeyondMimic / whole_body_tracking (mjlab's csv_to_npz lineage, registry setup):
  https://github.com/HybridRobotics/whole_body_tracking
- LAFAN1 retargeting dataset (HuggingFace, G1/H1 examples of the CSV format):
  https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset
- mjlab motion-imitation docs (CSV convention, registry, training):
  in-repo `docs/source/training/motion_imitation.rst` ;
  online https://mujocolab.github.io/mjlab/
- LimX humanoid RL deploy (mimic controller, joint order, action_scale, the
  pre-trained policy for option 2.4):
  https://github.com/limxdynamics/humanoid-rl-deploy-python
- LimX `gradmotion-cli` (unverified — name suggests a motion tool; worth checking):
  https://github.com/limxdynamics/gradmotion-cli
- mjlab tracking task to mirror: `src/mjlab/tasks/tracking/config/g1/` and
  `src/mjlab/scripts/csv_to_npz.py`.

## 2.6 Open blockers before mimic training
- GMR has no HU_D03 config — must be written.
- `csv_to_npz.py` is G1-specific — needs a HU_D03 variant.
- Rendering path (csv_to_npz --render, play) needs EGL system libs on the server.
- A WandB login on the server (have an account; key not yet on the box).
