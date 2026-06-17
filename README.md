# mjlab — LimX HU_D03 Integration

Integrating the **LimX HU_D03** humanoid (31 DOF, ≈51.7 kg) into the
[mjlab](https://github.com/mujocolab/mjlab) GPU-accelerated RL framework, and
training three control policies with PPO:

| Policy | Task ID | Result |
| --- | --- | --- |
| **Locomotion** | `Mjlab-Velocity-Flat-LimX-HU-D03` | Forward velocity tracking **87–99 %** (0.5–2.0 m/s), **0 % fall rate** |
| **Mimic — Dance** | `Mjlab-Tracking-Flat-LimX-HU-D03` | Completes **497/500** steps/episode (99.4 %), reward 26.35 |
| **Mimic — Jumps** | `Mjlab-Tracking-Flat-LimX-HU-D03` | Completes **488/500** (97.6 %), reward 29.90 |

All policies were trained on a single NVIDIA RTX 3060 (12 GB) using 4096 parallel
MuJoCo Warp environments.

> This repository is built on top of **mjlab** (Apache-2.0); see
> `README_upstream.md` for the original project. The motion-imitation data
> pipeline additionally uses [GMR](https://github.com/YanjieZe/GMR),
> [LAFAN1](https://github.com/ubisoft/ubisoft-laforge-animation-dataset), and the
> [BeyondMimic](https://beyondmimic.github.io/) methodology.

---

## Demo videos

In [`video_checkpoint/`](video_checkpoint/):

| Locomotion | Mimic |
| --- | --- |
| `locomotion_walk_0.6ms.mp4` — walk at 0.6 m/s | `mimic_dance.mp4` — LAFAN1 dance |
| `locomotion_run_1.5ms.mp4` — run at 1.5 m/s | `mimic_jumps.mp4` — LAFAN1 jumps |
| `locomotion_run_2.0ms.mp4` — run at 2.0 m/s | `reference_*_retarget.mp4` — retarget references |

---

## What was added on top of mjlab

```
src/mjlab/asset_zoo/robots/limx_hu_d03/      # robot package
    hu_d03_constants.py                      #   actuators (official LimX gains), keyframe, collisions
    xmls/hu_d03.xml  +  xmls/assets/*.STL    #   MJCF converted from the URDF
src/mjlab/tasks/velocity/config/limx_hu_d03/ # locomotion task config
src/mjlab/tasks/tracking/config/limx_hu_d03/ # mimic (tracking) task config
src/mjlab/scripts/csv_to_npz_hu_d03.py       # motion CSV -> NPZ for HU_D03 (31 DOF)
tools/                                       # helper scripts (convert / retarget / eval / render / plot)
motions/                                     # retargeted reference motions (.npz)
checkpoints/                                 # trained policies (.pt)
video_checkpoint/  +  figures/               # demo clips + training curves
```

The robot's mass, inertia, and joint limits come directly from the official LimX
URDF; the actuator PD gains and standing pose come from LimX's official
`humanoid-rl-deploy-python` controllers.

---

## Setup

mjlab requires an **NVIDIA GPU + CUDA**. Dependencies are managed with
[`uv`](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/tranthai189765/mjlab_LIMX_HU_D03.git
cd mjlab_LIMX_HU_D03
uv sync                       # creates .venv and installs everything
uv run list-envs              # should list Mjlab-*-LimX-HU-D03 tasks
```

> **Headless servers:** mjlab forces `MUJOCO_GL=egl`, which needs GL libraries.
> If the machine has no display, prefix training/eval commands with
> `MUJOCO_GL=disable`. For offscreen video rendering install Mesa EGL
> (`sudo apt-get install -y libegl1 libegl-mesa0 libgl1-mesa-dri libosmesa6`).

---

## Training

**Locomotion (no external data needed):**
```bash
MUJOCO_GL=disable uv run train Mjlab-Velocity-Flat-LimX-HU-D03 \
    --env.scene.num-envs 4096 --agent.logger tensorboard
```

**Mimic (uses a reference motion `.npz`):**
```bash
MUJOCO_GL=disable uv run train Mjlab-Tracking-Flat-LimX-HU-D03 \
    --env.commands.motion.motion-file motions/hu_d03_dance.npz \
    --env.scene.num-envs 4096 --agent.logger tensorboard
# or motions/hu_d03_jump.npz
```

Checkpoints are written to `logs/rsl_rl/<experiment>/<run>/model_<iter>.pt` every
`save_interval` iterations.

---

## Loading a checkpoint & evaluation

Pre-trained checkpoints are in [`checkpoints/`](checkpoints/):
`locomotion/model_14450.pt`, `dance/model_20000.pt`, `jumps/model_29999.pt`.

**Quantitative locomotion evaluation** (mean achieved velocity, tracking %, fall
rate across 256 parallel envs):
```bash
MUJOCO_GL=disable uv run python tools/eval_locomotion_hu_d03.py
# edit CKPT inside the script to point at checkpoints/locomotion/model_14450.pt
```

**Render a policy to mp4** (headless, needs Mesa EGL):
```bash
# locomotion (forced forward command, prints base displacement + velocity)
MUJOCO_GL=egl uv run python tools/render_clip_hu_d03.py
# mimic (follows a reference motion)
MUJOCO_GL=egl uv run python tools/render_mimic_hu_d03.py \
    --motion motions/hu_d03_dance.npz --out mimic.mp4
```

**Interactive viewer** (requires a local display):
```bash
uv run play Mjlab-Velocity-Flat-LimX-HU-D03 --checkpoint-file checkpoints/locomotion/model_14450.pt
```

**Sanity-check the MDP** before training:
```bash
uv run play Mjlab-Velocity-Flat-LimX-HU-D03 --agent zero    # zero actions
uv run play Mjlab-Velocity-Flat-LimX-HU-D03 --agent random  # random actions
```

---

## Mimic data pipeline (LAFAN1 → GMR → NPZ)

mjlab ships no motion data, so reference motions are produced by retargeting human
mocap onto HU_D03 (same convention as BeyondMimic):

1. **LAFAN1** — download the Ubisoft mocap dataset and extract a BVH clip
   (e.g. `dance1_subject2.bvh`).
2. **GMR retargeting** — HU_D03 is registered into
   [GMR](https://github.com/YanjieZe/GMR) by adding its MJCF asset and the IK
   config `tools/bvh_lafan1_to_hu_d03.json`. Retarget headless with
   `tools/retarget_hu_d03.py` → a GMR pickle → CSV.
3. **NPZ** — `csv_to_npz_hu_d03` replays the CSV through MuJoCo (forward
   kinematics) to produce the `.npz` reference consumed by the tracking task.

Every retargeted clip is rendered (`tools/render_motion_csv.py`) and visually
inspected before training.

---

## Results

| Command | Target | Achieved | Tracking | Fall rate |
| --- | ---: | ---: | ---: | ---: |
| Forward 0.5 m/s | 0.50 | 0.435 | 87 % | 0 % |
| Forward 1.0 m/s | 1.00 | 0.991 | 99 % | 0 % |
| Forward 1.5 m/s | 1.50 | 1.482 | 99 % | 0 % |
| Forward 2.0 m/s | 2.00 | 1.973 | 99 % | 0 % |

Training curves are in [`figures/`](figures/) (`fig_locomotion_training.png`,
`fig_mimic_training.png`).

---

## Credits

- **mjlab** — Zakka et al. (Apache-2.0). The base framework.
- **GMR** — Ze et al., *Retargeting Matters*, arXiv:2510.02252.
- **LAFAN1** — Harvey et al., *Robust Motion In-betweening*, SIGGRAPH 2020 (Ubisoft).
- **BeyondMimic** — arXiv:2508.08241; the tracking task re-implements its method.
- **LimX HU_D03** — robot description and control parameters from
  [limxdynamics](https://github.com/limxdynamics).
