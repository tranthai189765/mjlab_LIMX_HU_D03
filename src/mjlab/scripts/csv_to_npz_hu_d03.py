"""csv_to_npz for the LimX HU_D03 (31-DoF) robot.

Same machinery as mjlab.scripts.csv_to_npz but builds the HU_D03 tracking scene
and uses HU_D03's 31 joints in MJCF order. The input CSV must follow the Unitree
generalized-coordinate convention: per frame [base_pos(3), base_quat xyzw(4),
31 joint angles in the order below].
"""

import tyro

from mjlab.scene import Scene
from mjlab.scripts.csv_to_npz import run_sim
from mjlab.sim.sim import Simulation, SimulationCfg
from mjlab.tasks.tracking.config.limx_hu_d03.env_cfgs import (
  hu_d03_flat_tracking_env_cfg,
)
from mjlab.viewer.offscreen_renderer import OffscreenRenderer
from mjlab.viewer.viewer_config import ViewerConfig

# HU_D03 joints in MJCF / qpos order (matches GMR's robot DoF order when it loads
# the same hu_d03.xml). MUST match the CSV column order.
HU_D03_JOINT_NAMES = [
  "left_hip_pitch_joint",
  "left_hip_roll_joint",
  "left_hip_yaw_joint",
  "left_knee_joint",
  "left_ankle_pitch_joint",
  "left_ankle_roll_joint",
  "right_hip_pitch_joint",
  "right_hip_roll_joint",
  "right_hip_yaw_joint",
  "right_knee_joint",
  "right_ankle_pitch_joint",
  "right_ankle_roll_joint",
  "waist_yaw_joint",
  "waist_roll_joint",
  "waist_pitch_joint",
  "left_shoulder_pitch_joint",
  "left_shoulder_roll_joint",
  "left_shoulder_yaw_joint",
  "left_elbow_joint",
  "left_wrist_yaw_joint",
  "left_wrist_pitch_joint",
  "left_hand_yaw_joint",
  "right_shoulder_pitch_joint",
  "right_shoulder_roll_joint",
  "right_shoulder_yaw_joint",
  "right_elbow_joint",
  "right_wrist_yaw_joint",
  "right_wrist_pitch_joint",
  "right_hand_yaw_joint",
  "head_yaw_joint",
  "head_pitch_joint",
]


def main(
  input_file: str,
  output_name: str,
  input_fps: float = 30.0,
  output_fps: float = 50.0,
  device: str = "cuda:0",
  render: bool = False,
  line_range: tuple[int, int] | None = None,
):
  """Replay an HU_D03 motion CSV and write the npz (and upload to W&B)."""
  import torch

  if device.startswith("cuda") and not torch.cuda.is_available():
    print("[WARNING]: CUDA not available. Falling back to CPU.")
    device = "cpu"

  sim_cfg = SimulationCfg()
  sim_cfg.mujoco.timestep = 1.0 / output_fps

  scene = Scene(hu_d03_flat_tracking_env_cfg().scene, device=device)
  model = scene.compile()
  sim = Simulation(num_envs=1, cfg=sim_cfg, model=model, device=device)
  scene.initialize(sim.mj_model, sim.model, sim.data)

  renderer = None
  if render:
    viewer_cfg = ViewerConfig(
      height=480,
      width=640,
      origin_type=ViewerConfig.OriginType.ASSET_ROOT,
      entity_name="robot",
      distance=2.0,
      elevation=-5.0,
      azimuth=20,
    )
    renderer = OffscreenRenderer(model=sim.mj_model, cfg=viewer_cfg, scene=scene)
    renderer.initialize()

  run_sim(
    sim=sim,
    scene=scene,
    joint_names=HU_D03_JOINT_NAMES,
    input_fps=input_fps,
    input_file=input_file,
    output_fps=output_fps,
    output_name=output_name,
    render=render,
    line_range=line_range,
    renderer=renderer,
  )


if __name__ == "__main__":
  tyro.cli(main)
