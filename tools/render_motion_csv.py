"""Render a HU_D03 reference-motion CSV (kinematic playback) to mp4 to visually
verify GMR retargeting quality. CSV row = base_pos(3) + base_quat xyzw(4) + 31
joints (MJCF order)."""

import argparse

import imageio
import mujoco
import numpy as np

from mjlab.asset_zoo.robots.limx_hu_d03.hu_d03_constants import HU_D03_XML

JOINTS = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_yaw_joint", "left_wrist_pitch_joint",
    "left_hand_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_yaw_joint", "right_wrist_pitch_joint",
    "right_hand_yaw_joint",
    "head_yaw_joint", "head_pitch_joint",
]

ap = argparse.ArgumentParser()
ap.add_argument("--csv", required=True)
ap.add_argument("--out", default="/home/odindev/ml/dance_retarget.mp4")
ap.add_argument("--fps", type=int, default=30)
a = ap.parse_args()

csv = np.loadtxt(a.csv, delimiter=",")
print(f"[render] csv {csv.shape}")

spec = mujoco.MjSpec.from_file(str(HU_D03_XML))
floor = spec.worldbody.add_geom()
floor.name = "floor"
floor.type = mujoco.mjtGeom.mjGEOM_PLANE
floor.size = [10.0, 10.0, 0.1]
model = spec.compile()
data = mujoco.MjData(model)

jadr = [
    model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]
    for n in JOINTS
]
base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")

cam = mujoco.MjvCamera()
cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
cam.trackbodyid = base_id
cam.distance = 3.0
cam.elevation = -10.0
cam.azimuth = 90.0

renderer = mujoco.Renderer(model, height=360, width=480)
frames = []
for row in csv:
    data.qpos[0:3] = row[0:3]
    data.qpos[3:7] = [row[6], row[3], row[4], row[5]]  # xyzw -> wxyz
    for k, adr in enumerate(jadr):
        data.qpos[adr] = row[7 + k]
    mujoco.mj_forward(model, data)
    renderer.update_scene(data, camera=cam)
    frames.append(renderer.render())

imageio.mimwrite(a.out, frames, fps=a.fps, macro_block_size=None)
print(f"[render] wrote {a.out} ({len(frames)} frames)")
