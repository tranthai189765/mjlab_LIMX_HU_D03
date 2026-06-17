"""Compute HU_D03 standing base height for the OFFICIAL LimX stand pose."""

import pathlib

import mujoco

PKG_XML = (
    pathlib.Path.home()
    / "ml/mjlab/src/mjlab/asset_zoo/robots/limx_hu_d03/xmls/hu_d03.xml"
)

# Official LimX stand_pos (humanoid-rl-deploy-python/.../HU_D03_03).
STAND = {
    "left_hip_pitch_joint": -0.15, "left_hip_roll_joint": 0.0,
    "left_hip_yaw_joint": -0.05, "left_knee_joint": 0.30,
    "left_ankle_pitch_joint": -0.16, "left_ankle_roll_joint": 0.0,
    "right_hip_pitch_joint": -0.15, "right_hip_roll_joint": 0.0,
    "right_hip_yaw_joint": 0.05, "right_knee_joint": 0.30,
    "right_ankle_pitch_joint": -0.16, "right_ankle_roll_joint": 0.0,
}

model = mujoco.MjModel.from_xml_path(str(PKG_XML))
data = mujoco.MjData(model)
for name, val in STAND.items():
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    data.qpos[model.jnt_qposadr[jid]] = val
data.qpos[2] = 1.0
data.qpos[3:7] = [1, 0, 0, 0]
mujoco.mj_forward(model, data)

foot_bottoms = []
for s in ("left", "right"):
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{s}_foot1_collision")
    half_z = model.geom_size[gid][2]
    foot_bottoms.append(data.geom_xpos[gid][2] - half_z)
fb = min(foot_bottoms)
print(f"foot_bottom @ base_z=1.0 -> {fb:.4f}")
print(f"OFFICIAL stand base_z = {1.0 - fb + 0.002:.4f}")
