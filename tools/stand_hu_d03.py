"""Phase 1.3a: build a curated HU_D03 MJCF and test standing in pure MuJoCo.

Curation:
  - add a free joint to base_link
  - make all URDF geoms visual-only, add box foot-collision geoms + foot sites
    on the ankle_roll links (feet-only collision)
  - compute the standing base height by forward kinematics (feet at z=0)
  - add a "home" keyframe (straight legs)

Standing test (pure MuJoCo, no mjlab): hold all 31 hinges at 0 with joint
springs + damping and check the base height stays ~constant under gravity.
The saved XML has NO actuators (mjlab adds them via the constants file).
"""

import pathlib

import mujoco
import numpy as np

ROOT = pathlib.Path.home() / "ml" / "humanoid-description" / "HU_D03_description"
URDF_TMP = ROOT / "_hu_d03_mj.urdf"
OUT_XML = ROOT / "hu_d03.xml"

spec = mujoco.MjSpec.from_file(str(URDF_TMP))

# 1. Free joint on base_link.
base = next(b for b in spec.bodies if b.name == "base_link")
base.add_freejoint()

# 2. All existing (URDF) geoms -> visual only.
for g in spec.geoms:
    g.contype = 0
    g.conaffinity = 0

# 3. Foot collision boxes + sites on the ankle-roll links (feet-only collision).
#    Box half-extents; offset below the ankle. Approximate (no datasheet).
FOOT_SIZE = [0.11, 0.05, 0.018]
FOOT_POS = [0.03, 0.0, -0.05]
for side in ("left", "right"):
    b = next(bb for bb in spec.bodies if bb.name == f"{side}_ankle_roll_link")
    g = b.add_geom()
    g.name = f"{side}_foot1_collision"
    g.type = mujoco.mjtGeom.mjGEOM_BOX
    g.size = FOOT_SIZE
    g.pos = FOOT_POS
    g.contype = 1
    g.conaffinity = 1
    g.condim = 3
    g.friction = [1.0, 0.005, 0.0001]
    s = b.add_site()
    s.name = f"{side}_foot"
    s.pos = [FOOT_POS[0], 0.0, FOOT_POS[2] - FOOT_SIZE[2]]

# 3b. IMU site on base_link + native MuJoCo sensors required by mjlab's
#     velocity/tracking observations (imu_ang_vel, imu_lin_vel, root_angmom;
#     imu_lin_acc / imu_upvector mirror the G1 sensor block).
imu = base.add_site()
imu.name = "imu_in_base"
imu.size = [0.01, 0.01, 0.01]
imu.pos = [0.0, 0.0, 0.0]
for nm, st in (
    ("imu_ang_vel", mujoco.mjtSensor.mjSENS_GYRO),
    ("imu_lin_vel", mujoco.mjtSensor.mjSENS_VELOCIMETER),
    ("imu_lin_acc", mujoco.mjtSensor.mjSENS_ACCELEROMETER),
):
    s = spec.add_sensor()
    s.name = nm
    s.type = st
    s.objtype = mujoco.mjtObj.mjOBJ_SITE
    s.objname = "imu_in_base"
s = spec.add_sensor()
s.name = "imu_upvector"
s.type = mujoco.mjtSensor.mjSENS_FRAMEZAXIS
s.objtype = mujoco.mjtObj.mjOBJ_BODY
s.objname = "world"
s.reftype = mujoco.mjtObj.mjOBJ_SITE
s.refname = "imu_in_base"
s = spec.add_sensor()
s.name = "root_angmom"
s.type = mujoco.mjtSensor.mjSENS_SUBTREEANGMOM
s.objtype = mujoco.mjtObj.mjOBJ_BODY
s.objname = "base_link"

# 4. Ground plane.
floor = spec.worldbody.add_geom()
floor.name = "floor"
floor.type = mujoco.mjtGeom.mjGEOM_PLANE
floor.size = [10.0, 10.0, 0.1]
floor.contype = 1
floor.conaffinity = 1

model = spec.compile()
data = mujoco.MjData(model)

# --- compute standing base height: feet touching z=0 with all hinges at 0 ---
mujoco.mj_resetData(model, data)
qadr_base_z = 2  # free joint: qpos[0:3]=pos, [3:7]=quat
data.qpos[qadr_base_z] = 1.0
data.qpos[3:7] = [1, 0, 0, 0]
mujoco.mj_forward(model, data)
foot_gids = [
    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{s}_foot1_collision")
    for s in ("left", "right")
]
foot_bottom = min(
    data.geom_xpos[g][2] - FOOT_SIZE[2] for g in foot_gids
)
stand_z = 1.0 - foot_bottom + 0.002
print(f"[height] foot_bottom@z=1.0 -> {foot_bottom:.3f}; standing base_z={stand_z:.3f}")

# --- standing test: spring-hold hinges at 0, damped, drop from stand_z ---
hinge_dof = np.arange(6, model.nv)  # 6 free dofs first
model.dof_damping[6:] = 8.0
model.jnt_stiffness[1:] = 300.0  # joint 0 is the free joint; 1..31 are hinges
# qpos_spring reference defaults to qpos0 (=0 for hinges) -> holds straight pose.

mujoco.mj_resetData(model, data)
data.qpos[qadr_base_z] = stand_z
data.qpos[3:7] = [1, 0, 0, 0]
mujoco.mj_forward(model, data)

z0 = float(data.qpos[qadr_base_z])
nan_step = -1
traj = []
N = 2000
for i in range(N):
    mujoco.mj_step(model, data)
    if not np.isfinite(data.qpos).all():
        nan_step = i
        break
    if i in (0, 199, 499, 999, 1999):
        traj.append((i, float(data.qpos[qadr_base_z]), int(data.ncon)))

print(f"[stand] start base_z={z0:.3f}")
print("[stand] step : base_z : ncontacts")
for i, z, nc in traj:
    print(f"         {i:4d} : {z:6.3f} : {nc}")
if nan_step >= 0:
    print(f"[FAIL] non-finite at step {nan_step}")
else:
    drop = z0 - float(data.qpos[qadr_base_z])
    verdict = "STANDS" if abs(drop) < 0.10 else "FELL/SANK"
    print(f"[stand] final base_z={data.qpos[qadr_base_z]:.3f}  drop={drop:.3f} -> {verdict}")

# --- add a home keyframe (straight legs at standing height) and save XML ---
key = spec.add_key()
key.name = "home"
qpos0 = [0.0] * model.nq
qpos0[0:3] = [0.0, 0.0, stand_z]
qpos0[3:7] = [1.0, 0.0, 0.0, 0.0]
key.qpos = qpos0
OUT_XML.write_text(spec.to_xml())
print(f"[ok] saved curated MJCF: {OUT_XML}")
