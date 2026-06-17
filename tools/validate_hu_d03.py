"""Validate the HU_D03 mjlab robot package: build the Entity, check the
actuators wire up, and run a position-actuator standing test on flat ground.
"""

import mujoco
import numpy as np

from mjlab.asset_zoo.robots.limx_hu_d03.hu_d03_constants import get_hu_d03_robot_cfg
from mjlab.entity.entity import Entity

ent = Entity(get_hu_d03_robot_cfg())
spec = ent.spec

# Add a ground plane to the entity spec for the standing test.
floor = spec.worldbody.add_geom()
floor.name = "floor"
floor.type = mujoco.mjtGeom.mjGEOM_PLANE
floor.size = [10.0, 10.0, 0.1]
floor.contype = 1
floor.conaffinity = 1

model = spec.compile()
data = mujoco.MjData(model)
print(
    f"[entity] nu(actuators)={model.nu}  njnt={model.njnt}  nq={model.nq}  "
    f"nv={model.nv}  nkey={model.nkey}"
)
assert model.nu == 31, f"expected 31 actuators, got {model.nu}"

# Reset to the mjlab-generated keyframe (base at standing height, joints at 0).
if model.nkey > 0:
    mujoco.mj_resetDataKeyframe(model, data, 0)
else:
    mujoco.mj_resetData(model, data)

# Position actuators: hold the keyframe pose by setting each actuator's target
# to the current joint angle (the official stand pose loaded from the keyframe).
for i in range(model.nu):
    jid = int(model.actuator_trnid[i, 0])
    data.ctrl[i] = float(data.qpos[model.jnt_qposadr[jid]])

z0 = float(data.qpos[2])
nan_step = -1
traj = []
N = 2000
for i in range(N):
    mujoco.mj_step(model, data)
    if not np.isfinite(data.qpos).all():
        nan_step = i
        break
    if i in (0, 499, 999, 1999):
        traj.append((i, float(data.qpos[2]), int(data.ncon)))

print(f"[stand] start base_z={z0:.3f}  (ctrl=position targets, holding keyframe)")
print("[stand] step : base_z : ncontacts")
for i, z, nc in traj:
    print(f"         {i:4d} : {z:6.3f} : {nc}")
if nan_step >= 0:
    print(f"[FAIL] non-finite at step {nan_step}")
else:
    drop = z0 - float(data.qpos[2])
    verdict = "STANDS" if abs(drop) < 0.12 else "FELL/SANK"
    print(f"[stand] final base_z={data.qpos[2]:.3f}  drop={drop:.3f} -> {verdict}")
