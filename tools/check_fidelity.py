"""Fidelity audit of the converted HU_D03 MJCF vs the URDF."""

import pathlib

import mujoco
import numpy as np

PKG_XML = (
    pathlib.Path.home()
    / "ml/mjlab/src/mjlab/asset_zoo/robots/limx_hu_d03/xmls/hu_d03.xml"
)
m = mujoco.MjModel.from_xml_path(str(PKG_XML))

print(f"total mass = {m.body_mass.sum():.3f} kg")

# Joint position limits: every hinge should be limited with a finite range.
n_limited = 0
n_unlimited = 0
sample = []
for j in range(m.njnt):
    if m.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE:
        continue
    name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
    lim = bool(m.jnt_limited[j])
    rng = m.jnt_range[j]
    if lim:
        n_limited += 1
    else:
        n_unlimited += 1
    if len(sample) < 4:
        sample.append((name, lim, float(rng[0]), float(rng[1])))
print(f"hinge joints limited={n_limited} unlimited={n_unlimited}")
for name, lim, lo, hi in sample:
    print(f"   {name:24s} limited={lim} [{lo:.3f}, {hi:.3f}]")

# Bodies with negligible mass (virtual frames) -> the ones balanceinertia touched.
tiny = []
for b in range(1, m.nbody):
    if m.body_mass[b] < 1e-6:
        tiny.append(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b))
print(f"massless/virtual bodies ({len(tiny)}): {tiny}")
