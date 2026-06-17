"""Phase 1.2 physics-integrity check for HU_D03.

Adds a floating base + ground plane to the converted model, reports mass/inertia
sanity, then drops the robot under gravity (joints damped, no actuation) and
checks the simulation stays finite. Also saves a floating-base MJCF.
"""

import pathlib

import mujoco
import numpy as np

ROOT = pathlib.Path.home() / "ml" / "humanoid-description" / "HU_D03_description"
URDF_TMP = ROOT / "_hu_d03_mj.urdf"
MJCF_FLOAT = ROOT / "hu_d03_floating.xml"

spec = mujoco.MjSpec.from_file(str(URDF_TMP))

# --- add a free joint to base_link and lift it up ---
base = next((b for b in spec.bodies if b.name == "base_link"), None)
assert base is not None, "base_link not found"
try:
    base.add_freejoint()
except Exception:  # noqa: BLE001
    base.add_joint(type=mujoco.mjtJoint.mjJNT_FREE)
base.pos = [0.0, 0.0, 1.0]

# --- add a ground plane ---
floor = spec.worldbody.add_geom()
floor.name = "floor"
floor.type = mujoco.mjtGeom.mjGEOM_PLANE
floor.size = [5.0, 5.0, 0.1]

model = spec.compile()
print(
    f"[ok] compiled floating model: nbody={model.nbody} nq={model.nq} "
    f"nv={model.nv} ngeom={model.ngeom}"
)

# --- mass / inertia sanity ---
total_mass = float(model.body_mass.sum())
nonbase = model.body_mass[1:]  # skip world
print(f"[mass] total={total_mass:.2f} kg  "
      f"min_link={nonbase.min():.4f}  max_link={nonbase.max():.4f}")
diag = model.body_inertia
bad_inertia = int((diag <= 0).any(axis=1).sum())
print(f"[inertia] bodies with a non-positive principal inertia: {bad_inertia}")

# --- drop test (damped joints, no actuation) ---
model.dof_damping[6:] = 2.0  # damp the 31 hinges, leave the 6 free dofs
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)

nan_step = -1
samples = []
N = 1500
for i in range(N):
    mujoco.mj_step(model, data)
    if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
        nan_step = i
        break
    if i in (0, 249, 499, 999, N - 1):
        samples.append((i, float(data.qpos[2]), int(data.ncon)))

print("[drop] step : base_z : ncontacts")
for i, z, nc in samples:
    print(f"        {i:4d} : {z:6.3f} : {nc}")
if nan_step >= 0:
    print(f"[FAIL] simulation went non-finite at step {nan_step}")
else:
    print(f"[ok] simulation stayed finite for {N} steps; "
          f"final base_z={data.qpos[2]:.3f}")

# --- save floating MJCF artifact ---
MJCF_FLOAT.write_text(spec.to_xml())
print(f"[ok] saved floating MJCF: {MJCF_FLOAT}")
