"""First-pass URDF -> MJCF conversion for LimX HU_D03 (Phase 1.1).

Strips package:// mesh prefixes, injects a <mujoco> compiler block, compiles in
MuJoCo, reports model stats + any errors, and saves a raw MJCF.
"""

import pathlib
import re

import mujoco

ROOT = pathlib.Path.home() / "ml" / "humanoid-description" / "HU_D03_description"
URDF_IN = ROOT / "urdf" / "HU_D03_03.urdf"
URDF_TMP = ROOT / "_hu_d03_mj.urdf"
MJCF_OUT = ROOT / "hu_d03_from_urdf.xml"

text = URDF_IN.read_text()

# 1. package://HU_D03_description/meshes/... -> meshes/...  (relative to meshdir)
text = text.replace("package://HU_D03_description/", "")

# 2. Inject a MuJoCo compiler block as a child of <robot ...>.
inject = (
    f'\n  <mujoco><compiler meshdir="{ROOT}" balanceinertia="true" '
    f'discardvisual="false" fusestatic="false" strippath="false"/></mujoco>'
)
text = re.sub(r"(<robot\b[^>]*>)", r"\1" + inject, text, count=1)
URDF_TMP.write_text(text)
print(f"[ok] wrote patched URDF: {URDF_TMP}")

# 3. Compile in MuJoCo.
try:
    model = mujoco.MjModel.from_xml_path(str(URDF_TMP))
except Exception as e:  # noqa: BLE001
    print("[FAIL] MuJoCo could not compile the URDF:")
    print(repr(e))
    raise SystemExit(1)

print("[ok] compiled in MuJoCo")
print(
    f"     nbody={model.nbody}  njnt={model.njnt}  nu={model.nu}  "
    f"nq={model.nq}  nv={model.nv}  ngeom={model.ngeom}  nmesh={model.nmesh}"
)

# List joints (name + type) so we can sanity check the 31 actuated DoFs.
JT = {0: "free", 1: "ball", 2: "slide", 3: "hinge"}
print("     joints:")
for j in range(model.njnt):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
    jtype = JT.get(int(model.jnt_type[j]), str(model.jnt_type[j]))
    print(f"       [{j:2d}] {jtype:6s} {name}")

# 4. Save raw MJCF.
mujoco.mj_saveLastXML(str(MJCF_OUT), model)
print(f"[ok] saved raw MJCF: {MJCF_OUT}")
