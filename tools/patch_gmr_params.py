"""Register limx_hu_d03 in GMR params.py (idempotent)."""

import pathlib

p = pathlib.Path.home() / "ml/GMR/general_motion_retargeting/params.py"
t = p.read_text()
if "limx_hu_d03" in t:
    print("already patched")
    raise SystemExit

reps = [
    (
        '"unitree_g1": ASSET_ROOT / "unitree_g1" / "g1_mocap_29dof.xml",',
        '"unitree_g1": ASSET_ROOT / "unitree_g1" / "g1_mocap_29dof.xml",\n'
        '    "limx_hu_d03": ASSET_ROOT / "limx_hu_d03" / "hu_d03.xml",',
    ),
    (
        '"unitree_g1": IK_CONFIG_ROOT / "bvh_lafan1_to_g1.json",',
        '"unitree_g1": IK_CONFIG_ROOT / "bvh_lafan1_to_g1.json",\n'
        '        "limx_hu_d03": IK_CONFIG_ROOT / "bvh_lafan1_to_hu_d03.json",',
    ),
    (
        '"unitree_g1": "pelvis",',
        '"unitree_g1": "pelvis",\n    "limx_hu_d03": "base_link",',
    ),
    (
        '"unitree_g1": 2.0,',
        '"unitree_g1": 2.0,\n    "limx_hu_d03": 2.0,',
    ),
]
for a, b in reps:
    n = t.count(a)
    assert n == 1, f"anchor not unique ({n}): {a}"
    t = t.replace(a, b)
p.write_text(t)
print("patched params.py with limx_hu_d03")
