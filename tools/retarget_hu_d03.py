"""Headless GMR retarget of a LAFAN1 BVH onto HU_D03 -> GMR pickle.

No viewer (works on the server). Saves the same pickle schema as GMR's
bvh_to_robot.py (root_rot stored xyzw), consumable by batch_gmr_pkl_to_csv.py.
"""

import argparse
import os
import pickle

import numpy as np
from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting.utils.lafan1 import load_bvh_file

ap = argparse.ArgumentParser()
ap.add_argument("--bvh_file", required=True)
ap.add_argument("--save_path", required=True)
ap.add_argument("--robot", default="limx_hu_d03")
ap.add_argument("--format", default="lafan1")
ap.add_argument("--motion_fps", type=int, default=30)
ap.add_argument("--start", type=int, default=0)
ap.add_argument("--max_frames", type=int, default=0, help="0 = all frames")
a = ap.parse_args()

frames, human_height = load_bvh_file(a.bvh_file, format=a.format)
print(f"[retarget] loaded {len(frames)} frames, human_height={human_height:.3f}")

retargeter = GMR(
    src_human=f"bvh_{a.format}",
    tgt_robot=a.robot,
    actual_human_height=human_height,
)
print(f"[retarget] robot DoF order: {list(retargeter.robot_dof_names.keys())}")

end = len(frames) if a.max_frames <= 0 else min(len(frames), a.start + a.max_frames)
qpos_list = []
for i in range(a.start, end):
    qpos = retargeter.retarget(frames[i])
    qpos_list.append(np.asarray(qpos).copy())

root_pos = np.array([q[:3] for q in qpos_list])
root_rot = np.array([q[3:7][[1, 2, 3, 0]] for q in qpos_list])  # wxyz -> xyzw
dof_pos = np.array([q[7:] for q in qpos_list])

os.makedirs(os.path.dirname(a.save_path) or ".", exist_ok=True)
with open(a.save_path, "wb") as f:
    pickle.dump(
        {
            "fps": a.motion_fps,
            "root_pos": root_pos,
            "root_rot": root_rot,
            "dof_pos": dof_pos,
            "local_body_pos": None,
            "link_body_list": None,
        },
        f,
    )
print(f"[retarget] saved {a.save_path}: {dof_pos.shape[0]} frames, {dof_pos.shape[1]} dof")
