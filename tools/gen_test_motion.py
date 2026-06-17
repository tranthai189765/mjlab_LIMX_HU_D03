"""Generate a trivial HU_D03 motion CSV (standing pose + gentle sway) to verify
the tracking data pipeline. CSV row = base_pos(3) + base_quat xyzw(4) + 31 joints
(MJCF order)."""

import numpy as np

# Standing pose, MJCF joint order (matches csv_to_npz_hu_d03 HU_D03_JOINT_NAMES).
STAND = [
    -0.15, 0.0, -0.05, 0.30, -0.16, 0.0,   # left leg
    -0.15, 0.0, 0.05, 0.30, -0.16, 0.0,    # right leg
    0.0, 0.0, 0.0,                          # waist
    0.1, 0.1, -0.2, -0.2, 0.0, 0.0, 0.0,   # left arm
    0.1, -0.1, 0.2, -0.2, 0.0, 0.0, 0.0,   # right arm
    0.0, 0.0,                               # head
]
BASE_Z = 0.911
N, FPS = 180, 30

rows = []
for i in range(N):
    t = i / FPS
    s = 0.3 * np.sin(2 * np.pi * 0.5 * t)
    q = list(STAND)
    q[15] += s          # left_shoulder_pitch sway
    q[22] += s          # right_shoulder_pitch sway
    q[3] += 0.1 * np.sin(2 * np.pi * 0.5 * t)   # left_knee bob
    q[9] += 0.1 * np.sin(2 * np.pi * 0.5 * t)   # right_knee bob
    z = BASE_Z + 0.02 * np.sin(2 * np.pi * 0.5 * t)
    rows.append([0.0, 0.0, z, 0.0, 0.0, 0.0, 1.0] + q)  # pos + quat(xyzw) + joints

arr = np.asarray(rows)
np.savetxt("/home/odindev/ml/test_motion.csv", arr, delimiter=",")
print(f"wrote /home/odindev/ml/test_motion.csv  shape={arr.shape}")
