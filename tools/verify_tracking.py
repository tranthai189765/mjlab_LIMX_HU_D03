"""Verify the HU_D03 tracking env loads a motion npz and steps without crashing."""

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.tracking.config.limx_hu_d03.env_cfgs import (
  hu_d03_flat_tracking_env_cfg,
)

cfg = hu_d03_flat_tracking_env_cfg(play=True)
cfg.scene.num_envs = 1
cfg.commands["motion"].motion_file = "/tmp/motion.npz"

env = ManagerBasedRlEnv(cfg, device="cuda:0")
print(f"[verify] tracking env built. action_dim={env.action_manager.total_action_dim}")

env.reset()
nact = env.action_manager.total_action_dim
for i in range(30):
    env.step(torch.zeros((1, nact), device="cuda:0"))
print("[verify] OK: motion npz loaded into tracking command, stepped 30x, no crash")
