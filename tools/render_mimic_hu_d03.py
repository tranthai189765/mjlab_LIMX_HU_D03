"""Headless render of a HU_D03 mimic (tracking) policy following a reference -> mp4."""

import argparse
import glob
import pathlib
from dataclasses import asdict

import imageio
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

TASK = "Mjlab-Tracking-Flat-LimX-HU-D03"
DEVICE = "cuda:0"

ap = argparse.ArgumentParser()
ap.add_argument("--motion", default="/home/odindev/ml/motions/hu_d03_jump.npz")
ap.add_argument("--out", default="/home/odindev/ml/mimic_jump.mp4")
ap.add_argument("--run-match", default="", help="substring to pick the run dir")
ap.add_argument("--frames", type=int, default=800)
a = ap.parse_args()

runs = sorted(glob.glob(str(pathlib.Path.home() / "ml/mjlab/logs/rsl_rl/hu_d03_tracking/*")))
if a.run_match:
    runs = [r for r in runs if a.run_match in r]
run = runs[-1]
ckpts = sorted(
    glob.glob(run + "/model_*.pt"),
    key=lambda p: int(p.split("model_")[1].split(".pt")[0]),
)
ckpt = ckpts[-1]
print(f"[mimic] run={run.split('/')[-1]} checkpoint={ckpt.split('/')[-1]} motion={a.motion.split('/')[-1]}")

env_cfg = load_env_cfg(TASK, play=True)
env_cfg.scene.num_envs = 1
env_cfg.viewer.height = 300
env_cfg.viewer.width = 400
env_cfg.commands["motion"].motion_file = a.motion
agent_cfg = load_rl_cfg(TASK)

raw_env = ManagerBasedRlEnv(cfg=env_cfg, device=DEVICE, render_mode="rgb_array")
env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)

runner_cls = load_runner_cls(TASK) or MjlabOnPolicyRunner
runner = runner_cls(env, asdict(agent_cfg), device=DEVICE)
runner.load(ckpt, load_cfg={"actor": True}, strict=True, map_location=DEVICE)
policy = runner.get_inference_policy(device=DEVICE)

res = env.get_observations()
obs = res[0] if isinstance(res, tuple) else res

frames = []
for i in range(a.frames):
    with torch.inference_mode():
        act = policy(obs)
    obs = env.step(act)[0]
    f = raw_env.render()
    if f is not None:
        frames.append(f)

print(f"[mimic] collected {len(frames)} frames")
imageio.mimwrite(a.out, frames, fps=50, macro_block_size=None)
print(f"[mimic] WROTE {a.out}")
