"""Headless rollout + render of the HU_D03 velocity policy -> mp4.

Loads the latest checkpoint, runs the policy in a 1-env play environment, renders
each control step with the offscreen renderer, and writes an mp4. No interactive
viewer (works headless via EGL software rendering).
"""

import glob
import pathlib
from dataclasses import asdict

import imageio
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

TASK = "Mjlab-Velocity-Flat-LimX-HU-D03"
DEVICE = "cuda:0"
N_FRAMES = 900
OUT = "/home/odindev/ml/hu_d03_play.mp4"

# Latest run + latest checkpoint.
runs = sorted(
    glob.glob(str(pathlib.Path.home() / "ml/mjlab/logs/rsl_rl/hu_d03_velocity/*"))
)
run = runs[-1]
ckpts = sorted(
    glob.glob(run + "/model_*.pt"),
    key=lambda p: int(p.split("model_")[1].split(".pt")[0]),
)
ckpt = ckpts[-1]
print(f"[render] run={run.split('/')[-1]}  checkpoint={ckpt.split('/')[-1]}")

env_cfg = load_env_cfg(TASK, play=True)
env_cfg.scene.num_envs = 1
env_cfg.viewer.height = 300
env_cfg.viewer.width = 400
# No terminations: if the robot falls it stays down (natural "death") instead of
# resetting, so the clip runs continuously until N_FRAMES.
env_cfg.terminations = {}

# Force a constant forward walk command (0.6 m/s) so the clip shows the robot
# actually trying to walk, not a randomly-sampled near-zero / standing command.
tw = env_cfg.commands["twist"]
tw.ranges.lin_vel_x = (2.0, 2.0)  # max commanded speed
tw.ranges.lin_vel_y = (0.0, 0.0)
tw.ranges.ang_vel_z = (0.0, 0.0)
if hasattr(tw, "rel_standing_envs"):
    tw.rel_standing_envs = 0.0
if hasattr(tw, "heading_command"):
    tw.heading_command = False
    if hasattr(tw.ranges, "heading"):
        tw.ranges.heading = None
agent_cfg = load_rl_cfg(TASK)

raw_env = ManagerBasedRlEnv(cfg=env_cfg, device=DEVICE, render_mode="rgb_array")
env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)

runner_cls = load_runner_cls(TASK) or MjlabOnPolicyRunner
runner = runner_cls(env, asdict(agent_cfg), device=DEVICE)
runner.load(ckpt, load_cfg={"actor": True}, strict=True, map_location=DEVICE)
policy = runner.get_inference_policy(device=DEVICE)

res = env.get_observations()
obs = res[0] if isinstance(res, tuple) else res

robot = raw_env.scene["robot"]
cmd0 = raw_env.command_manager.get_command("twist")[0].tolist()
print(f"[diag] applied command (vx,vy,wz) = {[round(c, 3) for c in cmd0]}")
x_start = float(robot.data.root_link_pos_w[0, 0])
y_start = float(robot.data.root_link_pos_w[0, 1])
fwd_vels = []

frames = []
for i in range(N_FRAMES):
    with torch.inference_mode():
        act = policy(obs)
    step_out = env.step(act)
    obs = step_out[0]
    fwd_vels.append(float(robot.data.root_link_lin_vel_b[0, 0]))
    frame = raw_env.render()
    if frame is not None:
        frames.append(frame)

x_end = float(robot.data.root_link_pos_w[0, 0])
y_end = float(robot.data.root_link_pos_w[0, 1])
mean_fwd = sum(fwd_vels) / max(1, len(fwd_vels))
print(f"[diag] base displacement: dx={x_end - x_start:.3f} m  dy={y_end - y_start:.3f} m")
print(f"[diag] mean forward vel = {mean_fwd:.3f} m/s  (commanded {cmd0[0]:.2f} m/s)")
print(f"[render] collected {len(frames)} frames")
imageio.mimwrite(OUT, frames, fps=30, macro_block_size=None)
print(f"[render] WROTE {OUT}")
