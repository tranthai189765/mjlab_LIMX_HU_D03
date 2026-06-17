"""Quantitative evaluation of the HU_D03 locomotion policy across velocity
commands. For each command, runs 256 parallel envs, warms up, then measures the
mean achieved base velocity (body frame), tracking error, and fall rate."""

from dataclasses import asdict

import numpy as np
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

TASK = "Mjlab-Velocity-Flat-LimX-HU-D03"
DEVICE = "cuda:0"
NENV = 256
CKPT = "/home/odindev/ml/deliverables/locomotion/model_14450.pt"
WARMUP, MEASURE = 200, 300

# (label, vx, vy, wz, which component is the target: 'x'|'y'|'w')
TESTS = [
    ("backward 0.5 m/s", -0.5, 0.0, 0.0, "x"),
    ("walk+turn vx0.6 wz0.5", 0.6, 0.0, 0.5, "w"),
    ("walk+turn fwd-comp", 0.6, 0.0, 0.5, "x"),
]

agent_cfg = load_rl_cfg(TASK)


def make_env(vx, vy, wz):
    cfg = load_env_cfg(TASK, play=True)
    cfg.scene.num_envs = NENV
    tw = cfg.commands["twist"]
    tw.ranges.lin_vel_x = (vx, vx)
    tw.ranges.lin_vel_y = (vy, vy)
    tw.ranges.ang_vel_z = (wz, wz)
    for attr in ("rel_standing_envs", "rel_heading_envs", "rel_forward_envs"):
        if hasattr(tw, attr):
            setattr(tw, attr, 0.0)
    if hasattr(tw, "heading_command"):
        tw.heading_command = False
        if hasattr(tw.ranges, "heading"):
            tw.ranges.heading = None
    return cfg


print(f"{'command':<18}{'target':>8}{'achieved':>10}{'±std':>8}{'|err|':>8}{'track%':>8}{'falls':>8}")
print("-" * 70)
rows = []
for label, vx, vy, wz, comp in TESTS:
    raw = ManagerBasedRlEnv(cfg=make_env(vx, vy, wz), device=DEVICE)
    env = RslRlVecEnvWrapper(raw, clip_actions=agent_cfg.clip_actions)
    runner = (load_runner_cls(TASK) or MjlabOnPolicyRunner)(env, asdict(agent_cfg), device=DEVICE)
    runner.load(CKPT, load_cfg={"actor": True}, strict=True, map_location=DEVICE)
    policy = runner.get_inference_policy(device=DEVICE)
    robot = raw.scene["robot"]

    res = env.get_observations()
    obs = res[0] if isinstance(res, tuple) else res
    for _ in range(WARMUP):
        with torch.inference_mode():
            obs = env.step(policy(obs))[0]

    vals, falls = [], 0
    for _ in range(MEASURE):
        with torch.inference_mode():
            out = env.step(policy(obs))
        obs = out[0]
        falls += int(out[2].sum().item())
        if comp == "x":
            v = robot.data.root_link_lin_vel_b[:, 0]
        elif comp == "y":
            v = robot.data.root_link_lin_vel_b[:, 1]
        else:
            v = robot.data.root_link_ang_vel_b[:, 2]
        vals.append(v.detach().cpu().numpy())

    arr = np.concatenate(vals)
    target = vx if comp == "x" else (vy if comp == "y" else wz)
    achieved, std = float(arr.mean()), float(arr.std())
    err = float(np.abs(arr - target).mean())
    track = 100.0 * achieved / target if target != 0 else 0.0
    fall_rate = 100.0 * falls / (NENV * MEASURE)
    print(f"{label:<18}{target:>8.2f}{achieved:>10.3f}{std:>8.3f}{err:>8.3f}{track:>7.1f}%{fall_rate:>7.2f}%")
    rows.append((label, target, achieved, std, err, track, fall_rate))

    try:
        raw.close()
    except Exception:
        pass
    del env, raw, runner, policy
    torch.cuda.empty_cache()

print("\n[done]")
