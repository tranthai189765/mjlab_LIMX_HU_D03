"""Parse the saved training logs and plot reward / episode-length curves -> PNG."""

import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DELIV = "/home/odindev/ml/deliverables"


def parse(logpath, metrics):
    data = {k: ([], []) for k in metrics}
    cur = None
    it_re = re.compile(r"Learning iteration\s+(\d+)\s*/")
    mre = {k: re.compile(p) for k, p in metrics.items()}
    with open(logpath, errors="ignore") as f:
        for line in f:
            m = it_re.search(line)
            if m:
                cur = int(m.group(1))
                continue
            for k, r in mre.items():
                mm = r.search(line)
                if mm and cur is not None:
                    data[k][0].append(cur)
                    data[k][1].append(float(mm.group(1)))
    return data


def smooth(x, y, w=41):
    x, y = np.asarray(x), np.asarray(y)
    if len(y) < w:
        return x, y
    k = np.ones(w) / w
    ys = np.convolve(y, k, mode="valid")
    xs = x[w // 2 : w // 2 + len(ys)]
    return xs, ys


def draw(ax, x, y, color, label, ylabel, title):
    ax.plot(x, y, color=color, alpha=0.25, lw=0.8)
    xs, ys = smooth(x, y)
    ax.plot(xs, ys, color=color, lw=2.0, label=label)
    ax.set_title(title)
    ax.set_xlabel("Training iteration")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)


REW = r"Mean reward:\s*([-\d.]+)"
TRACK = r"track_linear_velocity:\s*([-\d.]+)"
EP = r"Mean episode length:\s*([-\d.]+)"

loco = parse(f"{DELIV}/locomotion/training.log", {"r": REW, "t": TRACK})
dance = parse(f"{DELIV}/dance/training.log", {"r": REW, "e": EP})
jumps = parse(f"{DELIV}/jumps/training.log", {"r": REW, "e": EP})

# Figure 1: Locomotion
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
draw(ax[0], *loco["r"], "C0", "mean reward", "Mean reward",
     "Locomotion — Mean reward")
draw(ax[1], *loco["t"], "C2", "lin-vel tracking", "Reward",
     "Locomotion — Linear-velocity tracking reward")
fig.suptitle("HU_D03 Locomotion (velocity task) — PPO training", fontweight="bold")
fig.tight_layout()
fig.savefig(f"{DELIV}/fig_locomotion_training.png", dpi=140, bbox_inches="tight")

# Figure 2: Mimic (dance + jumps) — clip both to the same iteration range.
MIMIC_MAX = max(dance["r"][0])  # dance's last iteration (the shorter run)


def clip(xy, maxit=MIMIC_MAX):
    xs, ys = xy
    keep = [(x, y) for x, y in zip(xs, ys) if x <= maxit]
    return ([p[0] for p in keep], [p[1] for p in keep])


fig, ax = plt.subplots(1, 2, figsize=(11, 4))
draw(ax[0], *clip(dance["r"]), "C3", "dance", "Mean reward", "Mimic — Mean reward")
draw(ax[0], *clip(jumps["r"]), "C1", "jumps", "Mean reward", "Mimic — Mean reward")
ax[0].set_xlim(0, MIMIC_MAX)
ax[0].legend()
draw(ax[1], *clip(dance["e"]), "C3", "dance", "Episode length", "Mimic — Mean episode length")
draw(ax[1], *clip(jumps["e"]), "C1", "jumps", "Episode length", "Mimic — Mean episode length")
ax[1].axhline(500, ls="--", color="gray", alpha=0.6)
ax[1].set_xlim(0, MIMIC_MAX)
ax[1].legend()
fig.suptitle("HU_D03 Mimic (motion tracking) — PPO training", fontweight="bold")
fig.tight_layout()
fig.savefig(f"{DELIV}/fig_mimic_training.png", dpi=140, bbox_inches="tight")

# Figure 3: combined reward (all 3)
fig, ax = plt.subplots(figsize=(7, 4.5))
draw(ax, *loco["r"], "C0", "locomotion", "Mean reward", "HU_D03 — PPO training reward (all policies)")
draw(ax, *dance["r"], "C3", "mimic: dance", "Mean reward", "HU_D03 — PPO training reward (all policies)")
draw(ax, *jumps["r"], "C1", "mimic: jumps", "Mean reward", "HU_D03 — PPO training reward (all policies)")
ax.legend()
fig.tight_layout()
fig.savefig(f"{DELIV}/fig_all_rewards.png", dpi=140, bbox_inches="tight")

print("[plot] wrote fig_locomotion_training.png, fig_mimic_training.png, fig_all_rewards.png")
for name, d in [("loco", loco), ("dance", dance), ("jumps", jumps)]:
    print(f"  {name}: {len(d['r'][0])} reward points, last={d['r'][1][-1] if d['r'][1] else 'NA'}")
