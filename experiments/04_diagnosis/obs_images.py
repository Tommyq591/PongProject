"""Save what the network actually sees: real 4x84x84 stacks from the notebook's PongEnv.

For 3 moments with the ball in play: the raw RGB frame, then the 4 stacked frames
(oldest -> newest) exactly as stored (uint8), upscaled with nearest-neighbour only.
Also measures ball contrast in the newest 84x84 frame.
"""
import importlib.util, json, os, sys
from dataclasses import replace
from pathlib import Path
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]   # repo root (all paths are relative to it)
sys.path.insert(0, str(REPO / "common"))
OUT = HERE / "output"; OUT.mkdir(exist_ok=True)   # re-runs write here; recorded results are in recorded/
os.environ.setdefault("MPLCONFIGDIR", str(HERE / ".matplotlib"))
spec = importlib.util.spec_from_file_location("pong_train_extract", REPO / "common" / "pong_train_extract.py")
nb = importlib.util.module_from_spec(spec); sys.modules["pong_train_extract"] = nb; spec.loader.exec_module(nb)
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt

shared = json.loads((REPO / "Assignment2" / "experiment_plan.json").read_text())["shared_config"]
config = nb.Config(**shared)
env = nb.PongEnv(config, video=True)          # identical wrappers; rgb only for the raw panel
obs, _ = env.reset(seed=5)
rng = np.random.default_rng(5)
WHITE = np.array([236, 236, 236])
shots, stats, t = [], [], 0
while len(shots) < 3 and t < 3000:
    obs, r, term, trunc, _ = env.step(int(rng.integers(6)))
    t += 1
    frame = env.raw.render()
    ys, xs = np.nonzero(np.all(frame[34:194] == WHITE, axis=2))
    if len(ys) and t > 60 and t % 37 == 0:
        by, bx = ys.mean() + 34, xs.mean()
        oy, ox = int(round(by * 84 / 210)), int(round(bx * 84 / 160))
        newest = obs[-1].astype(int)
        patch = newest[max(0, oy - 2):oy + 3, max(0, ox - 2):ox + 3]
        bg = int(np.median(newest[8:78, 20:64]))
        stats.append({"t": t, "ball_raw_yx": [round(by, 1), round(bx, 1)], "ball_obs_yx": [oy, ox],
                      "ball_max_grey": int(patch.max()), "background_grey": bg,
                      "ball_pixels_above_bg+30": int((patch > bg + 30).sum())})
        shots.append((frame.copy(), obs.copy(), (oy, ox)))
    if term or trunc:
        obs, _ = env.reset()
env.close()

fig, axes = plt.subplots(len(shots), 5, figsize=(14, 3.2 * len(shots)), constrained_layout=True)
for row, (frame, stack, (oy, ox)) in zip(np.atleast_2d(axes), shots):
    row[0].imshow(frame); row[0].set_title("raw RGB 210x160"); row[0].axis("off")
    for i in range(4):
        row[i + 1].imshow(stack[i], cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        row[i + 1].set_title(f"obs[{i}] {'(newest)' if i == 3 else '(oldest)' if i == 0 else ''}  84x84")
        row[i + 1].axis("off")
    row[4].add_patch(plt.Circle((ox, oy), 4, fill=False, color="red", lw=1.2))
fig.suptitle("What the network sees: 4 stacked greyscale 84x84 frames (uint8; divided by 255 inside QNetwork). Red circle = ball.")
fig.savefig(OUT / "observations.png", dpi=110)
np.save(OUT / "observation_example.npy", shots[0][1])
# frame order check on these real stacks: consecutive frames should differ (motion), oldest->newest
order = [[int(np.abs(s[1][i + 1].astype(int) - s[1][i]).sum()) for i in range(3)] for s in shots]
out = {"dtype": str(shots[0][1].dtype), "shape": list(shots[0][1].shape), "ball_contrast": stats,
       "abs_diff_between_consecutive_frames": order}
(OUT / "observations.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=1))
