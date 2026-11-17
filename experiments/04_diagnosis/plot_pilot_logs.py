"""Plot the 700k pilot's own training logs (../03_pilot_old_settings/recorded/run, read-only) -> output/pilot_learning_diagnostics.png + .json"""
import csv, json, os, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]   # repo root (all paths are relative to it)
sys.path.insert(0, str(REPO / "common"))
OUT = HERE / "output"; OUT.mkdir(exist_ok=True)   # re-runs write here; recorded results are in recorded/
RUN = REPO / "experiments" / "03_pilot_old_settings" / "recorded" / "run"
OUT.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(HERE / ".matplotlib"))
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt

def load(name):
    with (RUN / name).open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {k: np.array([float(r[k]) if r[k] not in ("True", "False") else r[k] == "True" for r in rows]) for k in rows[0]}
U, P, E, V = load("updates.csv"), load("progress.csv"), load("episodes.csv"), load("evaluations.csv")

def smooth(x, k=25):
    return np.convolve(x, np.ones(k) / k, "valid") if len(x) > k else x

fig, ax = plt.subplots(3, 2, figsize=(13, 10), constrained_layout=True)
s = U["step"]; k = 25; sm = s[k - 1:]
for a, key, title in ((ax[0, 0], "loss", "Huber loss (every 100th update)"),
                      (ax[0, 1], "mean_q", "mean Q(s,a) of sampled batch  vs  mean target"),
                      (ax[1, 0], "mean_abs_td_error", "mean |TD error|"),
                      (ax[1, 1], "gradient_norm", "gradient norm (before clipping at 10)")):
    a.plot(s, U[key], color="#c9d3dd", lw=.6); a.plot(sm, smooth(U[key], k), color="#1f6feb", lw=1.5)
    a.set_title(title); a.set_xlabel("decision")
ax[0, 1].plot(sm, smooth(U["mean_target"], k), color="#d1242f", lw=1.2, label="mean target (smoothed)")
ax[0, 1].legend(fontsize=8)
ax[1, 1].axhline(10, color="#d1242f", ls="--", lw=.8)
ax[2, 0].plot(P["step"], P["epsilon"], color="#52606d"); ax[2, 0].set_title("epsilon (exploration)"); ax[2, 0].set_xlabel("decision")
ax2 = ax[2, 0].twinx(); ax2.plot(P["step"], P["updates"], color="#1a7f37", lw=1); ax2.set_ylabel("gradient updates so far", color="#1a7f37")
ax[2, 1].plot(E["step"], E["score"], ".", color="#9aa5b1", ms=2, label="training episode return")
ax[2, 1].plot(E["step"][19:], smooth(E["score"], 20), color="#52606d", label="20-episode mean")
ev = sorted(set(V["step"]))
ax[2, 1].plot(ev, [V["score"][V["step"] == x].mean() for x in ev], "o-", color="#1f6feb", label="evaluation (5 games)")
ax[2, 1].set_ylim(-22, 0); ax[2, 1].set_title("episode return"); ax[2, 1].set_xlabel("decision"); ax[2, 1].legend(fontsize=8)
fig.suptitle("Pilot DQN (seed 17, cuda, replay 100k) - training diagnostics from its own logs, 0-700k decisions")
fig.savefig(OUT / "pilot_learning_diagnostics.png", dpi=120)

def window(arr, steps, lo, hi):
    m = (steps >= lo) & (steps < hi); return float(arr[m].mean()) if m.any() else None
bands = [(0, 50_000), (50_000, 100_000), (100_000, 200_000), (200_000, 400_000), (400_000, 600_000), (600_000, 700_000)]
summary = {f"{lo//1000}-{hi//1000}k": {key: window(U[key], s, lo, hi) for key in ("loss", "mean_q", "mean_target", "mean_abs_td_error", "gradient_norm")}
           | {"train_return": window(E["score"], E["step"], lo, hi), "episode_length": window(E["length"], E["step"], lo, hi)}
           for lo, hi in bands}
summary["grad_norm_share_above_clip_10"] = float(np.mean(U["gradient_norm"] > 10))
summary["updates_by_700k"] = float(P["updates"][-1]); summary["last_step"] = float(P["step"][-1])
summary["eval"] = {int(x): float(V["score"][V["step"] == x].mean()) for x in ev}
(OUT / "pilot_learning_diagnostics.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=1))
