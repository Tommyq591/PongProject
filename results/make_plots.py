"""Rebuild the graphs and score tables in results/ from the recorded experiment logs.

  python results/make_plots.py
Reads experiments/03_pilot_old_settings/recorded/run/evaluations.csv (old settings pilot) and
experiments/06_configA_extend_5M/recorded/run/evaluations.csv (config A, 0-5M: the 2M run's
history is included because the extension was resumed from a copy of that run folder).
"""
import csv, os, sys
from collections import defaultdict
from pathlib import Path
HERE = Path(__file__).resolve().parent
REPO = HERE.parent
os.environ.setdefault("MPLCONFIGDIR", str(HERE / ".matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

def evals(path):
    by = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            by[int(r["step"])].append(float(r["score"]))
    steps = np.array(sorted(by))
    s = [np.array(by[k]) for k in steps]
    return steps, np.array([x.mean() for x in s]), np.array([x.min() for x in s]), np.array([x.max() for x in s]), [len(x) for x in s]

A = evals(REPO / "experiments/06_configA_extend_5M/recorded/run/evaluations.csv")
OLD = evals(REPO / "experiments/03_pilot_old_settings/recorded/run/evaluations.csv")

for name, (st, mean, lo, hi, n) in (("eval_scores_configA_5M.csv", A), ("eval_scores_oldsettings_pilot.csv", OLD)):
    with open(HERE / name, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["step", "games", "mean_score", "min_score", "max_score"])
        for row in zip(st, n, mean, lo, hi):
            w.writerow([row[0], row[1], f"{row[2]:.2f}", f"{row[3]:.0f}", f"{row[4]:.0f}"])

# ---- 1. config A learning curve 0-5M
st, mean, lo, hi, n = A
fig, ax = plt.subplots(figsize=(10, 5.2))
ax.fill_between(st / 1e6, lo, hi, color="tab:blue", alpha=0.18, label="min-max of the evaluation games")
ax.plot(st / 1e6, mean, color="tab:blue", marker="o", ms=3.5, lw=1.8,
        label="mean evaluation score (5 games; 10 at 2M and 5M)")
ax.axhline(0, color="black", ls="--", lw=1, label="0 = break-even (wins as many points as it loses)")
ax.axvline(2.0, color="tab:red", ls=":", lw=1.5)
ax.annotate("resumed from checkpoint\n(2M run extended to 5M)", xy=(2.0, mean[st == 2_000_000][0]),
            xytext=(2.15, -17), color="tab:red", fontsize=9,
            arrowprops=dict(arrowstyle="->", color="tab:red"))
ax.annotate(f"5M: {mean[-1]:+.1f} (10/10 games won)" if lo[-1] > 0 else f"5M: {mean[-1]:+.1f}",
            xy=(st[-1] / 1e6, mean[-1]), xytext=(3.75, 17.5), fontsize=9,
            arrowprops=dict(arrowstyle="->"))
ax.set(xlim=(0, 5.05), ylim=(-21, 21), xlabel="training decisions (millions)",
       ylabel="game score (our points - opponent points)",
       title="DQN, config A, seed 17: evaluation score during training (epsilon 0.01, held-out games)")
ax.set_yticks(range(-21, 22, 3)); ax.grid(alpha=0.3); ax.legend(loc="upper left", fontsize=8.5)
fig.tight_layout(); fig.savefig(HERE / "learning_curve_configA_5M.png", dpi=130); plt.close(fig)

# ---- 2. old settings pilot vs config A, first 1M
fig, ax = plt.subplots(figsize=(9, 5))
for (st, mean, lo, hi, n), colour, label in (
        (OLD, "tab:orange", "old settings pilot (small network, replay 100k; killed at 700k)"),
        (A, "tab:blue", "config A (Nature network, replay 200k, ...)")):
    k = st <= 1_000_000
    ax.fill_between(st[k] / 1e3, lo[k], hi[k], color=colour, alpha=0.15)
    ax.plot(st[k] / 1e3, mean[k], color=colour, marker="o", ms=3.5, lw=1.8, label=label)
ax.axhline(0, color="black", ls="--", lw=1)
ax.set(xlim=(0, 1000), ylim=(-21, 21), xlabel="training decisions (thousands)",
       ylabel="game score (our points - opponent points)",
       title="Old settings vs config A over the first 1M decisions (DQN, seed 17)\n"
             "line = mean of 5 evaluation games, shading = min-max")
ax.set_yticks(range(-21, 22, 3)); ax.grid(alpha=0.3); ax.legend(loc="upper left", fontsize=9)
fig.tight_layout(); fig.savefig(HERE / "old_vs_new_settings.png", dpi=130); plt.close(fig)
print("wrote learning_curve_configA_5M.png, old_vs_new_settings.png, eval_scores_*.csv")
