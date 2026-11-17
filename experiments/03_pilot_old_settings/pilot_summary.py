"""Write output/summary.txt and output/learning_curve.png from the run's CSV logs
(output/run if you re-ran the pilot, otherwise the recorded logs in recorded/run)."""
import csv, json, os, sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]   # repo root (all paths are relative to it)
sys.path.insert(0, str(REPO / "common"))
OUT = HERE / "output"; OUT.mkdir(exist_ok=True)   # re-runs write here; recorded results are in recorded/
RUN = OUT / "run" if (OUT / "run" / "evaluations.csv").exists() else HERE / "recorded" / "run"
OUT.mkdir(exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(HERE / ".matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

def rows(name):
    p = RUN / name
    return list(csv.DictReader(p.open(encoding="utf-8"))) if p.exists() else []

evals = defaultdict(list)
for r in rows("evaluations.csv"):
    evals[int(r["step"])].append(float(r["score"]))
episodes = [(int(r["step"]), float(r["score"])) for r in rows("episodes.csv")]
SESS = RUN.parent / "sessions.jsonl"   # the recorded pilot has none: it was killed before writing it
sessions = [json.loads(l) for l in SESS.read_text().splitlines() if l.strip()] if SESS.exists() else []
def smax(key): return max([s[key] for s in sessions] or [float('nan')])   # nan = no session record
status = json.loads((RUN / "status.json").read_text())
progress = rows("progress.csv")
last = progress[-1] if progress else {}

steps = sorted(evals)
final_step = max(steps) if steps else None
lines = ["PILOT 1M - DQN seed 17, cuda, replay 100k, total 1,000,000 decisions",
         "Verbatim notebook code (run_training). Overrides: device=cuda, replay_capacity=100000,",
         "total_steps=1000000, eval_interval=50000, checkpoint_interval=100000. epsilon_decay_steps=80000 (unchanged).",
         f"Status: {status['status']}  step {status['step']:,} / {status['planned_steps']:,}", "",
         "Evaluation (raw game score, epsilon 0.01, held-out seeds):",
         f"{'step':>10}  {'games':>5}  {'mean':>7}  {'min':>6}  {'max':>6}  scores"]
for s in steps:
    sc = evals[s]
    lines.append(f"{s:>10,}  {len(sc):>5}  {np.mean(sc):>7.2f}  {min(sc):>6.0f}  {max(sc):>6.0f}  {', '.join(f'{x:.0f}' for x in sc)}")
if final_step == status["planned_steps"]:
    lines += ["", f"FINAL mean score ({len(evals[final_step])} games at {final_step:,}): {np.mean(evals[final_step]):.2f}"]
wall = sum(s["wall_clock_s"] for s in sessions)
lines += ["",
          f"Wall-clock: {wall/3600:.2f} h over {len(sessions)} session(s)"
          + (f"  (training {float(last['train_elapsed_s'])/3600:.2f} h, evaluation {float(last['eval_elapsed_s'])/3600:.2f} h)" if last else ""),
          f"Peak RAM (Windows peak working set): {smax('peak_working_set_mib')/1024:.2f} GiB (nan = no session record)"
          f"  (sampled RSS peak {smax('peak_rss_mib_sampled')/1024:.2f} GiB)",
          f"Peak GPU memory allocated: {smax('peak_gpu_mem_allocated_mib'):.0f} MiB",
          f"Training episodes completed: {len(episodes)}",
          "Sessions: " + "; ".join(f"{s['start_utc'][:19]} -> {s['status']}" + (f" ({s['error']})" if s.get('error') else "") for s in sessions),
          "", "Plot: learning_curve.png"]
(OUT / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

fig, ax = plt.subplots(figsize=(9, 4.8), constrained_layout=True)
if episodes:
    es, sc = np.array(episodes).T
    ax.plot(es, sc, ".", color="#9aa5b1", ms=3, alpha=.5, label="training episodes (epsilon-greedy)")
    if len(sc) >= 20:
        k = 20
        ax.plot(es[k-1:], np.convolve(sc, np.ones(k)/k, "valid"), color="#52606d", lw=1.2, label=f"training, {k}-episode mean")
if steps:
    m = [np.mean(evals[s]) for s in steps]
    sd = [np.std(evals[s]) for s in steps]
    ax.errorbar(steps, m, yerr=sd, color="#1f6feb", marker="o", lw=2, capsize=3, label="evaluation mean ± sd")
ax.axhline(0, color="#999", lw=.8, ls="--")
ax.set(xlabel="decisions", ylabel="game score (-21 to 21)", ylim=(-22, 22),
       title="DQN pilot, seed 17, 1M decisions (RTX 5050)")
ax.legend(loc="lower right", fontsize=9)
fig.savefig(OUT / "learning_curve.png", dpi=150)
print("\n".join(lines))
