"""Standard behaviour metrics for comparing configs/checkpoints.

Runs demos/explain_ai.py --no-window on a policy (same held-out eval games, epsilon 0.01)
and appends one row to output/metrics_table.csv:
  mean score, paddle bottom-third share (ball approaching / no ball), tracking correlation
  (paddle y vs ball y while approaching), Q(down)-Q(up) vs arrival point correlation,
  share of first-half-of-approach decisions heading to the arrival point,
  median Q action gap (best minus 2nd-best: how strongly it prefers an action).

  python compare_metrics.py <label> <policy.pt> [--games 3]
  e.g. python compare_metrics.py configA_5.0M ../../models/dqn_configA_seed17_5.0M.pt
"""
import argparse, csv, json, subprocess, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]   # repo root (all paths are relative to it)
sys.path.insert(0, str(REPO / "common"))
OUT = HERE / "output"; OUT.mkdir(exist_ok=True)   # re-runs write here; recorded results are in recorded/
EXPLAIN = REPO / "demos" / "explain_ai.py"

ap = argparse.ArgumentParser(); ap.add_argument("label"); ap.add_argument("policy"); ap.add_argument("--games", type=int, default=3)
a = ap.parse_args()
out = OUT / "metrics" / a.label
subprocess.run([sys.executable, "-I", "-B", str(EXPLAIN), a.policy, "--no-window", "--games", str(a.games),
                "--out", str(out)], check=True, stdout=subprocess.DEVNULL)
S = json.loads((out / "summary.json").read_text())
with (out / "decisions.csv").open(encoding="utf-8") as f:
    rows = list(csv.DictReader(f))
Q = np.sort(np.array([[float(r[k]) for k in r if k.startswith("q_")] for r in rows]), 1)
pp, tr = S["paddle_position"], S["tracking"]
row = {"label": a.label, "policy": a.policy, "games": a.games,
       "mean_score": float(np.mean([x - y for x, y in S["games"]])),
       "bottom_third_ball_approaching": pp.get("ball moving towards us", {}).get("bottom_third"),
       "bottom_third_no_ball": pp.get("no ball in play", {}).get("bottom_third"),
       "tracking_corr_approaching": tr["corr_paddle_y_ball_y_when_ball_approaching"],
       "Qdown_minus_Qup_vs_arrival_r": tr["corr_Q_down_minus_up_vs_arrival_below"],
       "first_half_heading_to_arrival": tr["first_half_of_approach_moving_towards_arrival_share"],
       "median_action_gap_top2": float(np.median(Q[:, -1] - Q[:, -2])),
       "lost_points_reachable_from_paddle": S.get("approaches", {}).get("lost_points_reachable_from_paddle")}
OUT.mkdir(exist_ok=True)
table = OUT / "metrics_table.csv"
new = not table.exists()
with table.open("a", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(row)); new and w.writeheader(); w.writerow(row)
print(json.dumps(row, indent=1))
