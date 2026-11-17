"""See what a trained Pong DQN agent is "thinking".

Plays Pong against the built-in opponent with the notebook's own PongEnv (same sticky
actions, AtariPreprocessing and 4-frame stack as training) and evaluation-style acting
(epsilon = config.eval_epsilon, same held-out seeds/RNG as evaluate()). Shows a live
window: game on the left, the 6 Q-values on the right (chosen action highlighted).
Every decision is logged with Q-values and positions read from Atari RAM, then plots
and a text summary are saved to demos/explain_output/<agent>_<time>/.

  python explain_ai.py                                   # models/dqn_configA_seed17_5.0M.pt, 1 game, live window
  python explain_ai.py ../models/dqn_oldsettings_seed17_0.6M.pt   # any model / notebook policy.pt
  python explain_ai.py --games 5 --no-window             # analysis only, as fast as possible
Keys in the window: SPACE = pause/resume, q = stop early (results still saved).

RAM addresses (verified against screen pixels by experiments/04_diagnosis/verify_ram.py, R^2 >= 0.9988):
  51 our paddle y, 50 opponent paddle y, 54 ball y, 49 ball x. Converted to screen
  pixels (y grows downwards; court rows 34-193). Our paddle is the green one on the right.
"""
import argparse, copy, csv, importlib.util, json, os, sys, time, warnings
from dataclasses import replace
from datetime import datetime
from pathlib import Path

sys.dont_write_bytecode = True
warnings.filterwarnings("ignore", category=UserWarning)
HERE = Path(__file__).resolve().parent
REPO = HERE.parent   # repo root
DEFAULT_POLICY = REPO / "models" / "dqn_configA_seed17_5.0M.pt"

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("policy", nargs="?", default=str(DEFAULT_POLICY), help="notebook policy.pt (read-only)")
ap.add_argument("--games", type=int, default=1)
ap.add_argument("--start-index", type=int, default=0, help="first evaluation game index (held-out seed)")
ap.add_argument("--speed", type=float, default=1.0, help="1 = real Atari speed; 0 = no pacing")
ap.add_argument("--no-window", action="store_true", help="no live window (fast analysis)")
ap.add_argument("--device", default=None)
ap.add_argument("--out", default=None, help="output folder (default explain_output/<agent>_<time>)")
args = ap.parse_args()

os.environ.setdefault("MPLCONFIGDIR", str(HERE / ".matplotlib"))
import matplotlib
matplotlib.use("Agg" if args.no_window else "TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import torch

spec = importlib.util.spec_from_file_location("pong_train_extract", REPO / "common" / "pong_train_extract.py")
nb = importlib.util.module_from_spec(spec)
sys.modules["pong_train_extract"] = nb
spec.loader.exec_module(nb)

# ------------------------------------------------------------------ agent
policy_path = Path(args.policy).resolve()
saved = torch.load(policy_path, map_location="cpu", weights_only=True)   # read-only, no pickled code
if not {"config", "online"} <= set(saved):
    sys.exit(f"{policy_path} is not a notebook policy.pt (needs 'config' and 'online').")
device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
config = replace(nb.Config(**{k: v for k, v in saved["config"].items()
                              if k in nb.Config.__dataclass_fields__}), device=device)
if saved["config"].get("network", "small") == "nature":          # config A models (common/pong_train_diag.py)
    sys.path.insert(0, str(REPO / "common"))
    from pong_train_diag import NatureQNetwork
    network = NatureQNetwork(6).to(device)
else:
    network = nb.QNetwork(6).to(device)
try:
    network.load_state_dict(saved["online"])
except RuntimeError as e:
    sys.exit(f"Weights don't match the notebook's standard QNetwork (guided policies use a different "
             f"network): {e}")
network.eval()
step_trained = saved.get("step", 0)
agent_name = f"{config.method}_seed{config.seed}_{step_trained // 1000}k"
print(f"Agent: {policy_path}\n  {config.method}, seed {config.seed}, trained {step_trained:,} decisions, "
      f"epsilon {config.eval_epsilon}, device {device}")

# ------------------------------------------------------------------ env (identical wrappers; rgb frames for display)
env = nb.PongEnv(config, video=True)
meanings = env.raw.unwrapped.get_action_meanings()
MOVE = {i: (-1 if m.startswith("RIGHT") else 1 if m.startswith("LEFT") else 0) for i, m in enumerate(meanings)}
# -1 = paddle moves up the screen, +1 = down, 0 = stays (verified: RIGHT lowers RAM[51])
labels = [f"{m}\n({'up' if MOVE[i] < 0 else 'down' if MOVE[i] > 0 else 'stay'})" for i, m in enumerate(meanings)]
UP = [i for i in MOVE if MOVE[i] < 0]; DOWN = [i for i in MOVE if MOVE[i] > 0]

TOP, BOTTOM = 34, 194                       # court rows on screen
COURT_MID = (TOP + BOTTOM - 1) / 2
PADDLE_X = 140                              # our paddle's left edge (screen x)
GREEN, WHITE = np.array([92, 186, 92]), np.array([236, 236, 236])
def ram_positions(ram):
    r = ram.astype(float)
    return {"paddle_y": 0.948 * r[51] - 0.03, "opp_y": 0.982 * r[50] - 5.17,
            "ball_y": 0.994 * r[54] - 11.66, "ball_x": r[49] - 48.5}
def screen_check(frame):
    court = frame[TOP:BOTTOM]
    ys, xs = np.nonzero(np.all(court == WHITE, axis=2))
    ball = (ys.mean() + TOP, xs.mean()) if len(ys) else None
    gy, _ = np.nonzero(np.all(court[:, 130:160] == GREEN, axis=2))
    paddle = gy.mean() + TOP if len(gy) >= 14 else None      # only when fully on screen
    return ball, paddle

# ------------------------------------------------------------------ live window
state = {"paused": False, "quit": False}
if not args.no_window:
    plt.ion()
    fig, (ax_game, ax_q) = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={"width_ratios": [1, 1.25]})
    fig.canvas.manager.set_window_title(f"Pong agent explained - {agent_name}")
    img = ax_game.imshow(np.zeros((210, 160, 3), np.uint8))
    ax_game.set_axis_off()
    game_title = ax_game.set_title("")
    bars = ax_q.bar(range(6), np.zeros(6), color="#c9d3dd", edgecolor="#52606d")
    ax_q.set_xticks(range(6), labels, fontsize=9)
    ax_q.set_ylabel("Q-value  (expected discounted future points)")
    ax_q.axhline(0, color="#999", lw=.8)
    q_title = ax_q.set_title("")
    help_text = fig.text(0.5, 0.01, "SPACE pause/resume   q stop", ha="center", color="#666")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    def on_key(e):
        if e.key == " ": state["paused"] = not state["paused"]
        elif e.key in ("q", "escape"): state["quit"] = True
    fig.canvas.mpl_connect("key_press_event", on_key)
    fig.canvas.mpl_connect("close_event", lambda e: state.update(quit=True))
    plt.show(block=False)
    q_lo, q_hi = [np.inf], [-np.inf]
    def show(frame, q, action, explored, score, k, direction):
        img.set_data(frame)
        for i, b in enumerate(bars):
            b.set_height(q[i])
            b.set_color("#1f6feb" if i == action else "#c9d3dd"); b.set_edgecolor("#52606d")
        q_lo[0], q_hi[0] = min(q_lo[0], q.min()), max(q_hi[0], q.max())
        pad = max(0.05, 0.1 * (q_hi[0] - q_lo[0]))
        ax_q.set_ylim(min(q_lo[0] - pad, -pad), max(q_hi[0] + pad, pad))
        game_title.set_text(f"agent {score[0]} : {score[1]} opponent   (agent = green, right)")
        q_title.set_text(f"decision {k}: {meanings[action]}{'  [random, epsilon]' if explored else ''}"
                         f"   max Q {q.max():+.3f}   ball {direction}")
        fig.canvas.draw_idle()
        fig.canvas.start_event_loop(0.001)
        while state["paused"] and not state["quit"]:
            help_text.set_text("PAUSED - press SPACE to resume")
            fig.canvas.draw_idle(); fig.canvas.start_event_loop(0.05)
        help_text.set_text("SPACE pause/resume   q stop")

# ------------------------------------------------------------------ play + log
rows, game_scores = [], []
frame_s = 1 / (60 * args.speed) if args.speed > 0 and not args.no_window else 0.0
try:
    for game in range(args.start_index, args.start_index + args.games):
        eval_seed = 100_000 + config.seed * 100 + game                 # as in evaluate()
        rng = np.random.default_rng(eval_seed + 1_000_000)
        obs, _ = env.reset(seed=eval_seed)
        score, prev_bx, k = [0, 0], None, 0
        clock = time.perf_counter()
        print(f"\nGame {game - args.start_index + 1} (eval seed {eval_seed})")
        done = False
        while not done and not state["quit"]:
            frame = env.raw.render()
            pos = ram_positions(env.raw.unwrapped.ale.getRAM())
            ball_px, paddle_px = screen_check(frame)
            visible = ball_px is not None
            if visible and prev_bx is not None:
                dx = pos["ball_x"] - prev_bx
                direction = "towards" if dx > 0 else "away" if dx < 0 else "still"
            else:
                direction = "none"
            prev_bx = pos["ball_x"] if visible else None
            with torch.inference_mode():
                q = network(torch.as_tensor(obs[None], device=device))[0].float().cpu().numpy()
            explored = bool(copy.deepcopy(rng).random() < config.eval_epsilon)   # peek: same draw choose_action makes
            action = nb.choose_action(network, obs, config.eval_epsilon, rng, env.action_count, device)
            if not args.no_window:
                show(frame, q, action, explored, score, k, direction)
            obs, reward, term, trunc, info = env.step(action)
            done = term or trunc
            if reward > 0: score[0] += 1; print(f"  agent scores    {score[0]:2d}:{score[1]:2d}", flush=True)
            if reward < 0: score[1] += 1; print(f"  opponent scores {score[0]:2d}:{score[1]:2d}", flush=True)
            rows.append({"game": game, "decision": k, **{f"q_{meanings[i]}": float(q[i]) for i in range(6)},
                         "max_q": float(q.max()), "action": meanings[action], "action_id": action,
                         "random_action": explored, "paddle_y": pos["paddle_y"], "opp_y": pos["opp_y"],
                         "ball_visible": visible,
                         "ball_x": pos["ball_x"] if visible else None, "ball_y": pos["ball_y"] if visible else None,
                         "ball_direction": direction, "reward": reward,
                         "agent_score": score[0], "opponent_score": score[1],
                         "check_ball_err_px": (abs(ball_px[0] - pos["ball_y"]) + abs(ball_px[1] - pos["ball_x"])) if visible else None,
                         "check_paddle_err_px": abs(paddle_px - pos["paddle_y"]) if paddle_px is not None else None})
            k += 1
            if frame_s:
                clock += info["actual_game_frames"] * frame_s
                delay = clock - time.perf_counter()
                if delay > 0: time.sleep(delay)
                else: clock = time.perf_counter()
        game_scores.append(tuple(score))
        print(f"FINAL: agent {score[0]} : {score[1]} opponent" + ("  (stopped early)" if state["quit"] else ""))
        if state["quit"]:
            break
finally:
    env.close()
if not rows:
    sys.exit("No decisions recorded.")

# ------------------------------------------------------------------ analysis
out = Path(args.out) if args.out else HERE / "explain_output" / f"{agent_name}_{datetime.now():%Y%m%d_%H%M%S}"
out.mkdir(parents=True, exist_ok=True)
with (out / "decisions.csv").open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

N = len(rows)
py = np.array([r["paddle_y"] for r in rows])
vis = np.array([r["ball_visible"] for r in rows])
by = np.array([r["ball_y"] if r["ball_visible"] else np.nan for r in rows], float)
bx = np.array([r["ball_x"] if r["ball_visible"] else np.nan for r in rows], float)
dirs = np.array([r["ball_direction"] for r in rows])
acts = np.array([r["action_id"] for r in rows])
rand = np.array([r["random_action"] for r in rows])
Q = np.array([[r[f"q_{m}"] for m in meanings] for r in rows])
maxq = Q.max(1)
rew = np.array([r["reward"] for r in rows])
games_col = np.array([r["game"] for r in rows])
won, lost = np.nonzero(rew > 0)[0], np.nonzero(rew < 0)[0]
S = {"agent": agent_name, "policy": str(policy_path), "decisions": N, "games": game_scores}

# RAM vs screen agreement during this run
be = np.array([r["check_ball_err_px"] for r in rows if r["check_ball_err_px"] is not None])
pe = np.array([r["check_paddle_err_px"] for r in rows if r["check_paddle_err_px"] is not None])
S["ram_check"] = {"ball_median_err_px": float(np.median(be)) if len(be) else None,
                  "ball_p95_err_px": float(np.percentile(be, 95)) if len(be) else None,
                  "paddle_median_err_px": float(np.median(pe)) if len(pe) else None,
                  "paddle_p95_err_px": float(np.percentile(pe, 95)) if len(pe) else None}

# Where does the paddle sit?
third = (BOTTOM - TOP) / 3
def zones(y):
    y = np.asarray(y)
    return {"top_third": float(np.mean(y < TOP + third)), "middle_third": float(np.mean((y >= TOP + third) & (y < TOP + 2 * third))),
            "bottom_third": float(np.mean(y >= TOP + 2 * third)), "median_y": float(np.median(y)) if len(y) else None}
groups = {"ball moving away": dirs == "away", "ball moving towards us": dirs == "towards", "no ball in play": dirs == "none"}
S["paddle_position"] = {g: {"n": int(m.sum()), **zones(py[m])} for g, m in groups.items() if m.sum()}
S["paddle_range_seen"] = [float(py.min()), float(py.max())]

# Actions
S["action_share"] = {meanings[i]: float(np.mean(acts == i)) for i in range(6)}
S["action_share_by_context"] = {g: {"up": float(np.isin(acts[m], UP).mean()), "down": float(np.isin(acts[m], DOWN).mean()),
                                    "stay": float(np.isin(acts[m], [i for i in MOVE if MOVE[i] == 0]).mean())}
                                for g, m in groups.items() if m.sum()}
S["random_actions"] = int(rand.sum())

# Empirical paddle reach: the furthest the paddle actually travelled in n decisions this run
# (the paddle has momentum, so this is more honest than a single top speed).
MAXN = 40
reach = np.zeros(MAXN + 1)
for n in range(1, MAXN + 1):
    same = games_col[n:] == games_col[:-n]
    if same.any(): reach[n] = np.abs(py[n:] - py[:-n])[same].max()
reach = np.maximum.accumulate(reach)
def can_reach(dist, n): return dist <= reach[min(int(n), MAXN)]

# Approach segments: consecutive decisions with the ball moving towards us; a point ends one.
segs, cur = [], []
for i in range(N):
    if dirs[i] == "towards" and (not cur or (i == cur[-1] + 1 and games_col[i] == games_col[cur[-1]])):
        cur.append(i)
    else:
        if len(cur) >= 3: segs.append(cur)
        cur = [i] if dirs[i] == "towards" else []
    if rew[i] != 0 and cur:          # the point is decided here: close the approach
        if len(cur) >= 3: segs.append(cur)
        cur = []
if len(cur) >= 3: segs.append(cur)
approach = []
for s in segs:
    end = s[-1]
    at_paddle = [i for i in s if bx[i] >= PADDLE_X - 4]
    hit = at_paddle[0] if at_paddle else end          # when the ball reaches our paddle's column
    arrival = by[hit]
    outcome = "lost" if np.any(rew[s[0]:end + 3] < 0) else "returned_or_won"
    n_avail = hit - s[0] + 1
    approach.append({"start": s[0], "hit": hit, "end": end, "decisions": n_avail, "arrival_y": float(arrival),
                     "paddle_y_at_start": float(py[s[0]]), "paddle_y_at_hit": float(py[hit]),
                     "reachable_from_paddle": bool(can_reach(abs(arrival - py[s[0]]), n_avail)),
                     "reachable_from_mid": bool(can_reach(abs(arrival - COURT_MID), n_avail)),
                     "outcome": outcome})
if approach:
    A = approach
    L = [a for a in A if a["outcome"] == "lost"]
    full_court = next((n for n in range(1, MAXN + 1) if reach[n] >= (BOTTOM - TOP) - 30), None)
    S["approaches"] = {"n": len(A), "lost_after_approach": len(L),
                       "decisions_for_paddle_to_cross_court": full_court,
                       "paddle_max_px_in_1_decision": float(reach[1]),
                       "mean_decisions_ball_takes_to_arrive": float(np.mean([a["decisions"] for a in A])),
                       "reachable_from_where_paddle_was": float(np.mean([a["reachable_from_paddle"] for a in A])),
                       "reachable_if_paddle_had_started_mid_court": float(np.mean([a["reachable_from_mid"] for a in A])),
                       "lost_points_reachable_from_paddle": float(np.mean([a["reachable_from_paddle"] for a in L])) if L else None,
                       "lost_points_reachable_from_mid": float(np.mean([a["reachable_from_mid"] for a in L])) if L else None,
                       "lost_points_miss_px_median": float(np.median([abs(a["arrival_y"] - a["paddle_y_at_hit"]) for a in L])) if L else None,
                       "lost_points_paddle_start_bottom_third": float(np.mean([a["paddle_y_at_start"] >= TOP + 2 * (BOTTOM - TOP) / 3 for a in L])) if L else None,
                       "lost_points_arrival_top_half": float(np.mean([a["arrival_y"] < COURT_MID for a in L])) if L else None}

# Does it track / anticipate? Compare moves with where the ball IS vs where it WILL arrive.
DEAD = 8
agree_now, agree_arr, decisive = [], [], 0
pref, tgt_now, tgt_arr = [], [], []
early_to_arrival, start_moves = [], []
for a in approach:
    half = a["start"] + max(1, (a["hit"] - a["start"]) // 2)
    for i in range(a["start"], half):
        m = MOVE[int(acts[i])]
        d_arr = a["arrival_y"] - py[i]
        if not rand[i] and abs(d_arr) > DEAD:
            early_to_arrival.append(m == np.sign(d_arr))     # moving the right way early on (stay counts as no)
    # how many decisions after the ball turns towards us before the paddle first heads the right way?
    first = next((i - a["start"] for i in range(a["start"], a["hit"] + 1)
                  if MOVE[int(acts[i])] == np.sign(a["arrival_y"] - py[i]) and abs(a["arrival_y"] - py[i]) > DEAD), None)
    if abs(a["arrival_y"] - py[a["start"]]) > 3 * DEAD and first is not None:
        start_moves.append(first / max(1, a["hit"] - a["start"]))
    for i in range(a["start"], a["hit"] + 1):
        if rand[i]: continue
        d_now, d_arr = by[i] - py[i], a["arrival_y"] - py[i]
        pref.append(Q[i, DOWN].max() - Q[i, UP].max()); tgt_now.append(d_now); tgt_arr.append(d_arr)
        m = MOVE[int(acts[i])]
        if m == 0: continue
        if abs(d_now - d_arr) > 15 and np.sign(d_now) != np.sign(d_arr) and abs(d_now) > DEAD and abs(d_arr) > DEAD:
            decisive += 1
            agree_now.append(m == np.sign(d_now)); agree_arr.append(m == np.sign(d_arr))
moves_all = [(MOVE[int(acts[i])], by[i] - py[i]) for i in range(N) if vis[i] and not rand[i] and MOVE[int(acts[i])] != 0 and abs(by[i] - py[i]) > DEAD]
S["tracking"] = {
    "corr_paddle_y_ball_y_all_visible": float(np.corrcoef(py[vis], by[vis])[0, 1]) if vis.sum() > 2 else None,
    "corr_paddle_y_ball_y_when_ball_approaching": float(np.corrcoef(py[dirs == "towards"], by[dirs == "towards"])[0, 1]) if (dirs == "towards").sum() > 2 else None,
    "moves_towards_current_ball_y_share": float(np.mean([m == np.sign(d) for m, d in moves_all])) if moves_all else None,
    "corr_Q_down_minus_up_vs_ball_below_now": float(np.corrcoef(pref, tgt_now)[0, 1]) if len(pref) > 2 else None,
    "corr_Q_down_minus_up_vs_arrival_below": float(np.corrcoef(pref, tgt_arr)[0, 1]) if len(pref) > 2 else None,
    "first_half_of_approach_moving_towards_arrival_share": float(np.mean(early_to_arrival)) if early_to_arrival else None,
    "first_correct_move_fraction_of_approach_median": float(np.median(start_moves)) if start_moves else None,
    "conflict_moves_n": decisive,
    "conflict_moves_follow_current_ball": float(np.mean(agree_now)) if agree_now else None,
    "conflict_moves_head_to_arrival": float(np.mean(agree_arr)) if agree_arr else None}

# When does it "expect" to lose? max-Q aligned on points
W = 60
points = np.sort(np.concatenate([won, lost]))
def aligned(idx):
    """max-Q in the W decisions before each point, cut off at the previous point (NaN before it)."""
    M = []
    for i in idx:
        prev = points[points < i]
        start = max(i - W, (prev[-1] + 1) if len(prev) else 0, int(np.argmax(games_col == games_col[i])))
        row = np.full(W + 1, np.nan)
        row[W - (i - start):] = maxq[start:i + 1]
        M.append(row)
    return np.array(M)
def nanmean0(M):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(M, 0)
AL, AW = aligned(lost), aligned(won)
base = float(np.median(maxq))
S["value"] = {"median_max_q": base, "max_q_range": [float(maxq.min()), float(maxq.max())],
              "lost_points_used": len(AL), "won_points_used": len(AW)}
offs = [-40, -30, -20, -15, -10, -5, -2, 0]
if len(AL):
    mL = nanmean0(AL)
    S["value"]["mean_max_q_before_lost_point"] = {str(o): float(mL[W + o]) for o in offs if np.isfinite(mL[W + o])}
    if len(AW):
        mW = nanmean0(AW)
        S["value"]["mean_max_q_before_won_point"] = {str(o): float(mW[W + o]) for o in offs if np.isfinite(mW[W + o])}
    # when does the value start falling before a lost point? (vs its level 30 decisions earlier)
    ref = mL[W - 30] if np.isfinite(mL[W - 30]) else np.nanmax(mL)
    drop = mL[W] - ref
    k_half = next((o for o in range(-30, 1) if np.isfinite(mL[W + o]) and mL[W + o] - ref <= drop / 2), None) if drop < 0 else None
    S["value"]["lost_points_half_drop_decisions_before"] = (-k_half if k_half is not None else None)

(out / "summary.json").write_text(json.dumps(S, indent=2, default=float))

# ------------------------------------------------------------------ plots
plt.ioff()
C = {"away": "#1f6feb", "towards": "#d1242f", "none": "#8c959f"}
def court_axis(ax, which="y"):
    (ax.set_ylim if which == "y" else ax.set_xlim)(BOTTOM, TOP)
    ax.axhline(COURT_MID, color="#999", ls="--", lw=.8) if which == "y" else ax.axvline(COURT_MID, color="#999", ls="--", lw=.8)

# 1 paddle position histogram
fig1, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
bins = np.linspace(TOP, BOTTOM, 33)
for g, key in (("ball moving away", "away"), ("ball moving towards us", "towards"), ("no ball in play", "none")):
    m = groups[g]
    if m.sum(): ax.hist(py[m], bins=bins, alpha=.55, color=C[key], label=f"{g} (n={m.sum()})", density=True)
ax.set_xlim(TOP, BOTTOM); ax.axvline(COURT_MID, color="#333", ls="--", lw=1, label="middle of court")
ax.set_xlabel("our paddle centre, screen y  (left = TOP of court, right = BOTTOM)"); ax.set_ylabel("share of decisions (density)")
ax.set_title(f"Where our paddle sits - {agent_name}"); ax.legend()
fig1.savefig(out / "1_paddle_position_hist.png", dpi=140)

# 2 action frequency
fig2, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
xs = np.arange(6); wdt = .27
for j, (g, key) in enumerate((("ball moving away", "away"), ("ball moving towards us", "towards"), ("no ball in play", "none"))):
    m = groups[g]
    if m.sum(): ax.bar(xs + (j - 1) * wdt, [np.mean(acts[m] == i) for i in range(6)], wdt, color=C[key], label=g)
ax.plot(xs, [np.mean(acts == i) for i in range(6)], "ko", label="overall")
ax.set_xticks(xs, labels); ax.set_ylabel("share of decisions"); ax.legend(fontsize=9)
ax.set_title(f"How often each action is chosen - {agent_name}")
fig2.savefig(out / "2_action_frequency.png", dpi=140)

# 3 max-Q over time with points
fig3, (ax, ax2) = plt.subplots(2, 1, figsize=(12, 7), constrained_layout=True, gridspec_kw={"height_ratios": [2, 1]})
ax.plot(maxq, color="#52606d", lw=.8, label="highest Q-value")
ax.scatter(won, maxq[won], color="#1a7f37", marker="^", s=60, zorder=3, label="point WON")
ax.scatter(lost, maxq[lost], color="#d1242f", marker="v", s=40, zorder=3, label="point LOST")
for g in np.unique(games_col)[1:]:
    ax.axvline(np.argmax(games_col == g), color="#999", ls=":")
ax.set_xlabel("decision"); ax.set_ylabel("max Q"); ax.legend(loc="lower left", fontsize=9)
ax.set_title("What the agent expects: highest Q-value over the game (points marked when they happen)")
t = np.arange(-W, 1)
for M, col, lab in ((AL, "#d1242f", "before LOST points"), (AW, "#1a7f37", "before WON points")):
    if len(M):
        ax2.plot(t, M.T, color=col, alpha=.12, lw=.7)
        ax2.plot(t, nanmean0(M), color=col, lw=2.2, label=f"mean, {lab} (n={len(M)})")
ax2.axhline(base, color="#999", ls="--", lw=.8, label="median over game")
ax2.set_xlabel("decisions before the point (0 = point scored; each line starts after the previous point)"); ax2.set_ylabel("max Q"); ax2.legend(fontsize=9)
fig3.savefig(out / "3_max_q_over_time.png", dpi=140)

# 4 paddle y vs ball y
fig4, ax = plt.subplots(figsize=(12, 4.8), constrained_layout=True)
ax.plot(py, color="#1a7f37", lw=1.2, label="our paddle y")
for key in ("towards", "away"):
    m = dirs == key
    ax.scatter(np.nonzero(m)[0], by[m], s=3, color=C[key], label=f"ball y ({'moving towards us' if key=='towards' else 'moving away'})")
ax.scatter(lost, np.full(len(lost), TOP + 2), marker="v", color="#d1242f", s=25, label="point lost")
ax.scatter(won, np.full(len(won), TOP + 2), marker="^", color="#1a7f37", s=40, label="point won")
court_axis(ax); ax.set_xlabel("decision"); ax.set_ylabel("screen y  (top of plot = TOP of court)")
ax.set_title("Does the paddle follow the ball?"); ax.legend(loc="lower right", fontsize=8, ncol=3)
fig4.savefig(out / "4_paddle_vs_ball_y.png", dpi=140)

# 5 reachability of each approaching ball
if approach:
    fig5, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for a in approach:
        col = "#d1242f" if a["outcome"] == "lost" else "#1a7f37"
        ax.scatter(a["paddle_y_at_start"], a["arrival_y"], color=col, s=30, alpha=.8)
    ax.plot([TOP, BOTTOM], [TOP, BOTTOM], color="#999", lw=.8)
    ax.set_xlim(BOTTOM, TOP); ax.set_ylim(BOTTOM, TOP)
    ax.set_xlabel("paddle y when the ball starts coming towards us  (right = TOP)")
    ax.set_ylabel("ball y when it arrives  (up = TOP)")
    ax.scatter([], [], color="#d1242f", label="point lost"); ax.scatter([], [], color="#1a7f37", label="returned / won")
    ax.set_title("Each incoming ball: where we started vs where it arrived"); ax.legend()
    fig5.savefig(out / "5_incoming_ball_reach.png", dpi=140)

# ------------------------------------------------------------------ text summary
def pct(x): return "n/a" if x is None else f"{100 * x:.0f}%"
def num(x, f="{:.2f}"): return "n/a" if x is None else f.format(x)
pp = S["paddle_position"]; tr = S["tracking"]; ap_ = S.get("approaches", {}); v = S["value"]
lines = [f"EXPLAIN - {agent_name}", f"policy: {policy_path}", f"games: " + ", ".join(f"{a}:{b}" for a, b in game_scores)
         + f"   decisions logged: {N}   random (epsilon) actions: {S['random_actions']}", "",
         "RAM positions vs screen pixels this run: ball median err "
         f"{num(S['ram_check']['ball_median_err_px'])} px (p95 {num(S['ram_check']['ball_p95_err_px'])}), paddle median "
         f"{num(S['ram_check']['paddle_median_err_px'])} px (p95 {num(S['ram_check']['paddle_p95_err_px'])})", "",
         f"WHERE THE PADDLE SITS (court rows {TOP}-{BOTTOM - 1}, middle = {COURT_MID:.0f}):"]
for g, z in pp.items():
    lines.append(f"  {g:24s} n={z['n']:5d}  top {pct(z['top_third'])}  middle {pct(z['middle_third'])}  "
                 f"bottom {pct(z['bottom_third'])}   median y {z['median_y']:.0f}")
lines += ["", "ACTIONS (share of decisions):  " + "  ".join(f"{m} {pct(s)}" for m, s in S["action_share"].items())]
for g, s in S["action_share_by_context"].items():
    lines.append(f"  {g:24s} up {pct(s['up'])}  down {pct(s['down'])}  stay {pct(s['stay'])}")
lines += ["", "TRACKING / ANTICIPATION:",
          f"  correlation paddle y vs ball y (ball approaching): {num(tr['corr_paddle_y_ball_y_when_ball_approaching'])}",
          f"  moves go towards the ball's CURRENT height: {pct(tr['moves_towards_current_ball_y_share'])} of moves",
          f"  Q(down)-Q(up) vs ball below paddle now: r={num(tr['corr_Q_down_minus_up_vs_ball_below_now'])};"
          f"  vs ball will ARRIVE below: r={num(tr['corr_Q_down_minus_up_vs_arrival_below'])}",
          f"  first half of each incoming ball: moving towards where it WILL arrive {pct(tr['first_half_of_approach_moving_towards_arrival_share'])} of decisions",
          f"  first move in the right direction comes {pct(tr['first_correct_move_fraction_of_approach_median'])} of the way through the approach (median)",
          f"  when 'chase current ball' and 'go to arrival point' disagree ({tr['conflict_moves_n']} moves): "
          f"follows current ball {pct(tr['conflict_moves_follow_current_ball'])}, heads to arrival {pct(tr['conflict_moves_head_to_arrival'])}"]
if ap_:
    lines += ["", "INCOMING BALLS:",
              f"  {ap_['n']} approaches, {ap_['lost_after_approach']} ended in a lost point; ball takes ~{ap_['mean_decisions_ball_takes_to_arrive']:.0f} decisions"
              f" to reach our paddle; paddle crossed (most of) the court in {ap_['decisions_for_paddle_to_cross_court']} decisions"
              f" (max {ap_['paddle_max_px_in_1_decision']:.0f} px in one)",
              f"  reachable from where the paddle actually was: {pct(ap_['reachable_from_where_paddle_was'])};"
              f" if it had started mid-court: {pct(ap_['reachable_if_paddle_had_started_mid_court'])}",
              f"  of the LOST points: reachable from paddle {pct(ap_['lost_points_reachable_from_paddle'])}, from mid-court"
              f" {pct(ap_['lost_points_reachable_from_mid'])}; median miss {num(ap_['lost_points_miss_px_median'], '{:.0f}')} px;"
              f" paddle started in bottom third {pct(ap_['lost_points_paddle_start_bottom_third'])}; ball arrived in top half {pct(ap_['lost_points_arrival_top_half'])}"]
lines += ["", "VALUE (highest Q):", f"  median {v['median_max_q']:+.3f}, range {v['max_q_range'][0]:+.3f} .. {v['max_q_range'][1]:+.3f}"]
if "mean_max_q_before_lost_point" in v:
    lines.append("  mean before LOST points (decisions before: value): " + "  ".join(f"{k}: {x:+.2f}" for k, x in v["mean_max_q_before_lost_point"].items()))
if "mean_max_q_before_won_point" in v:
    lines.append("  mean before WON points:                            " + "  ".join(f"{k}: {x:+.2f}" for k, x in v["mean_max_q_before_won_point"].items()))
if v.get("lost_points_half_drop_decisions_before") is not None:
    lines.append(f"  before a lost point, half of the value drop has happened ~{v['lost_points_half_drop_decisions_before']} decisions before it")
lines += ["", "Files: decisions.csv, summary.json, 1_paddle_position_hist.png, 2_action_frequency.png,",
          "       3_max_q_over_time.png, 4_paddle_vs_ball_y.png, 5_incoming_ball_reach.png"]
(out / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n" + "\n".join(lines))
print(f"\nSaved to {out}")
if not args.no_window:
    plt.close("all")
