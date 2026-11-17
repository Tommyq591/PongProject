"""Watch one of our trained agents (any file in models/) play Pong live in a window.

The opponent (left, orange) is the game's built-in computer player; our AI is GREEN (right).
Uses the verbatim notebook code in common/pong_train_extract.py:
  * PongEnv  - same sticky actions, AtariPreprocessing (frame skip 4, 84x84 grey,
               noop_max 30) and 4-frame stack as training. The ONLY difference is
               render_mode="human" so a window opens.
  * QNetwork (old settings) or NatureQNetwork (config A) - picked from the model's saved config,
               weights loaded read-only (weights_only=True).
  * choose_action with epsilon = config.eval_epsilon (0.01), seeded like evaluate().
Nothing is written to disk.

Usage (from anywhere):
  python watch_ai.py                                          # 5.0M config A model, one game to 21
  python watch_ai.py ../models/dqn_oldsettings_seed17_0.6M.pt  # any model file
  python watch_ai.py --games 3 --speed 2                      # several games, 2x speed (0 = max)
  python watch_ai.py --no-window                              # headless: just print the scores
"""
import argparse, importlib.util, sys, time
from dataclasses import replace
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
REPO = HERE.parent   # repo root
sys.path.insert(0, str(REPO / "common"))

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("model", nargs="?", default=str(REPO / "models" / "dqn_configA_seed17_5.0M.pt"),
                help="model file (default: models/dqn_configA_seed17_5.0M.pt)")
ap.add_argument("--no-window", action="store_true", help="no game window, no pacing (quick check)")
ap.add_argument("--games", type=int, default=1)
ap.add_argument("--start-index", type=int, default=0, help="evaluation game index (selects held-out seed)")
ap.add_argument("--speed", type=float, default=1.0, help="1 = real Atari speed (60 frames/s)")
ap.add_argument("--device", default=None, help="cuda or cpu (default: cuda if available)")
args = ap.parse_args()

spec = importlib.util.spec_from_file_location("pong_train_extract", REPO / "common" / "pong_train_extract.py")
nb = importlib.util.module_from_spec(spec)
sys.modules["pong_train_extract"] = nb
spec.loader.exec_module(nb)
import numpy as np, torch

# ---- load the agent (read-only; weights_only=True = plain tensors/dicts, no pickled code)
POLICY = Path(args.model).resolve()
saved = torch.load(POLICY, map_location="cpu", weights_only=True)
device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
config = replace(nb.Config(**{k: v for k, v in saved["config"].items() if k in nb.Config.__dataclass_fields__}),
                 device=device)
if saved["config"].get("network", "small") == "nature":     # config A models
    from pong_train_diag import NatureQNetwork
    network = NatureQNetwork(6).to(device)
else:                                                       # old-settings (notebook) models
    network = nb.QNetwork(6).to(device)
network.load_state_dict(saved["online"])
network.eval()
print(f"Loaded {POLICY.name}: {config.method}, trained {saved['step']:,} decisions; "
      f"epsilon {config.eval_epsilon}; device {device}")

# ---- identical PongEnv, but ask gym for a visible window instead of rgb_array frames
_real_make = nb.gym.make
def _make_human(*a, **kw):
    if kw.get("render_mode") == "rgb_array":
        kw["render_mode"] = "human"
    return _real_make(*a, **kw)
nb.gym.make = _make_human
if args.no_window:
    nb.gym.make = _real_make
    env = nb.PongEnv(config)
else:
    env = nb.PongEnv(config, video=True)       # video=True -> rgb_array -> swapped to human
    nb.gym.make = _real_make
    assert env.raw.unwrapped.render_mode == "human"

frame_s = 1 / (60 * args.speed) if args.speed > 0 and not args.no_window else 0.0
totals = []
try:
    for index in range(args.start_index, args.start_index + args.games):
        # Same held-out seeds and action RNG as the notebook's evaluate()
        eval_seed = 100_000 + config.seed * 100 + index
        rng = np.random.default_rng(eval_seed + 1_000_000)
        observation, _ = env.reset(seed=eval_seed)
        agent = opponent = 0
        print(f"\n=== Game {index + 1 - args.start_index} (eval seed {eval_seed}) - agent is GREEN (right paddle) ===")
        clock = time.perf_counter()
        terminated = truncated = False
        while not (terminated or truncated):
            action = nb.choose_action(network, observation, config.eval_epsilon, rng,
                                      env.action_count, device)
            observation, reward, terminated, truncated, info = env.step(action)
            if reward > 0:
                agent += 1;    print(f"  Agent scores    -> agent {agent:2d} : {opponent:2d} opponent", flush=True)
            elif reward < 0:
                opponent += 1; print(f"  Opponent scores -> agent {agent:2d} : {opponent:2d} opponent", flush=True)
            # Pace to real Atari speed: this decision covered info["actual_game_frames"] frames
            clock += info["actual_game_frames"] * frame_s
            delay = clock - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            else:
                clock = time.perf_counter()
        end = "time limit" if truncated and not terminated else "game over"
        print(f"FINAL ({end}): agent {agent} : {opponent} opponent  (score {agent - opponent:+d})")
        totals.append(agent - opponent)
except KeyboardInterrupt:
    print("\nStopped by user.")
finally:
    env.close()
if len(totals) > 1:
    print(f"\nMean score over {len(totals)} games: {np.mean(totals):+.2f}")
