"""Play Pong yourself with the keyboard against the game's BUILT-IN computer opponent.

NOTE: the opponent (left, orange) is the Atari game's own bot - NOT our AI. You are GREEN (right).
To watch OUR trained AI play that same bot, use watch_ai.py instead.

Based on Tommy's demos/Pong_Practice.ipynb (same environment and keys), but draws the game in a
plain tkinter window, so pygame is not needed:
    W / Up arrow    move up          S / Down arrow   move down
    Space           serve (FIRE)     R  new game      Esc  quit
Game: ALE/Pong-v5, every frame shown (frameskip 1), no sticky actions, 60 frames/s. First to 21.

  python play_pong.py              # play
  python play_pong.py --zoom 4     # bigger window
  python play_pong.py --selftest   # no window: 600 frames with random keys (checks the setup)
"""
import argparse, sys, time
import numpy as np
import gymnasium as gym
import ale_py

gym.register_envs(ale_py)
ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--zoom", type=int, default=3)
ap.add_argument("--fps", type=float, default=60.0)
ap.add_argument("--selftest", action="store_true")
args = ap.parse_args()

# Pong's minimal action set: 0 NOOP, 1 FIRE, 2 RIGHT (= up), 3 LEFT (= down), 4 RIGHTFIRE, 5 LEFTFIRE
NOOP, FIRE, UP, DOWN, UPFIRE, DOWNFIRE = range(6)
def action_for(keys):
    up = bool(keys & {"w", "up"}); down = bool(keys & {"s", "down"}); fire = "space" in keys
    if up and not down:
        return UPFIRE if fire else UP
    if down and not up:
        return DOWNFIRE if fire else DOWN
    return FIRE if fire else NOOP

env = gym.make("ALE/Pong-v5", render_mode="rgb_array", frameskip=1, repeat_action_probability=0.0)

if args.selftest:
    rng = np.random.default_rng(0)
    env.reset(seed=0)
    you = bot = 0
    for _ in range(600):
        keys = [{"w"}, {"s"}, {"space"}, set()][rng.choice(4, p=[.3, .3, .1, .3])]
        _, r, term, trunc, _ = env.step(action_for(keys))
        you += r > 0; bot += r < 0
        if term or trunc:
            env.reset()
    frame = env.render()
    env.close()
    print(f"selftest OK: 600 frames, frame {frame.shape}, score you {you} : {bot} built-in bot")
    sys.exit(0)

import tkinter as tk
from PIL import Image, ImageTk

root = tk.Tk()
root.title("Pong - you (green, right) vs the game's built-in bot (orange, left)")
label = tk.Label(root, bg="black"); label.pack()
status = tk.Label(root, font=("Consolas", 14)); status.pack(fill="x")
tk.Label(root, text="W/Up = up   S/Down = down   Space = serve   R = new game   Esc = quit",
         font=("Consolas", 10)).pack(fill="x")

held = set()
state = {"you": 0, "bot": 0, "over": False, "next": time.perf_counter()}
def key(event, down):
    k = event.keysym.lower()
    if down and k == "escape":
        root.destroy(); return
    if down and k == "r":
        new_game(); return
    (held.add if down else held.discard)(k)
root.bind("<KeyPress>", lambda e: key(e, True))
root.bind("<KeyRelease>", lambda e: key(e, False))

def new_game():
    env.reset()
    state.update(you=0, bot=0, over=False, next=time.perf_counter())

def show():
    frame = env.render()
    img = Image.fromarray(frame).resize((frame.shape[1] * args.zoom, frame.shape[0] * args.zoom), Image.NEAREST)
    label.photo = ImageTk.PhotoImage(img); label.configure(image=label.photo)
    msg = f"You {state['you']:2d} : {state['bot']:2d} built-in bot"
    if state["over"]:
        msg += "   -  " + ("YOU WIN!" if state["you"] > state["bot"] else "bot wins") + "  (R = play again)"
    status.configure(text=msg)

def tick():
    if not state["over"]:
        _, reward, terminated, truncated, _ = env.step(action_for(held))
        state["you"] += reward > 0; state["bot"] += reward < 0
        state["over"] = terminated or truncated
        show()
    state["next"] += 1 / args.fps
    delay = state["next"] - time.perf_counter()
    if delay < -0.25:                       # fell far behind (window dragged etc.): resync
        state["next"] = time.perf_counter(); delay = 0
    root.after(max(1, int(delay * 1000)), tick)

new_game(); show(); root.after(10, tick)
try:
    root.mainloop()
finally:
    env.close()
