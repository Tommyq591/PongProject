"""Verify Pong RAM addresses against positions measured from screen pixels.

Plays ~4000 raw frames (frameskip 1) with random held actions, detects our green paddle,
the orange opponent paddle and the white ball in each RGB frame, and fits
pixel = a * RAM[address] + b for each candidate address. Also checks which joystick
action moves our paddle up. Writes ram_verification.json next to this file.
"""
import json, sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")   # constant RAM bytes make polyfit warn during the address scan
import numpy as np
import ale_py, gymnasium as gym

sys.dont_write_bytecode = True
gym.register_envs(ale_py)
HERE = Path(__file__).resolve().parent

GREEN, ORANGE, WHITE = (92, 186, 92), (213, 130, 74), (236, 236, 236)
TOP, BOTTOM = 34, 194          # playfield rows (walls are white above/below)

def find(frame, colour, cols=None):
    mask = np.all(frame[TOP:BOTTOM] == colour, axis=2)
    if cols is not None:
        sub = np.zeros_like(mask); sub[:, cols] = mask[:, cols]; mask = sub
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return None
    return float(ys.mean() + TOP), float(xs.mean())

env = gym.make("ALE/Pong-v5", frameskip=1, repeat_action_probability=0.0, full_action_space=False,
               render_mode="rgb_array")
rng = np.random.default_rng(0)
env.reset(seed=0)
records, action, hold = [], 0, 0
moves = {2: [], 3: []}
prev_ram = None
for t in range(4000):
    if hold == 0:
        action, hold = int(rng.integers(6)), int(rng.integers(5, 40))
    hold -= 1
    _, _, term, trunc, _ = env.step(action)
    frame = env.render()
    ram = env.unwrapped.ale.getRAM().astype(int)
    if prev_ram is not None and action in (2, 3):
        moves[action].append(ram[51] - prev_ram[51])
    prev_ram = ram
    records.append({"ram": ram, "me": find(frame, GREEN, slice(130, 160)),
                    "opp": find(frame, ORANGE, slice(0, 30)), "ball": find(frame, WHITE)})
    if term or trunc:
        env.reset(); prev_ram = None
env.close()

def fit(address, getter, label):
    pairs = [(r["ram"][address], getter(r)) for r in records if getter(r) is not None]
    x, y = np.array(pairs, dtype=float).T
    a, b = np.polyfit(x, y, 1)
    resid = y - (a * x + b)
    r2 = 1 - resid.var() / y.var()
    return {"label": label, "address": address, "n": len(x), "a": a, "b": b, "r2": r2,
            "median_abs_err_px": float(np.median(np.abs(resid))),
            "p95_abs_err_px": float(np.percentile(np.abs(resid), 95))}

checks = [
    fit(51, lambda r: r["me"] and r["me"][0], "our paddle y (centre)"),
    fit(50, lambda r: r["opp"] and r["opp"][0], "opponent paddle y (centre)"),
    fit(54, lambda r: r["ball"] and r["ball"][0], "ball y"),
    fit(49, lambda r: r["ball"] and r["ball"][1], "ball x"),
]
# Alternatives, to show the chosen addresses are the right ones
alts = {}
for label, getter in (("ball y", lambda r: r["ball"] and r["ball"][0]),
                      ("our paddle y", lambda r: r["me"] and r["me"][0])):
    best = []
    for addr in range(128):
        try:
            f = fit(addr, getter, label)
            if np.isfinite(f["r2"]):
                best.append((f["r2"], addr))
        except Exception:
            pass
    alts[label] = sorted(best, reverse=True)[:3]
ball_visible = sum(r["ball"] is not None for r in records)
out = {"frames": len(records), "ball_visible_frames": ball_visible, "fits": checks,
       "best_addresses_by_r2": alts,
       "paddle_dy_per_frame": {"RIGHT(2)": float(np.mean(moves[2])), "LEFT(3)": float(np.mean(moves[3]))}}
(HERE / "ram_verification.json").write_text(json.dumps(out, indent=2, default=float))
for c in checks:
    print(f"{c['label']:28s} RAM[{c['address']}]  pixel = {c['a']:.3f}*ram + {c['b']:7.2f}   "
          f"R2={c['r2']:.4f}  median err {c['median_abs_err_px']:.2f}px  p95 {c['p95_abs_err_px']:.2f}px  (n={c['n']})")
print("best addresses by R2:", alts)
print("mean RAM[51] change per frame: RIGHT", round(np.mean(moves[2]), 2), " LEFT", round(np.mean(moves[3]), 2))
