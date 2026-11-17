"""Short GPU throughput benchmark of candidate configs (no checkpoints, outputs in a temp dir).
For each config: 2,000 warm-up decisions (learning starts), then 3,000 decisions at epsilon~1
(exploring) and 3,000 at epsilon 0.01 (greedy) -> decisions/s in both phases.
"""
import json, sys, tempfile, time
from dataclasses import replace
from pathlib import Path
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]   # repo root (all paths are relative to it)
sys.path.insert(0, str(REPO / "common"))
OUT = HERE / "output"; OUT.mkdir(exist_ok=True)   # re-runs write here; recorded results are in recorded/
from pong_train_diag import nb, DiagConfig, FastCheckpointTrainer
import torch

PLAN = json.loads((REPO / "Assignment2" / "experiment_plan.json").read_text())["shared_config"]
BASE = replace(DiagConfig(**PLAN), device="cuda", replay_capacity=100_000, learning_starts=2000)
CONFIGS = {
    "baseline_small_tf4": {},
    "A_nature_tf4": {"network": "nature"},
    "B_nature_tf1": {"network": "nature", "train_frequency": 1},
}
out = {}
for name, over in CONFIGS.items():
    res = {}
    for phase, decay in (("exploring", 10**9), ("greedy", 1)):   # decay 1 -> epsilon_end immediately
        cfg = replace(BASE, **over, epsilon_decay_steps=decay, epsilon_end=0.01 if phase == "greedy" else BASE.epsilon_end)
        with tempfile.TemporaryDirectory() as d:
            t = FastCheckpointTrainer(cfg, d)
            t.advance(2000); torch.cuda.synchronize()
            s = time.perf_counter(); u = t.updates
            t.advance(3000); torch.cuda.synchronize()
            res[phase] = {"dps": 3000 / (time.perf_counter() - s), "updates": t.updates - u}
            t.close()
    out[name] = {"overrides": over, **res}
    print(name, {p: round(v["dps"], 1) for p, v in res.items()}, flush=True)
(OUT / "bench_configs.json").write_text(json.dumps(out, indent=2))
