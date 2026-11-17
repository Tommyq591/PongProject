"""Tests for FastCheckpointTrainer.
1. exact recovery (like the notebook's check_full_recovery): save at 160, continue to 240,
   reload from disk, continue to 240 again -> identical pixels, rewards, update count, weights.
   Run for both networks and both methods.
2. RAM: fill a 100k replay buffer (no updates), then save; report peak working set during save
   vs before. Compare with the notebook save's measured 8.25 GiB peak.
"""
import json, sys, tempfile, threading, time
from dataclasses import replace
from pathlib import Path
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]   # repo root (all paths are relative to it)
sys.path.insert(0, str(REPO / "common"))
OUT = HERE / "output"; OUT.mkdir(exist_ok=True)   # re-runs write here; recorded results are in recorded/
from pong_train_diag import nb, DiagConfig, FastCheckpointTrainer
import numpy as np, psutil, torch

PLAN = json.loads((REPO / "Assignment2" / "experiment_plan.json").read_text())["shared_config"]
out = {}

for network in ("small", "nature"):
    for method in ("dqn", "double_dqn"):
        cfg = replace(DiagConfig(**PLAN), network=network, method=method, device="cpu", total_steps=320,
                      replay_capacity=256, learning_starts=64, epsilon_decay_steps=300, torch_threads=2)
        with tempfile.TemporaryDirectory() as d:
            t = FastCheckpointTrainer(cfg, d); t.advance(160); t.save(); t.advance(80)
            want = {k: v.clone() for k, v in t.online.state_dict().items()}
            obs, ret, upd = t.observation.copy(), t.episode_return, t.updates
            rb_ids = t.replay.ids.copy(); t.close()
            r = FastCheckpointTrainer(cfg, d); r.load()
            assert r.step == 160
            r.advance(80)
            ok = (r.updates == upd and r.episode_return == ret and np.array_equal(r.observation, obs)
                  and all(torch.equal(v, want[k]) for k, v in r.online.state_dict().items())
                  and np.array_equal(r.replay.ids, rb_ids))
            r.close()
        out[f"exact_recovery_{network}_{method}"] = ok
        print(f"exact recovery {network:6s} {method:10s}: {'PASS' if ok else 'FAIL'}")

# RAM test with a full 100k buffer (cuda, like the real runs)
proc = psutil.Process(); peak = {"v": 0}; stop = threading.Event()
def sample():
    while not stop.is_set():
        peak["v"] = max(peak["v"], proc.memory_info().rss); time.sleep(0.05)
cfg = replace(DiagConfig(**PLAN), device="cuda" if torch.cuda.is_available() else "cpu",
              replay_capacity=100_000, learning_starts=10**9, total_steps=10**6)
with tempfile.TemporaryDirectory(dir=OUT) as d:
    t = FastCheckpointTrainer(cfg, d)
    t0 = time.perf_counter(); t.advance(100_000); fill_s = time.perf_counter() - t0
    before = proc.memory_info().rss
    threading.Thread(target=sample, daemon=True).start()
    t1 = time.perf_counter(); t.save(); save_s = time.perf_counter() - t1
    stop.set(); time.sleep(0.1)
    size = sum(p.stat().st_size for p in Path(d).rglob("*") if p.is_file())
    t2 = time.perf_counter(); r = FastCheckpointTrainer(cfg, d); r.load(); load_s = time.perf_counter() - t2
    same = all(np.array_equal(getattr(r.replay, n), getattr(t.replay, n)) for n in ("frames", "next_frames", "ids"))
    t.close(); r.close()
out["ram_100k"] = {"fill_s": fill_s, "rss_before_save_gib": before / 2**30, "peak_rss_during_save_gib": peak["v"] / 2**30,
                   "extra_during_save_gib": (peak["v"] - before) / 2**30, "save_s": save_s, "load_s": load_s,
                   "checkpoint_on_disk_gib": size / 2**30, "reloaded_replay_identical": same,
                   "notebook_save_peak_gib_for_comparison": 8.25}
print(json.dumps(out["ram_100k"], indent=1))
(OUT / "test_fast_checkpoint.json").write_text(json.dumps(out, indent=2))
