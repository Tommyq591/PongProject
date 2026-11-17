"""1M-decision DQN pilot using the notebook's own run_training (verbatim, cell 29).

Config overrides only. Re-running this script resumes from output/run/checkpoint.pt.gz:
run_training checks config.json matches, loads the checkpoint and continues.
Each session's wall-clock and peak RAM are appended to output/sessions.jsonl.
NOTE: uses the notebook's original Trainer, so it cannot resume on cuda (see common/pong_train_diag.py)
and its checkpoint save needs ~3.9x the replay size in RAM.
  --dry-run   print the configuration and output folder, then exit (no training)
"""
import importlib.util, json, sys, threading, time
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]   # repo root (all paths are relative to it)
sys.path.insert(0, str(REPO / "common"))
OUT = HERE / "output"; OUT.mkdir(exist_ok=True)   # re-runs write here; recorded results are in recorded/
PLAN = REPO / "Assignment2" / "experiment_plan.json"   # read-only

spec = importlib.util.spec_from_file_location("pong_train_extract", REPO / "common" / "pong_train_extract.py")
nb = importlib.util.module_from_spec(spec)
sys.modules["pong_train_extract"] = nb
spec.loader.exec_module(nb)
import psutil, torch

shared = json.loads(PLAN.read_text(encoding="utf-8"))["shared_config"]
config = replace(nb.Config(**shared),
                 method="dqn", seed=17, device="cuda",
                 replay_capacity=100_000,
                 total_steps=1_000_000,
                 eval_interval=50_000,          # plan: 20,000
                 eval_episodes=5,               # plan: 5 (unchanged)
                 checkpoint_interval=100_000)   # plan: 10,000
# Unchanged from the plan: epsilon_decay_steps=80,000, final_eval_episodes=10, everything else.
assert config.epsilon_decay_steps == shared["epsilon_decay_steps"]
if "--dry-run" in sys.argv:
    print(json.dumps(asdict(config), indent=1)); print("output folder:", OUT); sys.exit(0)

proc = psutil.Process()
peak = {"rss": 0}
stop = threading.Event()
def sampler():
    while not stop.is_set():
        try: peak["rss"] = max(peak["rss"], proc.memory_info().rss)
        except Exception: pass
        time.sleep(0.5)
threading.Thread(target=sampler, daemon=True).start()

OUT.mkdir(parents=True, exist_ok=True)
session = {"start_utc": datetime.now(timezone.utc).isoformat(), "config": asdict(config)}
t0 = time.perf_counter()
status, error = "unknown", None
try:
    result = nb.run_training(config, OUT / "run")
    status = result["status"]; session["result"] = result
except BaseException as e:   # run_training already emergency-saved; record and re-raise
    status, error = ("interrupted" if isinstance(e, KeyboardInterrupt) else "failed"), repr(e)
    raise
finally:
    stop.set()
    mi = proc.memory_info()
    session.update({"end_utc": datetime.now(timezone.utc).isoformat(), "status": status, "error": error,
                    "wall_clock_s": time.perf_counter() - t0,
                    "peak_rss_mib_sampled": max(peak["rss"], mi.rss) / 2**20,
                    "peak_working_set_mib": getattr(mi, "peak_wset", 0) / 2**20,
                    "peak_gpu_mem_allocated_mib": torch.cuda.max_memory_allocated() / 2**20})
    session.pop("config")
    with (OUT / "sessions.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(session) + "\n")
    print("SESSION", json.dumps(session), flush=True)
