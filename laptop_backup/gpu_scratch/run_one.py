"""One timed throughput run using the verbatim notebook Trainer.

Only config overrides are applied (device, replay_capacity, total_steps, method, seed);
all other settings come from experiment_plan.json's shared_config (read-only).
"""
import argparse, importlib.util, json, os, subprocess, threading, time
from dataclasses import asdict, replace
from pathlib import Path

HERE = Path(__file__).resolve().parent
p = argparse.ArgumentParser()
p.add_argument("--method", required=True)
p.add_argument("--seed", type=int, default=17)
p.add_argument("--device", required=True)
p.add_argument("--decisions", type=int, required=True)
p.add_argument("--replay", type=int, default=100_000)
p.add_argument("--plan", required=True)
p.add_argument("--out", required=True)
p.add_argument("--tag", default="")
p.add_argument("--threads", type=int, default=None)
args = p.parse_args()

spec = importlib.util.spec_from_file_location("pong_train_extract", HERE / "pong_train_extract.py")
nbcode = importlib.util.module_from_spec(spec)
import sys
sys.modules["pong_train_extract"] = nbcode  # dataclasses needs the module registered
spec.loader.exec_module(nbcode)
import numpy as np, psutil, torch

shared = json.loads(Path(args.plan).read_text(encoding="utf-8"))["shared_config"]
config = replace(nbcode.Config(**shared), method=args.method, seed=args.seed, device=args.device,
                 replay_capacity=args.replay, total_steps=args.decisions)
if args.threads is not None:
    config = replace(config, torch_threads=args.threads)
run_name = f"{args.method}_seed{args.seed}_{args.device}{args.tag}"
folder = HERE / "runs" / run_name          # all Trainer outputs (config.json, csvs) go here

proc = psutil.Process()
stop = threading.Event()
ram_peak = {"rss": 0}
def ram_sampler():
    while not stop.is_set():
        ram_peak["rss"] = max(ram_peak["rss"], proc.memory_info().rss)
        time.sleep(0.25)
gpu_samples = []
smi = {}
def gpu_sampler():
    try:
        sp = subprocess.Popen(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                               "--format=csv,noheader,nounits", "-lms", "250"],
                              stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                              stdin=subprocess.DEVNULL, text=True)
        smi["proc"] = sp
    except Exception as e:
        gpu_samples.append(("error", repr(e))); return
    while not stop.is_set():
        line = sp.stdout.readline()
        if not line: break
        try:
            u, m = (float(x) for x in line.strip().split(","))
            gpu_samples.append((time.perf_counter(), u, m))
        except ValueError:
            pass
    sp.terminate()
threading.Thread(target=ram_sampler, daemon=True).start()
threading.Thread(target=gpu_sampler, daemon=True).start()

cuda = args.device == "cuda"
if cuda:
    torch.cuda.reset_peak_memory_stats()
sync = torch.cuda.synchronize if cuda else (lambda: None)

trainer = nbcode.Trainer(config, folder)
# Phase 1: before learning starts (env stepping + replay writes, no updates)
t0 = time.perf_counter(); trainer.advance(config.learning_starts); sync()
pre_s = time.perf_counter() - t0
# Phase 2: measured, learning active. Chunks of 1000 so we can see stability.
chunks, measured = [], args.decisions - config.learning_starts
u0, m_start = trainer.updates, time.perf_counter()
for _ in range(measured // 1000):
    t = time.perf_counter(); trainer.advance(1000); sync()
    chunks.append(1000 / (time.perf_counter() - t))
m_end = time.perf_counter()
learn_s = m_end - m_start
updates = trainer.updates - u0

# Supplementary: greedy action-selection latency (epsilon=0), to adjust for late training
# when most actions use the network. Uses the notebook's choose_action unchanged.
rng = np.random.default_rng(0)
obs = trainer.observation
for _ in range(50):
    nbcode.choose_action(trainer.online, obs, 0.0, rng, 6, config.device)
t = time.perf_counter()
for _ in range(500):
    nbcode.choose_action(trainer.online, obs, 0.0, rng, 6, config.device)
greedy_ms = (time.perf_counter() - t) / 500 * 1000
checkpoint = None
if args.decisions >= 100_000:  # replay is full by now: time the notebook's own save()
    t = time.perf_counter(); trainer.save()
    checkpoint = {"save_s": time.perf_counter() - t, "replay_len": len(trainer.replay),
                  "checkpoint_gz_mib": (folder / "checkpoint.pt.gz").stat().st_size / 2**20,
                  "policy_pt_mib": (folder / "policy.pt").stat().st_size / 2**20}
trainer.close()
time.sleep(0.6)
stop.set()
if "proc" in smi:  # kill the sampler from the main thread so it can't outlive us
    smi["proc"].kill(); smi["proc"].wait(timeout=10)
ram_peak["rss"] = max(ram_peak["rss"], proc.memory_info().rss)

window = [s for s in gpu_samples if s[0] != "error" and m_start <= s[0] <= m_end]
result = {
    "run": run_name, "method": args.method, "seed": args.seed, "device": args.device,
    "total_decisions": trainer.step, "learning_starts": config.learning_starts,
    "replay_capacity": config.replay_capacity, "torch_threads": config.torch_threads,
    "pre_learning_s": pre_s, "pre_learning_decisions_per_s": config.learning_starts / pre_s,
    "measured_decisions_after_learning_starts": measured, "measured_updates": updates,
    "measured_s": learn_s, "decisions_per_s_after_learning_starts": measured / learn_s,
    "per_1000_chunk_decisions_per_s": [round(c, 1) for c in chunks],
    "epsilon_at_end": nbcode.epsilon_at(trainer.step, config),
    "greedy_action_latency_ms": greedy_ms,
    "peak_gpu_memory_allocated_mib": torch.cuda.max_memory_allocated() / 2**20 if cuda else None,
    "peak_gpu_memory_reserved_mib": torch.cuda.max_memory_reserved() / 2**20 if cuda else None,
    "peak_process_rss_mib_sampled": ram_peak["rss"] / 2**20,
    "peak_process_working_set_mib_windows": getattr(proc.memory_info(), "peak_wset", 0) / 2**20,
    "gpu_util_samples_in_window": len(window),
    "gpu_util_mean_pct": float(np.mean([s[1] for s in window])) if window else None,
    "gpu_util_max_pct": float(np.max([s[1] for s in window])) if window else None,
    "gpu_mem_used_total_mib_max": float(np.max([s[2] for s in window])) if window else None,
    "gpu_sampler_error": next((s[1] for s in gpu_samples if s[0] == "error"), None),
    "checkpoint_save": checkpoint,
    "env": {"CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG")},
    "config": asdict(config),
}
Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps({k: v for k, v in result.items() if k != "config"}, indent=1))
