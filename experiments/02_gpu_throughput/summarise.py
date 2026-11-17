"""Combine output/results/*.json into output/gpu_throughput.json (reads recorded/ if there is no output/ yet)."""
import csv, json, statistics as st
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]   # repo root (all paths are relative to it)
sys.path.insert(0, str(REPO / "common"))
OUT = HERE / "output"; OUT.mkdir(exist_ok=True)   # re-runs write here; recorded results are in recorded/
SRC = OUT if (OUT / "results").is_dir() else HERE / "recorded"
OUT.mkdir(exist_ok=True)
REPO = REPO / "Assignment2"
R = {p.stem: json.loads(p.read_text()) for p in sorted((SRC / "results").glob("*.json"))}
cpu_ref = json.loads((REPO / "results" / "throughput.json").read_text())   # read-only
plan = json.loads((REPO / "experiment_plan.json").read_text())             # read-only
SLOT_BYTES = 2 * 84 * 84 + 8 + 4 + 1 + 1 + 8 + 8   # frames+next_frames uint8, actions i64, rewards f32, 2 bools, episode_steps i64, ids i64
LS, DECAY0 = 2000, 80000

def dps(name): return R[name]["decisions_per_s_after_learning_starts"] if name in R else None
def mean(xs): xs = [x for x in xs if x is not None]; return st.mean(xs) if xs else None
def sd(xs): xs = [x for x in xs if x is not None]; return st.stdev(xs) if len(xs) > 1 else None
def g_avg(a, b, decay):   # mean greedy fraction (1 - epsilon) over decisions a..b
    s = np.arange(a, b); eps = 1.0 + np.minimum(s / decay, 1.0) * (0.05 - 1.0); return float(np.mean(1 - eps))

# ---- GPU rates per method across seeds / repeat
gpu = {m: {k: dps(k) for k in R if k.startswith(m + "_cuda") and "replay" not in k and "long" not in k}
       for m in ("dqn", "double_dqn")}
gpu = {m: {k: v for k, v in d.items() if not (m == "dqn" and k.startswith("double"))} for m, d in gpu.items()}
gpu["dqn"] = {k: v for k, v in gpu["dqn"].items() if not k.startswith("double")}

# ---- Model seconds/decision = t0 + g * delta, fitted on the long run's 1000-decision chunks
model = None
long = R.get("double_dqn_cuda_s17_long")
if long:
    chunks = long["per_1000_chunk_decisions_per_s"]
    g = [g_avg(LS + 1000 * i, LS + 1000 * (i + 1), DECAY0) for i in range(len(chunks))]
    spd = [1 / c for c in chunks]
    delta, t0_dd = np.polyfit(g, spd, 1)
    model = {"fit_on": "double_dqn_cuda_s17_long chunks", "delta_s_per_greedy_decision": float(delta),
             "t0_double_dqn_s": float(t0_dd),
             "first_10k_dps": float(np.mean(chunks[:10])), "last_10k_dps_eps_0.05": float(np.mean(chunks[-10:]))}
g12 = g_avg(LS, 12000, DECAY0)
t0 = {}
for m in ("dqn", "double_dqn"):
    r = mean(gpu[m].values())
    if r and model:
        t0[m] = 1 / r - g12 * model["delta_s_per_greedy_decision"]

pre_rate = mean([R[k]["pre_learning_decisions_per_s"] for k in R if "cuda" in k])
greedy_ms = mean([R[k]["greedy_action_latency_ms"] for k in R if "cuda" in k])
lengths = []
for f in (SRC / "runs").glob("*/episodes.csv"):
    with f.open() as s: lengths += [int(r["length"]) for r in csv.DictReader(s)]
ep_len = mean(lengths) or 800
save_s = long["checkpoint_save"]["save_s"] if long and long.get("checkpoint_save") else None

# CPU model (approximate: no long CPU run). Greedy cost on CPU = measured CPU greedy latency.
cpu_greedy_ms = mean([R[k]["greedy_action_latency_ms"] for k in R if k in ("dqn_cpu_s17", "double_dqn_cpu_s17")])
cpu_pre = mean([R[k]["pre_learning_decisions_per_s"] for k in R if k in ("dqn_cpu_s17", "double_dqn_cpu_s17")])
g3 = g_avg(LS, 3000, DECAY0)
t0_cpu = {m: 1 / dps(f"{m}_cpu_s17") - g3 * cpu_greedy_ms / 1000 for m in ("dqn", "double_dqn") if dps(f"{m}_cpu_s17")}

def run_time(m, N, device="cuda"):
    g = (0.8 * g_avg(LS, int(0.8 * N), int(0.8 * N)) + 0.2 * 0.95) if N > LS else 0   # plan uses decay = 0.8*N
    if device == "cuda":
        pr, gm, base_t, d = pre_rate, greedy_ms, t0[m], model["delta_s_per_greedy_decision"]
    else:
        pr, gm, base_t, d = cpu_pre, cpu_greedy_ms, t0_cpu[m], cpu_greedy_ms / 1000
    train = LS / pr + (N - LS) * (base_t + g * d)
    games = 5 * (N // 20000) + 5 + 5
    eval_rate = 1 / (1 / pr + gm / 1000)
    ev_lo = games * ep_len / eval_rate
    ev_hi = games * 3000 / eval_rate
    ck = (N // 10000) * save_s if save_s else 0
    return {"train_h": train / 3600, "mean_greedy_fraction": g, "eval_games": games,
            "eval_h_short_games": ev_lo / 3600, "eval_h_long_games_3000": ev_hi / 3600,
            "checkpoint_h_at_100k_replay": ck / 3600}

est = {}
if model and t0:
    for N in (500_000, 1_000_000, 2_000_000):
        a, b = run_time("dqn", N), run_time("double_dqn", N)
        tot_train = 3 * (a["train_h"] + b["train_h"])
        tot_lo = tot_train + 3 * (a["eval_h_short_games"] + b["eval_h_short_games"] + a["checkpoint_h_at_100k_replay"] + b["checkpoint_h_at_100k_replay"])
        tot_hi = tot_train + 3 * (a["eval_h_long_games_3000"] + b["eval_h_long_games_3000"] + a["checkpoint_h_at_100k_replay"] + b["checkpoint_h_at_100k_replay"])
        naive = 6 * N / mean([mean(gpu["dqn"].values()), mean(gpu["double_dqn"].values())]) / 3600
        ca, cb = run_time("dqn", N, "cpu"), run_time("double_dqn", N, "cpu")
        cpu_train = 3 * (ca["train_h"] + cb["train_h"])
        cpu_lo = cpu_train + 3 * (ca["eval_h_short_games"] + cb["eval_h_short_games"] + ca["checkpoint_h_at_100k_replay"] + cb["checkpoint_h_at_100k_replay"])
        teammate_cpu_h = 6 * N / cpu_ref[-1]["decisions_per_s"] / 3600
        est[f"{N:,}"] = {"dqn_per_run": a, "double_dqn_per_run": b,
                         "six_runs_training_only_h": tot_train,
                         "six_runs_incl_eval_ckpt_h_low": tot_lo, "six_runs_incl_eval_ckpt_h_high": tot_hi,
                         "six_runs_naive_from_12k_rate_h": naive,
                         "this_laptop_cpu_six_runs_training_only_h": cpu_train,
                         "this_laptop_cpu_six_runs_incl_eval_ckpt_h_low": cpu_lo,
                         "teammate_cpu_six_runs_naive_h_at_97dps": teammate_cpu_h}

# ---- RAM
ram_runs = {k: {"replay": R[k]["replay_capacity"], "device": R[k]["device"],
                "peak_rss_mib": R[k]["peak_process_rss_mib_sampled"],
                "peak_working_set_mib": R[k]["peak_process_working_set_mib_windows"],
                "slots_filled": min(R[k]["total_decisions"], R[k]["replay_capacity"])} for k in R}
# Observed: np.zeros(frames) is lazily committed (grows as slots fill) but np.zeros_like(next_frames)
# is touched up front, so a short run holds capacity*7056 + filled*7056 bytes of image data.
IMG = 84 * 84
def baseline(device):
    rows = [v for k, v in ram_runs.items() if v["device"] == device and "long" not in k]
    return mean([v["peak_working_set_mib"] - (v["replay"] * IMG + v["slots_filled"] * IMG) / 2**20 for v in rows])
base, base_cpu = baseline("cuda"), baseline("cpu")
long_ws = long["peak_process_working_set_mib_windows"] if long else None
steady_100k = base + 100000 * SLOT_BYTES / 2**20
save_extra = (long_ws - steady_100k) if long else None
save_factor = save_extra / (100000 * SLOT_BYTES / 2**20) if long else None
ram = {}
for cap in (100_000, 200_000, 500_000):
    buf = cap * SLOT_BYTES / 2**30
    ram[f"{cap:,}"] = {"replay_buffer_gib": buf,
                       "cuda_process_steady_gib": buf + base / 1024,
                       "cuda_process_peak_during_checkpoint_gib": buf * (1 + save_factor) + base / 1024,
                       "cpu_process_steady_gib": buf + base_cpu / 1024,
                       "cpu_process_peak_during_checkpoint_gib": buf * (1 + save_factor) + base_cpu / 1024}

summary = {
    "machine": "RTX 5050 Laptop GPU (8 GB), Intel Core 5 210H, 15.6 GB RAM; torch 2.15.0+cu132",
    "runs": {k: {x: y for x, y in v.items() if x != "config"} for k, v in R.items()},
    "gpu_decisions_per_s_by_method": {m: {"runs": d, "mean": mean(d.values()), "sd": sd(d.values())} for m, d in gpu.items()},
    "cpu_reference_teammate_i7_1360P": cpu_ref,
    "rate_model": model, "t0_s_per_decision": t0, "pre_learning_rate": pre_rate,
    "greedy_action_latency_ms_cuda": greedy_ms, "mean_training_episode_length": ep_len,
    "checkpoint_save_s_at_100k_replay": save_s,
    "estimates_six_runs": est,
    "cpu_model": {"t0_s_per_decision": t0_cpu, "greedy_ms": cpu_greedy_ms, "pre_learning_rate": cpu_pre},
    "ram": {"replay_bytes_per_slot": SLOT_BYTES, "baseline_process_mib_cuda_excl_replay": base,
            "baseline_process_mib_cpu_excl_replay": base_cpu, "steady_mib_full_100k_cuda": steady_100k,
            "checkpoint_extra_mib_at_100k": save_extra, "checkpoint_extra_as_multiple_of_replay": save_factor,
            "by_replay_capacity": ram, "measured": ram_runs},
}
(OUT / "gpu_throughput.json").write_text(json.dumps(summary, indent=2))
print(json.dumps({k: summary[k] for k in ("gpu_decisions_per_s_by_method", "rate_model", "t0_s_per_decision",
      "pre_learning_rate", "greedy_action_latency_ms_cuda", "mean_training_episode_length",
      "checkpoint_save_s_at_100k_replay", "estimates_six_runs", "ram")}, indent=1, default=str))
