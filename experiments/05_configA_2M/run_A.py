"""Config A (reference-aligned), DQN seed 17, 2M decisions. Re-run the same command to resume.
Writes to output/ (run/ = resumable run folder). --dry-run prints the config and exits.

Verbatim notebook run_training + FastCheckpointTrainer (npy replay checkpoints, cuda load fix).
Checkpoint every 100k; permanent policy snapshots policy_500k/1000k/1500k/2000k.pt.
"""
import json, shutil, sys, time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]   # repo root (all paths are relative to it)
sys.path.insert(0, str(REPO / "common"))
OUT = HERE / "output"; OUT.mkdir(exist_ok=True)   # re-runs write here; recorded results are in recorded/
from pong_train_diag import nb, DiagConfig, FastCheckpointTrainer
import psutil

PLAN = json.loads((REPO / "Assignment2" / "experiment_plan.json").read_text())["shared_config"]
config = replace(DiagConfig(**PLAN),
                 method="dqn", seed=17, device="cuda", total_steps=2_000_000,
                 network="nature", replay_capacity=200_000, target_update_interval=1_000,
                 epsilon_end=0.01, epsilon_decay_steps=100_000, learning_starts=20_000,
                 eval_interval=100_000, eval_episodes=5, checkpoint_interval=100_000)

class SnapshotTrainer(FastCheckpointTrainer):
    def save(self):
        super().save()
        if self.step > 0 and self.step % 500_000 == 0:
            snap = self.folder.parent / f"policy_{self.step // 1000}k.pt"
            if not snap.exists():                       # never overwritten
                shutil.copy2(self.folder / "policy.pt", snap)

if "--dry-run" in sys.argv:
    from dataclasses import asdict
    print(json.dumps(asdict(config), indent=1)); print("output folder:", OUT); sys.exit(0)
OUT.mkdir(parents=True, exist_ok=True)
t0 = time.perf_counter(); start = datetime.now(timezone.utc).isoformat(); status, err = "unknown", None
try:
    result = nb.run_training(config, OUT / "run", trainer_factory=SnapshotTrainer)
    status = result["status"]
except BaseException as e:
    status, err = type(e).__name__, repr(e); raise
finally:
    mi = psutil.Process().memory_info()
    with (OUT / "sessions.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({"start_utc": start, "end_utc": datetime.now(timezone.utc).isoformat(),
                            "status": status, "error": err, "wall_clock_s": time.perf_counter() - t0,
                            "peak_working_set_mib": getattr(mi, "peak_wset", 0) / 2**20}) + "\n")
    print("SESSION END", status, err, flush=True)
