"""Record both checkpoint policies; metadata always states the trained step."""
from pathlib import Path
from pong_dqn import record_video

ROOT = Path(__file__).resolve().parent

if __name__ == "__main__":
    for method in ("dqn", "double_dqn"):
        policy = ROOT / f"runs/main/{method}_seed_17/policy.pt"
        if not policy.exists():
            raise FileNotFoundError(f"A real checkpoint is required: {policy}")
        print(record_video(policy, ROOT / f"videos/{method}_seed_17.mp4"), flush=True)
