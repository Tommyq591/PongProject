"""Scratch-only extensions of the verbatim notebook code (../pong_train_extract.py is NOT edited).

1. FastCheckpointTrainer: same training loop, but checkpoints store the replay buffer as raw
   .npy files (np.save streams the arrays straight to disk; no pickled in-memory copy). The
   rest of the state (weights, optimiser, RNGs, emulator) goes in a small checkpoint.pt.gz.
   Fixes the ~3.9x-replay RAM spike of the notebook's Trainer.save(). Resume-safe:
   run_training() still detects checkpoint.pt.gz and calls load(), and the swap is ordered so
   a crash mid-save leaves a consistent (old or new) checkpoint.
2. DiagConfig.network: "small" (notebook CNN, default) or "nature" (Mnih et al. 2015 / CleanRL
   CNN: 32-64-64 conv, 512 dense). Applies identically to DQN and Double DQN.

Use:  run_training(cfg, folder, trainer_factory=FastCheckpointTrainer)
"""
import gzip, importlib.util, json, os, shutil, sys, time
from dataclasses import asdict, dataclass
from pathlib import Path
HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("pong_train_extract", HERE.parent / "pong_train_extract.py")
nb = sys.modules.get("pong_train_extract")
if nb is None:
    nb = importlib.util.module_from_spec(_spec); sys.modules["pong_train_extract"] = nb; _spec.loader.exec_module(nb)
import numpy as np, torch
from torch import nn

@dataclass
class DiagConfig(nb.Config):
    network: str = "small"
    def validate(self):
        super().validate(); assert self.network in {"small", "nature"}

class NatureQNetwork(nn.Module):
    """Mnih et al. (2015) / CleanRL dqn_atari CNN. Input uint8 4x84x84, scaled /255 like the notebook's."""
    def __init__(self, action_count=6):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4), nn.ReLU(), nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(), nn.Flatten())
        self.values = nn.Sequential(nn.Linear(64 * 7 * 7, 512), nn.ReLU(), nn.Linear(512, action_count))
    def forward(self, images):
        return self.values(self.features(images.float() / 255.0))

NAMES = ("frames", "next_frames", "actions", "rewards", "terminated", "truncated", "episode_steps", "ids")

class FastCheckpointTrainer(nb.Trainer):
    def __init__(self, config, folder):
        super().__init__(config, folder)
        if getattr(config, "network", "small") == "nature":
            torch.manual_seed(config.seed)            # same seeding point as the notebook's network
            self.online = NatureQNetwork(self.env.action_count).to(config.device)
            self.target = __import__("copy").deepcopy(self.online).eval()
            self.target.requires_grad_(False)
            self.optimizer = torch.optim.Adam(self.online.parameters(), lr=config.learning_rate)

    # ---- state without replay arrays (count + RNG of the replay are kept; arrays go to .npy)
    def state_dict(self):
        state = super().state_dict()
        rb = state.pop("replay")
        state["replay_meta"] = {"count": rb["count"], "rng": rb["rng"], "step": self.step}
        return state

    def load_state_dict(self, state):
        meta = state["replay_meta"]
        arrays = self._load_replay_arrays(meta["step"])
        state = dict(state)
        state["replay"] = {"count": meta["count"], "rng": meta["rng"], "arrays": arrays}
        try:
            super().load_state_dict(state)      # copies the memory-mapped arrays into the buffer
        finally:
            state.pop("replay"); del arrays     # release the .npy files (Windows file locks)

    def load(self):
        """Same as the notebook's Trainer.load(), except map_location='cpu'.
        BUG FIX: the notebook maps the whole checkpoint to config.device; on cuda that moves the CPU
        RNG ByteTensor to the GPU and torch.set_rng_state() raises 'RNG state must be a torch.ByteTensor'.
        Model/optimiser tensors still end up on the GPU via load_state_dict()."""
        import csv
        with gzip.open(self.folder / "checkpoint.pt.gz", "rb") as stream:
            state = torch.load(stream, map_location="cpu", weights_only=False)
        self.load_state_dict(state)
        del state
        for name in ("episodes.csv", "updates.csv", "progress.csv", "evaluations.csv"):
            path = self.folder / name
            if path.exists():
                with path.open(newline="", encoding="utf-8") as stream:
                    rows = list(csv.DictReader(stream))
                kept = [r for r in rows if int(r["step"]) <= self.step]
                if len(kept) != len(rows):
                    shutil.copy2(path, path.with_name(path.stem + f"_before_resume_{time.time_ns()}.csv"))
                    path.unlink()
                    for row in kept:
                        nb.append_csv(path, row)
        nb.append_event(self.folder, "checkpoint_resumed", step=self.step)

    def _load_replay_arrays(self, step):
        for d in (self.folder / "replay", self.folder / "replay.old"):
            m = d / "meta.json"
            if m.exists() and json.loads(m.read_text())["step"] == step:
                return {n: np.load(d / f"{n}.npy", mmap_mode="r") for n in NAMES}
        raise FileNotFoundError(f"no replay snapshot matching checkpoint step {step}")

    def save(self):
        started = time.perf_counter()
        f = self.folder
        tmp, cur, old = f / "replay.tmp", f / "replay", f / "replay.old"
        shutil.rmtree(tmp, ignore_errors=True); tmp.mkdir()
        n = len(self.replay)
        for name in NAMES:                             # views, streamed to disk; no copy in RAM
            np.save(tmp / f"{name}.npy", getattr(self.replay, name)[:n])
        (tmp / "meta.json").write_text(json.dumps({"step": self.step, "count": self.replay.count}))
        with gzip.open(f / "checkpoint.tmp.gz", "wb", compresslevel=1) as stream:
            torch.save(self.state_dict(), stream)
        # swap: keep the previous replay as replay.old until the new checkpoint is in place
        shutil.rmtree(old, ignore_errors=True)
        if cur.exists(): os.replace(cur, old)
        os.replace(tmp, cur)
        os.replace(f / "checkpoint.tmp.gz", f / "checkpoint.pt.gz")
        shutil.rmtree(old, ignore_errors=True)
        torch.save({"config": asdict(self.config), "step": self.step,
                    "online": self.online.state_dict()}, f / "policy.pt")
        nb.append_event(f, "checkpoint_saved", step=self.step, seconds=time.perf_counter() - started,
                        format="fast_npy_replay")
