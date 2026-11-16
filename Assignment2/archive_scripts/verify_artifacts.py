"""Verify saved policies, full checkpoints, measurements and practice preservation."""
import gc
import gzip
import hashlib
import json
from pathlib import Path

import nbformat
import pandas as pd
import torch

from pong_dqn import utc_now, write_json

ROOT = Path(__file__).resolve().parent


def verify():
    plan = json.loads((ROOT / "experiment_plan.json").read_text())
    notebook = nbformat.read(ROOT / "CS4287-Assign2_Atari_ID1_ID2_ID3.ipynb", as_version=4)
    assert all(c.execution_count is not None for c in notebook.cells if c.cell_type == "code")
    assert not any(o.output_type == "error" for c in notebook.cells if c.cell_type == "code" for o in c.outputs)
    for cell in notebook.cells:
        if cell.cell_type == "markdown":
            assert not any(ord(x) < 32 and x not in "\n\r" for x in cell.source)
    common_settings = None
    records = []
    truncations = []
    for run in plan["runs"]:
        folder = ROOT / run["folder"]
        status = json.loads((folder / "status.json").read_text())
        config = json.loads((folder / "config.json").read_text())
        shared = {k: v for k, v in config.items() if k not in ("method", "seed")}
        if common_settings is None:
            common_settings = shared
        assert shared == common_settings
        assert config["method"] == run["method"] and config["seed"] == run["seed"]
        assert status["status"] in {"paused", "complete"}
        with gzip.open(folder / "checkpoint.pt.gz", "rb") as stream:
            checkpoint = torch.load(stream, map_location="cpu", weights_only=False)
        assert checkpoint["step"] == status["step"]
        assert checkpoint["replay"]["count"] == checkpoint["step"]
        required = {"online", "target", "optimizer", "replay", "env", "observation",
                    "action_rng", "python_rng", "numpy_rng", "torch_rng"}
        assert required.issubset(checkpoint)
        assert {"sticky_rng", "previous_action"}.issubset(checkpoint["env"])
        updates = checkpoint["updates"]
        frames = checkpoint["game_frames"]
        train_seconds = checkpoint["elapsed_s"]
        evaluation_seconds = checkpoint["eval_elapsed_s"]
        del checkpoint
        gc.collect()
        policy = torch.load(folder / "policy.pt", map_location="cpu", weights_only=False)
        assert policy["step"] == status["step"]
        evaluations = pd.read_csv(folder / "evaluations.csv")
        episodes = pd.read_csv(folder / "episodes.csv")
        progress = pd.read_csv(folder / "progress.csv")
        assert not evaluations.duplicated(["step", "eval_index"]).any()
        assert int(evaluations.step.max()) == int(status["step"])
        assert int(progress.step.iloc[-1]) == int(status["step"])
        assert evaluations.score.between(-21, 21).all()
        assert evaluations.length.gt(0).all()
        records.append({"method": run["method"], "seed": run["seed"],
                        "trained_decisions": status["step"], "planned_decisions": config["total_steps"],
                        "status": status["status"], "learning_updates": updates,
                        "actual_interaction_game_frames": frames, "training_seconds": train_seconds,
                        "evaluation_seconds": evaluation_seconds,
                        "evaluation_games": len(evaluations), "training_episodes": len(episodes),
                        "latest_mean_score": float(evaluations[evaluations.step == status["step"]].score.mean()),
                        "checkpoint_bytes": (folder / "checkpoint.pt.gz").stat().st_size,
                        "checkpoint_sha256": hashlib.sha256((folder / "checkpoint.pt.gz").read_bytes()).hexdigest()})
        truncations.append({"method": run["method"], "seed": run["seed"],
                            "step": status["step"], "status": status["status"],
                            "training_truncations": int(episodes.truncated.sum()),
                            "evaluation_truncations": int(evaluations.truncated.sum())})
    practice = json.loads((ROOT / "results/practice_original_sha256.json").read_text())
    unchanged = hashlib.sha256(Path(practice["path"]).read_bytes()).hexdigest() == practice["sha256"]
    assert unchanged, "Practice notebook changed unexpectedly."
    report = {"verified_utc": utc_now(), "practice_notebook_unchanged": unchanged,
              "total_main_training_decisions": sum(r["trained_decisions"] for r in records),
              "total_main_learning_updates": sum(r["learning_updates"] for r in records),
              "total_actual_evaluation_games": sum(r["evaluation_games"] for r in records),
              "shared_actual_decisions_per_run": min(r["trained_decisions"] for r in records),
              "all_checkpoint_payloads_read_successfully": True, "runs": records,
              "remaining_team_details": True, "earlier_AI_assistance_pending": True}
    write_json(ROOT / "results/artifact_verification.json", report)
    write_json(ROOT / "results/truncation_inventory.json", truncations)
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    verify()
