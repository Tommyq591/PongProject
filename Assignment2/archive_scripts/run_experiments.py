"""Run/resume the predeclared six-run comparison, saving every planned run.

Examples (from this directory, using PongProject's .venv Python):
    python run_experiments.py --until 20000
    python run_experiments.py
The first command pauses at a common 20,000-decision milestone.
The second resumes every run to the full budget saved in experiment_plan.json.
"""
import argparse
import json
import time
import traceback
from dataclasses import asdict, replace
from pathlib import Path

from pong_dqn import Config, append_event, run_training, utc_now, write_json

ROOT = Path(__file__).resolve().parent


def create_plan(total_steps, threads, measured_dps):
    """Call once AFTER the pilot and BEFORE any final training scores are seen."""
    plan_path = ROOT / "experiment_plan.json"
    if plan_path.exists():
        raise FileExistsError("A declared experiment already exists; do not overwrite it.")
    shared = Config(total_steps=total_steps, torch_threads=threads,
                    epsilon_decay_steps=int(total_steps * .8))
    seeds = [17, 29, 43]
    plan = {
        "created_utc": utc_now(), "question": "DQN versus Double DQN on Pong under equal budgets",
        "status": "planned", "methods": ["dqn", "double_dqn"], "seeds": seeds,
        "steps_per_run": total_steps, "agent_step_definition": "one decision; at most four emulator frames",
        "pilot_measured_dps": measured_dps,
        "estimated_training_hours_all_runs": total_steps * 6 / measured_dps / 3600,
        "estimate_excludes": "evaluation, checkpoint I/O, startup and thermal slowdown",
        "selection_reason": "CPU/RAM feasibility measured in a separate pilot; no main-run score used for selection.",
        "primary_metric": "mean raw score over 10 held-out final games, then average across the three training seeds",
        "learning_speed": "area under periodic mean-score curve divided by decisions; first checkpoint with mean score >= 0",
        "consistency": "sample standard deviation of per-training-seed final mean scores; also show every seed",
        "threshold": 0.0, "threshold_reason": "zero is an interpretable non-losing average score, declared before main runs",
        "evaluation_rule": "epsilon=0.01 using online network, 5 periodic games and 10 final games; separate environment, no learning",
        "statistical_limit": "three independent training seeds are descriptive evidence, not a well-powered significance test",
        "run_order": "round-robin at evaluation milestones; method order alternates by seed and milestone",
        "shared_config": asdict(shared), "runs": []}
    for seed in seeds:
        for method in plan["methods"]:
            config = replace(shared, method=method, seed=seed)
            relative = f"runs/main/{method}_seed_{seed}"
            folder = ROOT / relative
            folder.mkdir(parents=True, exist_ok=True)
            write_json(folder / "config.json", asdict(config))
            write_json(folder / "status.json", {"status": "planned", "step": 0,
                                                 "planned_steps": total_steps, "time": utc_now()})
            append_event(folder, "run_planned", planned_steps=total_steps)
            plan["runs"].append({"method": method, "seed": seed, "folder": relative})
    write_json(plan_path, plan)
    return plan


def execute(until=None):
    plan = json.loads((ROOT / "experiment_plan.json").read_text(encoding="utf-8"))
    budget = plan["steps_per_run"]
    end = min(budget, until if until is not None else budget)
    interval = plan["shared_config"]["eval_interval"]
    milestones = list(range(interval, end + 1, interval))
    if not milestones or milestones[-1] != end:
        milestones.append(end)
    failures = []
    for milestone_index, milestone in enumerate(milestones):
        for seed_index, seed in enumerate(plan["seeds"]):
            methods = list(plan["methods"])
            if (seed_index + milestone_index) % 2:
                methods.reverse()
            for method in methods:
                item = next(r for r in plan["runs"] if r["method"] == method and r["seed"] == seed)
                folder = ROOT / item["folder"]
                status = json.loads((folder / "status.json").read_text(encoding="utf-8"))
                if status["step"] >= milestone:
                    continue
                config = Config(**json.loads((folder / "config.json").read_text(encoding="utf-8")))
                try:
                    print(f"START {method} seed={seed} milestone={milestone}", flush=True)
                    run_training(config, folder, stop_at=milestone)
                except KeyboardInterrupt:
                    raise
                except Exception as error:
                    failures.append({"method": method, "seed": seed, "milestone": milestone,
                                     "error": repr(error), "traceback": traceback.format_exc()})
                    write_json(ROOT / "results" / "main_failures.json", failures)
                    print(f"FAILED {method} seed={seed}: {error!r}", flush=True)
    write_json(ROOT / "results" / "last_session.json", {"finished_utc": utc_now(),
               "requested_common_milestone": end, "planned_budget": budget, "failures": failures})
    if failures:
        raise RuntimeError("Some planned runs failed. Inspect main_failures.json; all folders were retained.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--until", type=int)
    args = parser.parse_args()
    execute(args.until)


if __name__ == "__main__":
    main()
