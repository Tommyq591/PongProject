"""Read real CSV logs only; never create or fill in missing game scores."""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLORS = {"dqn": "#2171b5", "double_dqn": "#d94801"}
LABELS = {"dqn": "DQN", "double_dqn": "Double DQN"}


def read_logs(root=ROOT):
    root = Path(root)
    plan = json.loads((root / "experiment_plan.json").read_text(encoding="utf-8"))
    evaluations, progress, updates, episodes, inventory = [], [], [], [], []
    for run in plan["runs"]:
        folder = root / run["folder"]
        status = json.loads((folder / "status.json").read_text(encoding="utf-8"))
        inventory.append({"method": run["method"], "seed": run["seed"], **status})
        for filename, collection in (("evaluations.csv", evaluations), ("progress.csv", progress),
                                     ("updates.csv", updates), ("episodes.csv", episodes)):
            path = folder / filename
            if path.exists() and path.stat().st_size:
                frame = pd.read_csv(path)
                frame["method"], frame["seed"] = run["method"], run["seed"]
                collection.append(frame)
    combine = lambda frames: pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return plan, combine(evaluations), combine(progress), combine(updates), combine(episodes), pd.DataFrame(inventory)


def make_analysis(root=ROOT):
    root = Path(root)
    output = root / "results"
    output.mkdir(exist_ok=True)
    plan, evaluations, progress, updates, episodes, inventory = read_logs(root)
    inventory.to_csv(output / "run_inventory.csv", index=False)
    if evaluations.empty:
        raise RuntimeError("No evaluation scores exist. Run the experiment before reporting results.")
    curve = evaluations.groupby(["method", "seed", "step"], as_index=False).agg(
        mean_score=("score", "mean"), game_sd=("score", "std"), games=("score", "size"),
        mean_length=("length", "mean"), eval_train_seconds=("train_elapsed_s", "max"))
    curve.to_csv(output / "learning_curve.csv", index=False)
    observed_horizons = curve.groupby(["method", "seed"])["step"].max()
    all_evaluated = len(observed_horizons) == len(plan["runs"])
    common_horizon = int(observed_horizons.min()) if all_evaluated else 0
    per_run = []
    for run in plan["runs"]:
        method, seed = run["method"], run["seed"]
        rows = curve[(curve.method == method) & (curve.seed == seed)].sort_values("step")
        state = inventory[(inventory.method == method) & (inventory.seed == seed)].iloc[0]
        if rows.empty:
            per_run.append({"method": method, "seed": seed, "status": state["status"],
                            "trained_steps": int(state["step"]), "latest_evaluation_step": np.nan})
            continue
        latest = rows.iloc[-1]
        eligible = rows[rows.step <= common_horizon]
        crossing = eligible[eligible.mean_score >= plan["threshold"]]
        auc = (float(np.trapezoid(eligible.mean_score.to_numpy(), eligible.step.to_numpy()) / common_horizon)
               if common_horizon > 0 and len(eligible) >= 2 else np.nan)
        complete = (state["status"] == "complete" and int(latest.step) == plan["steps_per_run"]
                    and int(latest.games) >= plan["shared_config"]["final_eval_episodes"])
        per_run.append({"method": method, "seed": seed, "status": state["status"],
                        "trained_steps": int(state["step"]),
                        "latest_evaluation_step": int(latest.step),
                        "latest_mean_score": float(latest.mean_score),
                        "within_run_game_sd": float(latest.game_sd),
                        "evaluation_games": int(latest.games),
                        "mean_episode_decisions": float(latest.mean_length),
                        "training_seconds_to_latest_evaluation": float(latest.eval_train_seconds),
                        "auc_to_common_horizon": auc,
                        "first_nonlosing_step": float(crossing.step.iloc[0]) if len(crossing) else np.nan,
                        "main_budget_complete": bool(complete)})
    per_run = pd.DataFrame(per_run)
    per_run.to_csv(output / "comparison_per_seed.csv", index=False)

    # Aggregate ONLY a shared evaluated horizon; do not compare unequal endpoints.
    shared_rows = curve[curve.step == common_horizon] if common_horizon > 0 else curve.iloc[:0]
    summary = shared_rows.groupby("method", as_index=False).agg(
        training_seeds=("seed", "nunique"), mean_score=("mean_score", "mean"),
        across_seed_sd=("mean_score", "std"), mean_episode_decisions=("mean_length", "mean"))
    summary["comparison_step"] = common_horizon
    auc_by_method = per_run.groupby("method")["auc_to_common_horizon"].mean()
    summary["mean_auc_score"] = summary["method"].map(auc_by_method)
    summary.to_csv(output / "comparison_methods.csv", index=False)

    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(10, 5.2), constrained_layout=True)
    for method in plan["methods"]:
        subset = curve[curve.method == method]
        for seed, group in subset.groupby("seed"):
            ax.plot(group.step, group.mean_score, color=COLORS[method], alpha=.32,
                    linewidth=1, marker=".", label=f"{LABELS[method]} seed {seed}")
        aggregate = subset.groupby("step").mean_score.agg(["mean", "std", "count"])
        # Only draw the aggregate where all three declared seeds contributed.
        aggregate = aggregate[aggregate["count"] == len(plan["seeds"])]
        if len(aggregate):
            ax.plot(aggregate.index, aggregate["mean"], color=COLORS[method], linewidth=2.6,
                    marker="o", label=f"{LABELS[method]} mean ({len(plan['seeds'])} seeds)")
            ax.fill_between(aggregate.index, aggregate["mean"] - aggregate["std"],
                            aggregate["mean"] + aggregate["std"], color=COLORS[method], alpha=.12)
    ax.axhline(0, color="gray", linestyle="--", linewidth=1, label="Declared non-losing threshold")
    ax.set(xlabel="Training decisions (one decision usually = 4 game frames)",
           ylabel="Mean original Pong score", ylim=(-22, 22),
           title="Held-out Pong scores (identical curves overlap); shading = across-seed SD")
    ax.legend(fontsize=8, ncol=2, loc="upper left")
    fig.savefig(output / "learning_curves.png", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for method in plan["methods"]:
        sub = shared_rows[shared_rows.method == method]
        if len(sub):
            x = plan["methods"].index(method)
            offsets = np.linspace(-.12, .12, len(sub))
            axes[0].scatter(x + offsets, sub.mean_score, color=COLORS[method], s=65)
            for offset, (_, row) in zip(offsets, sub.iterrows()):
                axes[0].annotate(str(int(row.seed)), (x+offset, row.mean_score), xytext=(2, 7),
                                 textcoords="offset points", fontsize=8)
    axes[0].set(xticks=[0, 1], xticklabels=["DQN", "Double DQN"], ylim=(-22, 22),
                ylabel="Mean original score per training seed",
                title=f"Equal-budget scores\nat {common_horizon:,} decisions")
    if not updates.empty:
        for method in plan["methods"]:
            for index, (seed, group) in enumerate(updates[updates.method == method].groupby("seed")):
                axes[1].plot(group.step, group.loss.rolling(10, min_periods=1).mean(),
                             color=COLORS[method], alpha=.5, linewidth=1,
                             label=LABELS[method] if index == 0 else None)
    axes[1].set(xlabel="Training decisions", ylabel="Huber loss (10-log rolling mean)",
                title="Learning loss\n(diagnostic only)")
    axes[1].legend(fontsize=9)
    fig.savefig(output / "scores_and_loss.png", dpi=170)
    plt.close(fig)

    total_games = len(evaluations)
    completed = int(per_run.get("main_budget_complete", pd.Series(dtype=bool)).fillna(False).sum())
    findings = {"common_evaluated_horizon": common_horizon,
                "planned_steps_per_run": plan["steps_per_run"], "completed_main_runs": completed,
                "planned_runs": len(plan["runs"]), "actual_evaluation_games": total_games,
                "actual_training_decisions": int(inventory["step"].sum()),
                "actual_training_games_logged": len(episodes),
                "any_truncated_evaluation": bool(evaluations.truncated.any()),
                "all_planned_runs_evaluated": all_evaluated}
    if len(summary) == 2:
        paired = shared_rows.pivot(index="seed", columns="method", values="mean_score")
        differences = paired["double_dqn"] - paired["dqn"]
        findings["paired_score_difference_double_minus_dqn"] = float(differences.mean())
        findings["paired_score_differences_by_seed"] = {str(k): float(v) for k, v in differences.items()}
        findings["both_methods_reached_threshold"] = bool(per_run.first_nonlosing_step.notna().all())
    write_text = json.dumps(findings, indent=2, allow_nan=False)
    (output / "findings.json").write_text(write_text, encoding="utf-8")
    print(inventory[["method", "seed", "status", "step", "planned_steps"]].to_string(index=False))
    print(summary.to_string(index=False))
    print(write_text)
    return findings, per_run, summary


if __name__ == "__main__":
    make_analysis()
