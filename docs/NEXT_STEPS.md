# Next steps: the main experiment

This is a **proposal for the team to agree on.** Nothing here has been run or changed in the notebook yet.

## Proposed experiment

- **Question** (unchanged): DQN vs Double DQN on Pong under equal budgets.
- **Runs:** DQN and Double DQN × seeds 17, 29, 43, giving 6 runs.
- **Settings:** [config A](CONFIG_A.md) for every run, identical for both methods. The only difference between methods is the target: `max_a Q_target(s', a)` vs `Q_target(s', argmax_a Q_online(s', a))`.
- **Budget:** **3,000,000 decisions per run, run straight through.** Our seed-17 DQN went positive at about 2.4M and was around +7 to +10 by 3M (see [`results/learning_curve_configA_5M.png`](../results/learning_curve_configA_5M.png)). So 3M should show both whether each method learns and how fast.
- **Evaluation:** as in the plan, with ε = 0.01 on held-out seeds. That means 5 games every 100k decisions and 10 final games. The primary metric is the mean final score over the 3 seeds. Learning speed is the area under the curve and the first checkpoint with mean ≥ 0.
- **Run order:** the old plan says "round-robin at evaluation milestones". We propose running each run **from start to finish, one after another**, so nothing has to be resumed mid-run. Resuming worked, but see the unexplained dip at 2.1M in [WHAT_WE_DID.md](WHAT_WE_DID.md).
- **Time on the RTX 5050 laptop:** about 2.4 h per DQN run (0–2M took 1.6 h; 1M more took about 0.8 h) and about 2.5 h per Double DQN run (about 5% slower). **About 15 h for all 6.** That could be two overnight sessions of 3 runs each.
- **Resources:** about 6 GB of RAM per run and about 2.7 GB of disk per run folder while training. Delete the replay `.npy` files after a run finishes; keep CSVs, logs and policy snapshots.
- **Reusing the existing 5M run:** we suggest **not** using it as the DQN seed-17 run. It was resumed at 2M, its final 10-game evaluation is at 5M rather than 3M, and the dip at 2.1M is unexplained. Run all 6 fresh so they are directly comparable. Keep the 5M run as a "longer training" illustration.

## Notebook changes this needs

These go in `Assignment2/CS4287-Assign2_Atari_ID1_ID2_ID3.ipynb`. Cell numbers are the code cells that `common/build_extract.py` reads. All of them are already implemented and tested outside the notebook in `common/pong_train_diag.py`, so they can be copied in.

1. **GPU device.**
   - Use `device="cuda"` in the run configs: the `Config` default `device: str = "cpu"` (cell 6) and the plan's `shared_config`.
   - The guided follow-up section also hard-codes `config.device = "cpu"`.
2. **Nature network option.**
   - Add `network: str = "small"` to `Config` (cell 6) and a `NatureQNetwork` (conv 32-64-64, dense 512) next to `QNetwork` (cell 13).
   - Pick the network from the config wherever `QNetwork(...)` is built (Trainer, evaluation, video and analysis cells).
3. **CUDA resume fix.** In `Trainer.load()` (cell 28), use `torch.load(..., map_location="cpu")` and move tensors to the device afterwards. Without this, any resume on the GPU crashes with `RNG state must be a torch.ByteTensor`.
4. **Checkpoint RAM fix.**
   - In `Trainer.save()` and `load()` (cell 28), save the replay arrays with `np.save` instead of pickling them into `checkpoint.pt.gz`.
   - With a 200k replay, the current code needs about 14.7 GB of RAM while saving and will crash on a 16 GB laptop.
   - Keep the ordered swap (`replay.tmp` → `replay`, keeping `replay.old`) so a crash mid-save is safe.
5. **New experiment plan.**
   - `experiment_plan.json` was declared *before* the main runs (100k steps, CPU). Rather than overwrite it, add a new declared plan, e.g. `experiment_plan_v2.json`, with config A, 3M steps and sequential run order. Explain in the notebook why the plan changed: the diagnosis showed 100k was far too short. This keeps the "declared in advance" story honest.
   - Update the time estimate in the plan: about 15 h on the GPU.
6. **Committing results.** Today `Assignment2/runs/` commits `checkpoint.pt.gz` files of about 9 MB each. With config A, a full checkpoint is about 2.7 GB, too big for git and GitHub. Commit only `config.json`, the CSV logs, `policy.pt` and the snapshots, and add the replay folders to `.gitignore`.
7. **Guided follow-up section (section 17).**
   - It was a workaround for agents that didn't learn. Decide whether to keep it as an extra, rerun it with config A, or drop it.
   - Either way, it must stay identical for both methods.
8. Refresh the plots and gameplay videos from the new runs, and fill in team names, IDs and the filename (see [`submission/README.md`](../submission/README.md)).

## Before starting

- Run `experiments/04_diagnosis/test_core.py` (11 tests, about 1 min). It should print `11/11 tests passed`.
- Use `--dry-run` on the run script to print the exact config.
- Close browsers, chat apps and emulators. Plug in and disable sleep (see [SETUP.md](SETUP.md)).
