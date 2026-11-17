# 04: Diagnosis: why did learning look so slow? (16 Nov 2026)

**Purpose:** decide whether slow learning was a bug, bad settings or simply too little training, before tuning anything. **Read [`report.txt`](report.txt)**: it is the full write-up.

**Commands** (from the repo root; each writes to `experiments\04_diagnosis\output\`):

```powershell
.venv\Scripts\python experiments\04_diagnosis\test_core.py              # 11 correctness tests (~1 min)
.venv\Scripts\python experiments\04_diagnosis\obs_images.py             # what the network sees -> observations.png
.venv\Scripts\python experiments\04_diagnosis\plot_pilot_logs.py        # loss / Q / TD error of the 03 pilot
.venv\Scripts\python experiments\04_diagnosis\test_fast_checkpoint.py   # tests the .npy checkpoint fix (RAM + reload) (~3 min)
.venv\Scripts\python experiments\04_diagnosis\bench_configs.py          # speed: old settings vs config A vs config B (~3 min)
.venv\Scripts\python experiments\04_diagnosis\verify_ram.py             # checks the Atari RAM addresses used by explain_ai.py
.venv\Scripts\python experiments\04_diagnosis\compare_metrics.py configA_5.0M models\dqn_configA_seed17_5.0M.pt --games 3
```

**Key results:**
- **No learning bug: 11/11 tests pass.** We checked frame stacking, the replay buffer, the DQN and Double DQN targets, the loss, target sync, epsilon, sticky actions, reward clipping and actions. [`observations.png`](observations.png) shows the 4 stacked 84×84 frames the network sees.
- **The cause** was too little training (100k = 1% of the reference's 10M), a network about 4× smaller than standard, and a small replay buffer.
- **Two bugs were fixed** in `common/pong_train_diag.py`:
  - the checkpoint RAM spike: peak extra RAM went from +5.1 GB to about 0;
  - the CUDA resume crash in `Trainer.load`.
- **Config A** was proposed. It costs no extra time on the GPU: 504 and 387 decisions/s (exploring and greedy) against 494 and 376 for the old settings.

`recorded/` holds the outputs of the real runs: `test_results.json`, `test_fast_checkpoint.json`, `observations.json`, `pilot_learning_diagnostics.png/.json`, `bench_configs.json`, `ram_verification.json`, `metrics_table.csv` and `metrics/baseline_pilot_600k/`.
