# 05: Config A, DQN seed 17, 0 → 2M decisions (16 Nov 2026)

**Purpose:** the first real test of [config A](../../docs/CONFIG_A.md): the Nature network, replay 200k, target sync 1,000, epsilon →0.01 over 100k and learning starts at 20k.
- It evaluates every 100k decisions (5 games) and runs a 10-game final evaluation.
- It checkpoints every 100k so the run can be resumed.
- It keeps permanent policy snapshots every 500k.
- It uses `FastCheckpointTrainer`, with the RAM and CUDA-resume fixes.

**Commands** (from the repo root):

```powershell
.venv\Scripts\python experiments\05_configA_2M\run_A.py --dry-run     # print the exact config
.venv\Scripts\python experiments\05_configA_2M\run_A.py               # ~1.6 h; re-run the same command to resume
```

The recorded run was launched as a hidden, detached process so it kept going without a terminal:

```powershell
New-Item -ItemType Directory -Force experiments\05_configA_2M\output | Out-Null
Start-Process cmd.exe -WindowStyle Hidden -ArgumentList '/c ".venv\Scripts\python.exe -u experiments\05_configA_2M\run_A.py >> experiments\05_configA_2M\output\train.log 2>&1"'
```

Sleep was disabled first. The old values are in `recorded/previous_sleep_settings.txt`.

**Key result:**
- The score rose from **−20.8 at the start to −6.8 at 2M** (10-game final; range −16 to +7), with steady improvement from about 300k.
- It took 1.60 h wall-clock at about 390 decisions/s, with a 4.2 GB RAM peak.
- Snapshots are `models/dqn_configA_seed17_0.5M.pt` … `2.0M.pt`.

`recorded/` contains:
- `train.log` and `sessions.jsonl` (wall-clock and peak RAM);
- `run/`: config, evaluations, progress, episodes, updates, events and status.
