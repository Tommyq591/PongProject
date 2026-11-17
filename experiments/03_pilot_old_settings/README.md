# 03: Pilot with the old (notebook) settings (16 Nov 2026)

**Purpose:** before committing to the main experiment, check that the agent actually learns when given more than 100k decisions.

- **Run:** DQN, seed 17, cuda, replay 100k, 1M decisions planned, evaluated every 50k (5 games).
- **Everything else** comes from `Assignment2/experiment_plan.json`: the small network, epsilon 1.0→0.05 over 80k, and so on.
- **Code:** the notebook's own `run_training` and `Trainer`.

**Commands** (from the repo root):

```powershell
.venv\Scripts\python experiments\03_pilot_old_settings\pilot_run.py --dry-run   # print config only
.venv\Scripts\python experiments\03_pilot_old_settings\pilot_run.py             # ~45 min per 500k; writes output\
.venv\Scripts\python experiments\03_pilot_old_settings\pilot_summary.py         # summary.txt + learning_curve.png -> output\
```

⚠️ This uses the notebook's original `Trainer`, so it **can't resume on the GPU**, and its checkpoint saves need about 8 GB of RAM. That is how the recorded run died.

**Key result:**
- Windows' low-memory protection **killed it at about 700k decisions**, during a checkpoint save. The last good checkpoint was 600k, which is `models/dqn_oldsettings_seed17_0.6M.pt`.
- The score was still **−17 to −20**. Learning only started at about 400k.
- Watching it showed the paddle camping at the bottom: see `results/behaviour/dqn_oldsettings_seed17_0.6M/`.

`recorded/`:
- `train.log` is the console log;
- `run/` holds config, evaluations, progress, episodes, updates and events.
- There is no `sessions.jsonl`, because the process was killed before writing it.
