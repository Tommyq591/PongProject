# 02: GPU throughput test (16 Nov 2026)

**Purpose:** measure how fast the notebook's own training code runs on this laptop's GPU vs its CPU, how much RAM each replay size needs, and how long a full experiment would take. The code was unchanged; only config overrides were applied.

**Commands** (Git Bash, from this folder):

```bash
bash run_all.sh                          # 14 timed runs, one process each -> output/results, output/logs, output/runs
../../.venv/Scripts/python summarise.py  # -> output/gpu_throughput.json (uses recorded/ if output/ has no results yet)
# one run on its own:
../../.venv/Scripts/python run_one.py --method dqn --seed 17 --device cuda --decisions 12000 \
    --plan ../../Assignment2/experiment_plan.json --out output/one.json
```

**Key result** (`recorded/gpu_throughput.txt` is the full report):
- GPU: DQN **486.8 ± 6.4** and Double DQN **461.5 ± 4.1** decisions/s. This laptop's CPU managed 349 and 314; Tommy's CPU managed 97.
- GPU utilisation is only about 16%, because the small network makes the emulator and Python the bottleneck.
- The notebook's checkpoint save needs about **3.9× the replay size** in extra RAM: an 8.25 GB peak with a 100k replay. A 200k replay would not fit in 15.6 GB, which led to the fix in `04_diagnosis`.

`recorded/` contains:
- `gpu_throughput.txt/.json` (report);
- `results/*.json` and `logs/*.log` (one per run);
- `runs/<run>/` (each Trainer's config and CSVs);
- `first_single_runs/`, the first two measurements. These are superseded: an emulator was using the GPU at the time.
