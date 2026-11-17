# Experiments

These are all the scripts we actually ran, in order, one folder per stage. In each folder:

- the scripts are at the top;
- **`recorded/`** holds the logs and CSVs from the real run: config, evaluations, progress, episodes, updates, `train.log`, `sessions.jsonl`. There are no checkpoints or replay `.npy` files.
- **`output/`** is created when you re-run a script. Re-runs never overwrite `recorded/`. `output/` is git-ignored.

| stage | what | key result |
|---|---|---|
| [01_setup_check](01_setup_check/) | venv, CUDA torch, GPU check | RTX 5050 works with torch 2.15.0+cu132 (after turning off Smart App Control) |
| [02_gpu_throughput](02_gpu_throughput/) | speed and RAM of the notebook code on GPU vs CPU | ~487 (DQN) and ~462 (Double DQN) decisions/s on the GPU, ~5× Tommy's CPU |
| [03_pilot_old_settings](03_pilot_old_settings/) | 1M pilot with the notebook's settings | killed at ~700k (RAM spike during a checkpoint save); still −17 to −20 |
| [04_diagnosis](04_diagnosis/) | why learning was slow | no bug (11/11 tests); too little training and a small network; 2 engineering bugs fixed; config A proposed |
| [05_configA_2M](05_configA_2M/) | config A, DQN seed 17, 0→2M | −21 → −6.8 |
| [06_configA_extend_5M](06_configA_extend_5M/) | resumed 2M → 5M | **+11.4, 10/10 games won** |

Run the scripts with the repo's venv, from any folder. For example: `.venv\Scripts\python experiments\04_diagnosis\test_core.py`. The story is in [`docs/WHAT_WE_DID.md`](../docs/WHAT_WE_DID.md).
