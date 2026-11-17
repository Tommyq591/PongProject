# Config A: every setting compared

The table compares three configurations:

- **Tommy's original:** `shared_config` in `Assignment2/experiment_plan.json`, used for the CPU runs.
- **Config A:** what the 5M-decision run used (`experiments/05_configA_2M/run_A.py`, recorded in `experiments/06_configA_extend_5M/recorded/run/config.json`).
- **CleanRL:** the reference implementation `dqn_atari.py` with its default arguments.
  - Docs: https://docs.cleanrl.dev/rl-algorithms/dqn/#dqn_ataripy
  - Code: https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/dqn_atari.py
  - It reports **20.25 ± 0.41** on PongNoFrameskip-v4 after 10M steps.

Config A changes **only bolded rows**, and it applies them identically to DQN and Double DQN.

| setting | Tommy's original | **config A** | CleanRL reference | why config A |
|---|---|---|---|---|
| method | dqn / double_dqn | dqn (seed 17 so far) | dqn | – |
| **total_steps** (decisions) | 100,000 | **2,000,000 → extended to 5,000,000** | 10,000,000 | biggest factor: 100k was 1% of the reference budget |
| **network** | small CNN: conv 16-32-32, dense 256 (~425k params) | **Nature CNN: conv 32-64-64, dense 512 (~1.69M params)** | Nature CNN | standard network; no extra cost on the GPU |
| **replay_capacity** | 20,000 (pilot: 100,000) | **200,000** | 1,000,000 | 1M would need ~14 GB RAM; 200k is the practical maximum on 15.6 GB |
| **learning_starts** | 2,000 | **20,000** | 80,000 | avoid early updates on a tiny buffer |
| **target_update_interval** | 2,000 | **1,000** | 1,000 (hard copy, tau 1.0) | match the reference |
| epsilon_start | 1.0 | 1.0 | 1.0 | – |
| **epsilon_end** | 0.05 | **0.01** | 0.01 | match the reference |
| **epsilon_decay_steps** | 80,000 | **100,000** | 1,000,000 (10% of 10M) | ~5% of a 2M budget; kept short |
| batch_size | 32 | 32 | 32 | same |
| train_frequency | 1 update / 4 decisions | 1 / 4 | 1 / 4 | same |
| gamma | 0.99 | 0.99 | 0.99 | same |
| optimiser / learning_rate | Adam, 1e-4 | Adam, 1e-4 | Adam, 1e-4 | same |
| loss | Huber (smooth L1) | Huber | MSE | kept (Huber = Nature DQN error clipping) |
| max_grad_norm | 10 | 10 | none | kept (never reached in practice) |
| env_id | ALE/Pong-v5 | ALE/Pong-v5 | PongNoFrameskip-v4 | kept (modern standard) |
| sticky actions | p = 0.25, training AND evaluation | 0.25 | none | kept: harder but fairer protocol, same for both methods |
| frame_skip / screen / stack | 4 (max-pool) / 84×84 grey / 4 | same | 4 / 84×84 / 4 | same |
| noop_max | 30 | 30 | 30 | same |
| reward clipping | sign(r) | sign(r) | sign(r) | same |
| max_episode_steps | 27,000 | 27,000 | (env default) | – |
| **eval_interval** | 20,000 | **100,000** | – | evaluation is costly when games get long |
| eval_episodes / final_eval_episodes | 5 / 10 | 5 / 10 | – | same |
| eval_epsilon | 0.01 | 0.01 | – | same |
| **checkpoint_interval** | 10,000 | **100,000** | – | each checkpoint is ~2.7 GB on disk (replay as .npy) |
| **device** | cpu | **cuda** | cuda | GPU |
| torch_threads | 4 | 4 | – | same |
| checkpoint format | notebook `Trainer.save` (replay pickled into gzip) | **`FastCheckpointTrainer`: replay as raw .npy + small checkpoint.pt.gz** | – | fixes the RAM spike (see WHAT_WE_DID.md) |
| GPU resume | crashes (`Trainer.load` maps the RNG state to cuda) | **fixed (load to CPU first)** | – | bug fix |

## Measured cost of config A on this laptop

- Speed: 504 (exploring) and 387 (greedy) dec/s for config A, against 494 and 376 for the old settings. These come from 8k-decision benchmarks (`experiments/04_diagnosis/recorded/bench_configs.json`).
- Real runs: about 375–390 dec/s. 3M decisions took about 2.5 h including evaluation.
- RAM: 4.2 GB peak for 0–2M and 6.0 GB peak for 2–5M (from `sessions.jsonl`).
- Disk: about 2.7 GB per run folder while training, almost all of it the replay `.npy` files. Never commit these.

Config B, which was A plus one update per decision, was proposed but never run: it would be about 2.6× slower.
