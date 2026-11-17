# Trained models

These are policy snapshots: the online Q-network weights plus the config they were trained with. Each file is about 6.8 MB (1.7 MB for the old-settings model). The folder holds **no replay buffers or `checkpoint.pt.gz` files**, so these can't be used to resume training. They are for watching, analysing and evaluating.

- **Evaluation score:** the score recorded *during training* at that step: our points minus the opponent's, mean over held-out games, ε = 0.01, sticky actions on. Source: the run's `evaluations.csv` in `experiments/`.
- **Settings:** described in [`docs/CONFIG_A.md`](../docs/CONFIG_A.md). In short, *config A* is the Nature CNN with replay 200k. *Old settings* is the notebook's small CNN (plan settings, replay 100k).
- **Where they come from:** the 0.5M–2.0M snapshots come from the 2M run (`experiments/05_configA_2M`). The 2.5M–5.0M snapshots come from the extension that resumed it (`experiments/06_configA_extend_5M`). They are the same training run.

Run the commands from the repo root:

| file | method | settings | training steps (decisions) | evaluation score at that step | watch it |
|---|---|---|---|---|---|
| `dqn_oldsettings_seed17_0.6M.pt` | DQN | old settings (notebook defaults, replay 100k), seed 17 | 600,000 | −19.6 (5 games, range −21 to −18) | `.venv\Scripts\python demos\watch_ai.py models\dqn_oldsettings_seed17_0.6M.pt` |
| `dqn_configA_seed17_0.5M.pt` | DQN | config A, seed 17 | 500,000 | −18.0 (5 games, range −21 to −12) | `.venv\Scripts\python demos\watch_ai.py models\dqn_configA_seed17_0.5M.pt` |
| `dqn_configA_seed17_1.0M.pt` | DQN | config A, seed 17 | 1,000,000 | −9.2 (5 games, range −13 to −4) | `.venv\Scripts\python demos\watch_ai.py models\dqn_configA_seed17_1.0M.pt` |
| `dqn_configA_seed17_1.5M.pt` | DQN | config A, seed 17 | 1,500,000 | −11.4 (5 games, range −17 to −7) | `.venv\Scripts\python demos\watch_ai.py models\dqn_configA_seed17_1.5M.pt` |
| `dqn_configA_seed17_2.0M.pt` | DQN | config A, seed 17 | 2,000,000 | −6.8 (10 games, range −16 to +7) | `.venv\Scripts\python demos\watch_ai.py models\dqn_configA_seed17_2.0M.pt` |
| `dqn_configA_seed17_2.5M.pt` | DQN | config A, seed 17 | 2,500,000 | +3.8 (5 games, range −6 to +12) | `.venv\Scripts\python demos\watch_ai.py models\dqn_configA_seed17_2.5M.pt` |
| `dqn_configA_seed17_3.0M.pt` | DQN | config A, seed 17 | 3,000,000 | +7.2 (5 games, range −2 to +14) | `.venv\Scripts\python demos\watch_ai.py models\dqn_configA_seed17_3.0M.pt` |
| `dqn_configA_seed17_3.5M.pt` | DQN | config A, seed 17 | 3,500,000 | +10.0 (5 games, range +6 to +15) | `.venv\Scripts\python demos\watch_ai.py models\dqn_configA_seed17_3.5M.pt` |
| `dqn_configA_seed17_4.0M.pt` | DQN | config A, seed 17 | 4,000,000 | +9.2 (5 games, range +3 to +14) | `.venv\Scripts\python demos\watch_ai.py models\dqn_configA_seed17_4.0M.pt` |
| `dqn_configA_seed17_4.5M.pt` | DQN | config A, seed 17 | 4,500,000 | +10.2 (5 games, range +5 to +16) | `.venv\Scripts\python demos\watch_ai.py models\dqn_configA_seed17_4.5M.pt` |
| `dqn_configA_seed17_5.0M.pt` | DQN | config A, seed 17 | 5,000,000 | **+11.4 (10 games, all won, range +5 to +16)** | `.venv\Scripts\python demos\watch_ai.py models\dqn_configA_seed17_5.0M.pt` |

To see what a model is "thinking" (live Q-values, behaviour plots), run `.venv\Scripts\python demos\explain_ai.py models\<file>`.

There are no Double DQN models with config A yet. See [`docs/NEXT_STEPS.md`](../docs/NEXT_STEPS.md).

File format (PyTorch): a dict `{"config": {...}, "step": int, "online": state_dict}`. Load it with `torch.load(path, map_location="cpu", weights_only=True)`.
