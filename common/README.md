# Shared code

The demos and experiments import these files. They find this folder relative to their own location, so the repo can live anywhere.

| file | what it is |
|---|---|
| `pong_train_extract.py` | **Verbatim** training code from the submission notebook (cells 5, 6, 10, 13, 16, 18, 21, 27, 28, 29): `Config`, `PongEnv` (sticky actions + Atari preprocessing + frame stack), `QNetwork` (small CNN), `ReplayBuffer`, `choose_action`, DQN / Double DQN targets, `learning_update`, `evaluate`, `Trainer`, `run_training`. **Do not edit by hand.** Regenerate it with `build_extract.py`. |
| `build_extract.py` | Rebuilds `pong_train_extract.py` from `Assignment2/CS4287-Assign2_Atari_ID1_ID2_ID3.ipynb`, reading the notebook only. It checks that each cell still contains the expected definition. |
| `pong_train_diag.py` | Our additions on top of the notebook code. They work for both DQN and Double DQN. `DiagConfig` adds `network = "small" \| "nature"`. `NatureQNetwork` is the standard conv 32-64-64 / dense 512 network. `FastCheckpointTrainer` saves the replay as raw `.npy` (fixing the checkpoint RAM spike) and loads checkpoints on CPU (fixing the CUDA resume crash). |

Typical use (see `experiments/05_configA_2M/run_A.py`):

```python
sys.path.insert(0, str(REPO / "common"))
from pong_train_diag import nb, DiagConfig, FastCheckpointTrainer   # nb = the notebook code
nb.run_training(config, folder, trainer_factory=FastCheckpointTrainer)
```
