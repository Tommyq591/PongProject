# Demos

Run these from the repo root in PowerShell. They work from any folder, because every path is relative to the script.

## Play Pong yourself: `play_pong.py`

```powershell
.venv\Scripts\python demos\play_pong.py
.venv\Scripts\python demos\play_pong.py --zoom 4      # bigger window
```

**W / Up** moves up, **S / Down** moves down, **Space** serves, **R** starts a new game and **Esc** quits. You are the green paddle on the right.

> ⚠️ **The opponent here is the Atari game's own built-in computer player, NOT our AI.** Our AI was trained to play against that same built-in bot. To watch our AI, use `watch_ai.py`.

This is based on Tommy's `Pong_Practice.ipynb` (same keys and environment, every frame shown, no sticky actions). It draws the game with tkinter, so it doesn't need pygame. `--selftest` runs 600 frames without a window to check the setup.

## Watch our AI play: `watch_ai.py`

```powershell
.venv\Scripts\python demos\watch_ai.py                                              # best model: 5.0M config A
.venv\Scripts\python demos\watch_ai.py models\dqn_oldsettings_seed17_0.6M.pt        # the early "bottom-camping" agent
.venv\Scripts\python demos\watch_ai.py models\dqn_configA_seed17_2.0M.pt --games 3 --speed 2
.venv\Scripts\python demos\watch_ai.py --no-window                                  # no window, just the scores
```

- Our AI is **green (right)**; the built-in bot is orange (left).
- The script uses exactly the training environment (sticky actions, frame skip 4, 84×84 greyscale, 4-frame stack) and acts like the notebook's evaluation: ε = 0.01, held-out seeds.
- It prints each point and the final score. The model files are listed in [`models/README.md`](../models/README.md).

## See what the AI is thinking: `explain_ai.py`

```powershell
.venv\Scripts\python demos\explain_ai.py                                            # 5.0M model, live window
.venv\Scripts\python demos\explain_ai.py models\dqn_oldsettings_seed17_0.6M.pt
.venv\Scripts\python demos\explain_ai.py --games 3 --no-window                      # fast, no window
```

- The live window shows the game next to the 6 Q-values, with the chosen action highlighted. **Space** pauses and **q** stops.
- Afterwards it saves plots and a text summary to `demos\explain_output\<model>_<time>\`: where the paddle sits, whether it tracks the ball, how early it "realises" it will lose a point, and whether missed balls were reachable. That folder is git-ignored.
- Ball and paddle positions come from Atari RAM. The addresses were checked against screen pixels by `experiments/04_diagnosis/verify_ram.py`.

## `Pong_Practice.ipynb`

This is Tommy's original practice notebook. It needs `pygame` (`.venv\Scripts\python -m pip install pygame`), which `play_pong.py` doesn't.
