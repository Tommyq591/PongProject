"""Read the notebook (read-only) with nbformat and copy the training definitions
verbatim into pong_train_extract.py. Nothing in the notebook source is altered."""
import sys
from pathlib import Path
import nbformat

NB = Path(sys.argv[1])
OUT = Path(__file__).resolve().parent / "pong_train_extract.py"

# cell index -> text that must appear in that cell (guards against picking a wrong cell)
KEEP = {
    5: "def write_json",            # imports + small I/O helpers
    6: "class Config",              # config, seeding, epsilon schedule
    10: "class PongEnv",            # sticky actions + Atari preprocessing + frame stack
    13: "class QNetwork",           # CNN
    16: "class ReplayBuffer",       # replay + epsilon-greedy action choice
    18: "def bootstrap_values",     # DQN vs Double DQN target
    21: "def learning_update",      # weight update
    27: "def evaluate",             # evaluation (defined only, not called)
    28: "class Trainer",            # training loop
    29: "def run_training",         # resume/eval/checkpoint wrapper (defined only, not called)
}
# Excluded on purpose: 0/2 (repo path discovery), 7/11/14/19/23/24/25 (system info,
# plots, demos, checks that read/write ROOT), 31/32 (launch main training),
# 34+ (results, plots, videos, guided follow-up).

nb = nbformat.read(NB, as_version=4)
parts = ['"""Verbatim training definitions extracted from\n'
         f'{NB.name}\ncells {sorted(KEEP)}. Do not edit; overrides are applied in run_one.py."""\n']
for index, marker in KEEP.items():
    cell = nb.cells[index]
    assert cell.cell_type == "code", index
    assert marker in cell.source, f"cell {index} no longer contains {marker!r}"
    for banned in ("ROOT", "display(", "plt.", "savefig"):
        assert banned not in cell.source, f"cell {index} contains {banned!r}"
    parts.append(f"\n# ===== notebook cell {index} =====\n{cell.source}\n")
OUT.write_text("".join(parts), encoding="utf-8")
print("wrote", OUT, "from cells", sorted(KEEP))
