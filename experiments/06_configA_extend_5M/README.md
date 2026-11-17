# 06: Config A, extended from 2M to 5M (16–17 Nov 2026)

**Purpose:** at 2M the agent was still losing, but still improving. This stage continues the **same run** to 5M decisions.

**Exactly what changed:**
- The 2M run folder was copied, and in the copy's `run/config.json` **only** `"total_steps": 2000000` became `5000000`. That is a single byte; every other setting is identical.
- `run_training` requires the saved config to match, and the checkpoint loader allows only `total_steps` to differ.
- `run_A_ext.py` repeats this copy-and-edit automatically the first time you run it. It needs a finished `../05_configA_2M/output/`.

**Commands** (from the repo root):

```powershell
.venv\Scripts\python experiments\06_configA_extend_5M\run_A_ext.py --dry-run
.venv\Scripts\python experiments\06_configA_extend_5M\run_A_ext.py     # ~2.5 h; re-run to resume
```

**Key result:**
- The score crossed 0 at about **2.4M** and reached **+11.4 at 5M** in the 10-game final evaluation. It **won all 10 games** (per-game scores +5 to +16).
- It took 2.48 h wall-clock at about 375–390 decisions/s, with a 6.0 GB RAM peak.
- **Unexplained:** the first evaluation after resuming, at 2.1M, dropped to −17.8. It was −6.8 at 2M and −2.0 at 2.2M. See [`docs/WHAT_WE_DID.md`](../../docs/WHAT_WE_DID.md).
- Snapshots are `models/dqn_configA_seed17_2.5M.pt` … `5.0M.pt`. The curve is `results/learning_curve_configA_5M.png`.

In `recorded/`:
- `train.log` contains the copied 0–2M log, then a line `===== EXTENSION to 5M launched ... =====`, then 2M–5M.
- `sessions.jsonl` has both sessions.
- `run/evaluations.csv` and the other CSVs cover the full 0–5M.
