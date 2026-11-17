# 01: Setup check (16 Nov 2026)

**Purpose:** check whether the RTX 5050 laptop GPU can train the notebook's code. No training was run.

**Commands** (PowerShell, from the repo root; the full list is in `setup_check.txt`):

```powershell
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -r Assignment2\requirements.txt
.venv\Scripts\python -m pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cu132 --no-deps --force-reinstall
.venv\Scripts\python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

The verification itself was a throwaway script (torch and CUDA info, a GPU matmul, creating `ALE/Pong-v5`). Its output is in `setup_check.txt`.

**Key result:** it works: torch 2.15.0+cu132, CUDA available, sm_120. Windows **Smart App Control** blocked the torch and ale_py DLLs and had to be turned off. Details: [`docs/SETUP.md`](../../docs/SETUP.md).
