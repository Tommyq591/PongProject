# Setup (Windows laptop with an NVIDIA GPU)

Tested on 2026-11-16. Full log: [`experiments/01_setup_check/setup_check.txt`](../experiments/01_setup_check/setup_check.txt).

## The machine we tested on

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 5050 Laptop GPU, 8 GB VRAM, driver 610.71 (supports CUDA 13.3) |
| CPU / RAM | Intel Core 5 210H, 15.6 GB RAM |
| Python | 3.12.10 |
| Key packages | torch 2.15.0+cu132, gymnasium 1.4.0, ale-py 0.12.1, numpy 2.5.4 (everything else in `Assignment2/requirements.txt`) |

## Install (PowerShell, from the repo root)

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -r Assignment2\requirements.txt
# requirements.txt installs the CPU-only torch from PyPI. Replace it with the CUDA build of the SAME version:
.venv\Scripts\python -m pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cu132 --no-deps --force-reinstall
# check
.venv\Scripts\python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Expected output: `2.15.0+cu132 True NVIDIA GeForce RTX 5050 Laptop GPU`. Always call `.venv\Scripts\python` directly. You don't need to activate the venv.

Which CUDA index to use: the PyTorch indexes we checked carried these torch versions: cu128 → 2.11.0, cu129 → 2.9.0, cu130 → 2.14.1, **cu132 → 2.15.0**, cu133 → none. cu132 (CUDA 13.2) is the only one with the pinned 2.15.0, and a driver that supports CUDA 13.3 runs it. The RTX 50-series (Blackwell, sm_120) needs one of these recent builds.

## Problem we hit: Windows Smart App Control

On the first run, `import torch` failed with `OSError: [WinError 4551] An Application Control policy has blocked this file ... shm.dll`, and `import ale_py` failed with `DLL load failed ... blocked`.

- **Cause:** Windows 11 **Smart App Control** was on. To check, open the registry key `HKLM:\SYSTEM\CurrentControlSet\Control\CI\Policy`; `VerifiedAndReputablePolicyState = 1` means it is on.
- **Fix:** turn it off in Windows Security > App & browser control > Smart App Control. ⚠️ On many Windows 11 builds you **can't turn it back on** without resetting Windows, so decide for yourself.

## GPU speed (throughput test)

Full report: [`experiments/02_gpu_throughput/`](../experiments/02_gpu_throughput/). Machine-readable results: [`results/gpu_throughput.json`](../results/gpu_throughput.json).

These numbers come from the notebook's own training code with the original (small) network. "dec/s" means decisions per second, measured after learning starts:

| setup | DQN dec/s | Double DQN dec/s |
|---|---|---|
| this laptop, GPU (cuda) | 486.8 ± 6.4 (4 runs) | 461.5 ± 4.1 (3 runs) |
| this laptop, CPU | 349.4 | 313.8 |
| Tommy's laptop, CPU (i7-1360P, from `Assignment2/results/throughput.json`) | 97.2 | – |

- The GPU is only ~1.4× this laptop's CPU, because GPU utilisation is about 16%. The network is tiny, so most of the time goes on the emulator, replay sampling and Python overhead. This laptop is still about 5× faster than the CPU run the notebook was planned on.
- Speed falls to about 360–390 dec/s once most actions come from the network (low epsilon).
- With **config A** (bigger "Nature" network, see [CONFIG_A.md](CONFIG_A.md)), the real long runs averaged about **375–390 dec/s**. The 2M-decision run took 1.6 h of wall-clock time, and the 2M→5M extension took 2.5 h, both including evaluation.
- **RAM:** each replay slot costs about 14 KB. A 200k replay uses about 4.5 GB steady (peak 6.0 GB measured over the 5M run) with our fixed checkpointing. The notebook's original checkpoint code needs about **3.9× the replay size** extra RAM while saving. That is why the old-settings pilot was killed: see [WHAT_WE_DID.md](WHAT_WE_DID.md).
- Close other GPU users before training. An emulator running in the background cut Double DQN to 126 dec/s and invalidated one measurement.

## Before a long run

- Plug in and stop Windows from sleeping, e.g. `powercfg /change standby-timeout-ac 0`. Write down the old value first (`powercfg /q`) so you can restore it.
- Have at least about 6 GB of RAM free (close browsers and chat apps), and leave the GPU otherwise idle.
- Use `--dry-run` on the run scripts first to print the exact configuration.
