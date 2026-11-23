#!/bin/sh
# Sequential throughput suite. Each run is its own process. Outputs stay in gpu_scratch.
cd "$(dirname "$0")"
PY=../PongProject/.venv/Scripts/python
PLAN=../PongProject/Assignment2/experiment_plan.json
rm -rf runs results && mkdir -p results logs
run() { name=$1; shift
  echo "=== $name $(date +%H:%M:%S) gpu: $(nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader) apps: $(nvidia-smi --query-compute-apps=process_name --format=csv,noheader | tr '\n' ' ')"
  $PY -I run_one.py --plan $PLAN --out results/$name.json "$@" > logs/$name.log 2>&1
  echo "    exit=$? $(grep -o '"decisions_per_s_after_learning_starts": [0-9.]*' logs/$name.log)"
}
# --- required ---
run dqn_cuda_s17            --method dqn        --seed 17 --device cuda --decisions 12000
run double_dqn_cuda_s17     --method double_dqn --seed 17 --device cuda --decisions 12000
run dqn_cpu_s17             --method dqn        --seed 17 --device cpu  --decisions 3000
# --- extras: other seeds ---
run dqn_cuda_s29            --method dqn        --seed 29 --device cuda --decisions 12000
run double_dqn_cuda_s29     --method double_dqn --seed 29 --device cuda --decisions 12000
run dqn_cuda_s43            --method dqn        --seed 43 --device cuda --decisions 12000
run double_dqn_cuda_s43     --method double_dqn --seed 43 --device cuda --decisions 12000
# --- extras: CPU comparisons ---
run double_dqn_cpu_s17      --method double_dqn --seed 17 --device cpu  --decisions 3000
run dqn_cpu_s17_replay20k   --method dqn        --seed 17 --device cpu  --decisions 3000 --replay 20000 --tag _r20k
run dqn_cpu_s17_t8          --method dqn        --seed 17 --device cpu  --decisions 3000 --threads 8 --tag _t8
# --- extras: RAM scaling (replay 20k and 200k on GPU) ---
run dqn_cuda_s17_replay20k  --method dqn        --seed 17 --device cuda --decisions 12000 --replay 20000 --tag _r20k
run dqn_cuda_s17_replay200k --method dqn        --seed 17 --device cuda --decisions 12000 --replay 200000 --tag _r200k
# --- extras: sustained/thermal (100k decisions = one full original run length) ---
run double_dqn_cuda_s17_long --method double_dqn --seed 17 --device cuda --decisions 102000 --tag _long
# --- repeat of first run, to check run-to-run variation ---
run dqn_cuda_s17_repeat     --method dqn        --seed 17 --device cuda --decisions 12000 --tag _repeat
echo "=== done $(date +%H:%M:%S)"
