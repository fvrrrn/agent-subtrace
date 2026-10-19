#!/usr/bin/env bash
set -euo pipefail
root=${1:?Experiment root required}
export CUDA_VISIBLE_DEVICES=2
export HF_HOME=/dev/shm/danefedov-kev-rjudge-20261013/huggingface
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export PYTHONPATH=/dev/shm/danefedov-kev4b-fast-serving-20261014/deps:/home/danefedov/kev-rjudge-baseline-20261013/source:/dev/shm/danefedov-kev08-bidirectional-v1/src
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
export TRITON_F32_DEFAULT=ieee TRITON_CACHE_DIR=$root/triton
python=/dev/shm/danefedov-kev-rjudge-20261013/venv/bin/python
for phase in trained compare; do
  "$python" -u -m graph_safety.step_training "$phase" --root "$root/results" > "$root/${phase}.log" 2>&1 &
  pid=$!
  printf '%s\n' "$pid" > "$root/worker.pid"
  wait "$pid"
done
