#!/usr/bin/env bash
set -euo pipefail
root=${1:?Experiment root required}
stage=${2:?pilot or full required}
export CUDA_VISIBLE_DEVICES=0
export HF_HOME=/dev/shm/danefedov-kev-rjudge-20261013/huggingface
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export PYTHONPATH=/dev/shm/danefedov-kev4b-fast-serving-20261014/deps:/home/danefedov/kev-rjudge-baseline-20261013/source:$root/src
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
export TRITON_F32_DEFAULT=ieee TRITON_CACHE_DIR=$root/triton
python=/dev/shm/danefedov-kev-rjudge-20261013/venv/bin/python
exec "$python" -u -m graph_safety.controlled_training "$stage" --root "$root/results"
