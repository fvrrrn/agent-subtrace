#!/usr/bin/env bash
set -euo pipefail
root=${1:?Specify remote experiment directory}
python=${PYTHON_BIN:?Specify pinned Python executable}
pids=()
for shard in 0 1; do
  CUDA_VISIBLE_DEVICES=$((shard+2)) "$python" -u -m graph_safety.context_attention run \
    --root "$root/results" --shard "$shard" > "$root/shard_${shard}.log" 2>&1 &
  pids+=("$!")
  printf '%s\n' "$!" > "$root/shard_${shard}.pid"
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
if test "$failed" != 0; then exit 1; fi
"$python" -u -m graph_safety.context_attention aggregate --root "$root/results"
