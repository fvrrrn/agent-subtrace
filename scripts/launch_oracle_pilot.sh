#!/usr/bin/env bash
set -euo pipefail
root=${1:?Specify remote experiment directory}
python=${PYTHON_BIN:?Specify pinned Python executable}
for split in dev locked; do
  pids=()
  for shard in 0 1; do
    CUDA_VISIBLE_DEVICES=$((shard+2)) "$python" -u -m graph_safety.oracle_pilot run \
      --root "$root/results" --split "$split" --shard "$shard" > "$root/${split}_${shard}.log" 2>&1 &
    pids+=("$!")
    printf '%s\n' "$!" > "$root/${split}_${shard}.pid"
  done
  failed=0
  for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
  if test "$failed" != 0; then exit 1; fi
  if test "$split" = dev; then
    "$python" -u -m graph_safety.oracle_pilot select --root "$root/results"
  else
    "$python" -u -m graph_safety.oracle_pilot aggregate --root "$root/results"
  fi
done
