#!/usr/bin/env bash
set -euo pipefail
root=${1:?Specify remote experiment directory}
python=${PYTHON_BIN:?Specify pinned Python executable}
for suite in oracle reverse; do
  for split in dev locked; do
    pids=()
    for shard in 0 1; do
      CUDA_VISIBLE_DEVICES=$((shard+2)) "$python" -u -m graph_safety.model_replay run \
        --root "$root/results/$suite" --suite "$suite" --split "$split" --shard "$shard" > "$root/${suite}_${split}_${shard}.log" 2>&1 &
      pids+=("$!")
      printf '%s\n' "$!" > "$root/${suite}_${split}_${shard}.pid"
    done
    failed=0
    for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
    if test "$failed" != 0; then exit 1; fi
    if test "$split" = dev; then
      "$python" -u -m graph_safety.model_replay select --root "$root/results/$suite" --suite "$suite"
    else
      "$python" -u -m graph_safety.model_replay aggregate --root "$root/results/$suite" --suite "$suite"
    fi
  done
done
