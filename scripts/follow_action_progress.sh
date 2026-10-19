#!/usr/bin/env bash
set -euo pipefail
root=${1:-/var/local/danefedov/experiments/agent-safety/runs/action-models-v1}
ssh llm01 bash -s -- "$root" <<'REMOTE'
set -euo pipefail
root=$1
pid=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["pid"])' "$root/scheduler_start.json")
# The later queue logs do not exist yet. --retry attaches when they are created.
# Only tail's missing-file notices are suppressed; training tracebacks stay visible.
tail --pid="$pid" --retry -n 1 -F \
  "$root/scheduler.log" \
  "$root/logs/gpu2_0_cache.log" \
  "$root/logs/gpu2_1_baselines.log" \
  "$root/logs/gpu2_2_train.log" \
  "$root/logs/gpu2_3_train.log" \
  "$root/logs/gpu2_4_train.log" \
  "$root/logs/gpu3_0_train.log" \
  "$root/logs/gpu3_1_train.log" \
  "$root/logs/gpu3_2_train.log" \
  "$root/logs/gpu3_3_train.log" 2>/dev/null
REMOTE
