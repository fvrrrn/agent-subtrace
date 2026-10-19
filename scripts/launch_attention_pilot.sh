#!/usr/bin/env bash
set -euo pipefail
root=${1:?Specify remote experiment directory}
source_root=${2:?Specify upstream experiment directory}
python=${PYTHON_BIN:?Specify pinned Python executable}
# Wait for the actual upstream process; no interval polling or GPU overlap.
"$python" - "$source_root" <<'PY'
import os
import select
import sys
from pathlib import Path
root = Path(sys.argv[1])
pid = int((root / "run.pid").read_text())
try:
    fd = os.pidfd_open(pid)
except ProcessLookupError:
    pass
else:
    try:
        command = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
        if command and str(root / "launch_inferred_atbench.sh") not in command:
            raise RuntimeError("Upstream PID identity mismatch")
        select.select([fd], [], [])
    finally:
        os.close(fd)
if not (root / "results/results.json").exists():
    raise RuntimeError("Upstream failed: attention experiment will not start")
PY
"$python" -u -m graph_safety.attention_pilot prepare --root "$root/results" --source "$source_root/results"
pids=()
for shard in 0 1; do
  CUDA_VISIBLE_DEVICES=$((shard+2)) "$python" -u -m graph_safety.attention_pilot run \
    --root "$root/results" --shard "$shard" > "$root/shard_${shard}.log" 2>&1 &
  pids+=("$!")
  printf '%s\n' "$!" > "$root/shard_${shard}.pid"
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
if test "$failed" != 0; then exit 1; fi
"$python" -u -m graph_safety.attention_pilot aggregate --root "$root/results"
