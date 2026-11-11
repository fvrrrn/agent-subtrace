#!/bin/bash
# Resumable annotation of the full v3 candidate pool with the selected judge
# prompt (prompt.txt, loaded by annotate.py by default).
#
# Normalizes the raw pool once, then keeps re-running annotate.py until no
# pending rows remain. Failed rows are retried on the next pass because
# annotate.py only treats status ok/too_long as finished.
set -u

DATA=/var/local/danefedov/data/agent-safety
SCRIPTS=/var/local/danefedov/scripts
IMG=registry.k.avito.ru/dockerhub-proxy/vllm/vllm-openai:v0.32.0
RAW=candidates_v3.actions.jsonl
NORM=${NORM:-candidates_v3.normalized.jsonl}
OUT=annotations_v3a_full.jsonl
LOG=$DATA/annotate_v3a_full.log
CONCURRENCY=${CONCURRENCY:-32}

if [ ! -f "$DATA/$NORM" ]; then
  echo "normalizing $(date -Is)"
  python3 "$SCRIPTS/prepare_candidates.py" --input "$DATA/$RAW" --output "$DATA/$NORM" || exit 1
fi

pending_of() {
  docker run --rm --network host \
    -v "$SCRIPTS":/scripts:ro -v "$DATA":/data \
    --entrypoint python3 "$IMG" /scripts/annotate.py \
    --input "/data/$NORM" --output "/data/$OUT" --dry-run 2>/dev/null \
    | python3 -c 'import sys,json; print(json.load(sys.stdin)["pending"])'
}

ok_of() {
  grep -c '"status": "ok"' "$DATA/$OUT" 2>/dev/null || echo 0
}

# A killed model server used to leave the client retrying for hours and writing
# only error rows. Stop instead of burning wall time silently.
for attempt in $(seq 1 500); do
  pending=$(pending_of)
  echo "attempt $attempt pending=$pending at $(date -Is)"
  if [ -z "$pending" ] || [ "$pending" -eq 0 ]; then
    break
  fi
  code=$(curl -s -o /dev/null -w "%{http_code}" -m 5 http://127.0.0.1:18000/health)
  if [ "$code" != "200" ]; then
    echo "SERVER_DOWN health=$code at $(date -Is); stopping"
    exit 1
  fi
  before=$(ok_of)
  docker run --rm --network host \
    -v "$SCRIPTS":/scripts:ro -v "$DATA":/data \
    --entrypoint python3 "$IMG" /scripts/annotate.py \
    --input "/data/$NORM" --output "/data/$OUT" \
    --concurrency "$CONCURRENCY" >> "$LOG" 2>&1
  rc=$?
  after=$(ok_of)
  echo "pass rc=$rc ok=$before->$after at $(date -Is)"
  if [ "$after" -le "$before" ]; then
    echo "NO_PROGRESS: no new ok rows at $(date -Is); stopping"
    exit 1
  fi
  sleep 30
done

echo "ANNOTATION_ALL_DONE at $(date -Is)"
