#!/usr/bin/env bash
# Download the Qwen3.8-Flash-Next-FP8 checkpoint (~173 GiB) with the vLLM image.
set -euo pipefail

MODEL_ID="${MODEL_ID:-Qwen/Qwen3.8-Flash-Next-FP8}"
MODEL_DIR="${MODEL_DIR:-/var/local/danefedov/models/Qwen3.8-Flash-Next-FP8}"
HF_CACHE="${HF_CACHE:-/var/local/danefedov/models/hf}"
IMAGE="${IMAGE:-registry.k.avito.ru/dockerhub-proxy/vllm/vllm-openai:v0.32.0}"

mkdir -p "$MODEL_DIR" "$HF_CACHE"

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "image '$IMAGE' not found locally; pulling..." >&2
  docker pull "$IMAGE"
fi

docker run --rm \
  --entrypoint bash \
  -v "${MODEL_DIR}:/models/Qwen3.8-Flash-Next-FP8" \
  -v "${HF_CACHE}:/root/.cache/huggingface" \
  -e HF_HOME=/root/.cache/huggingface \
  "$IMAGE" -lc "
    set -euo pipefail
    if command -v hf >/dev/null 2>&1; then
      hf download '$MODEL_ID' --local-dir /models/Qwen3.8-Flash-Next-FP8 --max-workers 8
    elif command -v huggingface-cli >/dev/null 2>&1; then
      huggingface-cli download '$MODEL_ID' --local-dir /models/Qwen3.8-Flash-Next-FP8
    else
      python3 -m huggingface_hub.commands.huggingface_cli download '$MODEL_ID' --local-dir /models/Qwen3.8-Flash-Next-FP8
    fi
  "
