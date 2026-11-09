#!/usr/bin/env bash
# Launch Qwen3.8-Flash-Next-FP8 with vLLM on 4x H100 (80GB).
#
# Fastest reliable configuration on this hardware:
#   - FP8 checkpoint (BF16 does not fit in 4x80GB)
#   - TP4 + expert parallel (TEP4)
#   - Triton MoE backend (Hopper)
#   - native MTP speculative decoding (3 draft tokens)
#   - FP8 KV cache + FP8 sparse-attention indexer
#   - prefix caching
#   - PLE / n-gram table in host RAM (required on 80GB cards)
#
# DFlash2 is faster in vendor benchmarks but is validated on GB300 TP1 and
# requires an SGLang target-capture patch; it is not used here.
#
# Usage:
#   ./launch_flash_next.sh
#   GPU_UTIL=0.80 MAX_SEQS=32 ./launch_flash_next.sh
set -euo pipefail

MODEL_ID="${MODEL_ID:-Qwen/Qwen3.8-Flash-Next-FP8}"
MODEL_DIR="${MODEL_DIR:-/var/local/danefedov/models/Qwen3.8-Flash-Next-FP8}"
HF_CACHE="${HF_CACHE:-/var/local/danefedov/models/hf}"
IMAGE="${IMAGE:-registry.k.avito.ru/dockerhub-proxy/vllm/vllm-openai:v0.32.0}"
CONTAINER="${CONTAINER:-qwen38-flash-next}"
SERVED_NAME="${SERVED_NAME:-qwen3.8-flash-next}"
PORT="${PORT:-18000}"
TP="${TP:-4}"
GPU_UTIL="${GPU_UTIL:-0.85}"
MAX_LEN="${MAX_LEN:-32768}"
MAX_SEQS="${MAX_SEQS:-64}"
MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-16384}"

mkdir -p "$MODEL_DIR" "$HF_CACHE"

if docker ps -a --format '{{.Names}}' | grep -qx "$CONTAINER"; then
  echo "container '$CONTAINER' already exists; remove it first:" >&2
  echo "  docker rm -f $CONTAINER" >&2
  exit 1
fi

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "image '$IMAGE' not found locally; pulling..." >&2
  docker pull "$IMAGE"
fi

# Warn if another process already occupies a lot of GPU memory.
echo "current GPU memory:" >&2
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader >&2 || true

docker run -d \
  --name "$CONTAINER" \
  --gpus all \
  --ipc=host \
  --ulimit memlock=-1 \
  --shm-size=16g \
  -p "${PORT}:8000" \
  -v "${MODEL_DIR}:/models/Qwen3.8-Flash-Next-FP8:ro" \
  -v "${HF_CACHE}:/root/.cache/huggingface" \
  -e HF_HOME=/root/.cache/huggingface \
  -e VLLM_PLE_CPU_OFFLOAD=1 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$IMAGE" \
  --model /models/Qwen3.8-Flash-Next-FP8 \
  --served-model-name "$SERVED_NAME" \
  --tensor-parallel-size "$TP" \
  --enable-expert-parallel \
  --moe-backend triton \
  --gpu-memory-utilization "$GPU_UTIL" \
  --max-model-len "$MAX_LEN" \
  --max-num-seqs "$MAX_SEQS" \
  --max-num-batched-tokens "$MAX_BATCHED_TOKENS" \
  --enable-prefix-caching \
  --kv-cache-dtype fp8 \
  --attention-config.indexer_kv_dtype fp8 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --engram-config '{"cpu_offload":true}' \
  --reasoning-parser qwen3 \
  --trust-remote-code

echo "started container '$CONTAINER' on port $PORT" >&2
echo "logs: docker logs -f $CONTAINER" >&2
echo "health: curl -s http://127.0.0.1:$PORT/health" >&2
