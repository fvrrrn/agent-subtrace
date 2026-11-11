# Qwen3.8-Flash-Next FP8 — annotation launcher

Launch script for action-safety annotation on `dsswat27` (4× H100 80GB PCIe).

## Why this configuration

| Choice | Reason |
|---|---|
| `Qwen/Qwen3.8-Flash-Next-FP8` | 125B MoE / 6B active, 262K context; FP8 is the only variant that fits 4×80GB with KV headroom. BF16 is ~335 GiB and does not fit. |
| vLLM `v0.32.0` | Supports the architecture, MTP head and PLE offload; stable Docker image. |
| TP4 + expert parallel | Fits the FP8 checkpoint across 4 GPUs; TEP is faster than plain TP on MoE. |
| Triton MoE backend | Hopper path recommended by the vLLM recipe. |
| FP8 KV cache + FP8 indexer | Roughly doubles KV capacity and reduces prefill/decode memory. |
| Native MTP, 3 draft tokens | Built into the checkpoint; measured ~3.3 accepted tokens on similar workloads. |
| Prefix caching | Agent traces share long prefixes; avoids recomputing them. |
| PLE / n-gram CPU offload | Required on 80GB cards: the 51B n-gram table lives in host RAM. Needs ≥64GB free RAM and `--ulimit memlock=-1`. |
| `GPU_UTIL=0.85`, `MAX_SEQS=64` | Conservative defaults because other processes already use 5–8GB per GPU. Raise after measuring. |

## DFlash2

`incoai/Qwen3.8-Flash-Next-DFlash2` is a block-diffusion drafter that is faster
than native MTP in vendor benchmarks, but it is validated on **GB300 TP1** and
needs an SGLang target-feature-capture patch. It is not used in this launcher.
If needed, it should be a separate experiment, not the default annotation path.

## Files

- `download_flash_next.sh` — download the FP8 checkpoint (~173 GiB).
- `launch_flash_next.sh` — start the vLLM container.

## Usage on dsswat27

```bash
mkdir -p /var/local/danefedov/scripts
cp download_flash_next.sh launch_flash_next.sh /var/local/danefedov/scripts/
chmod +x /var/local/danefedov/scripts/*.sh

# 1. Download (one time, ~173 GiB)
/var/local/danefedov/scripts/download_flash_next.sh

# 2. Launch
/var/local/danefedov/scripts/launch_flash_next.sh

# 3. Check
docker logs -f qwen38-flash-next
curl -s http://127.0.0.1:18000/health
curl -s http://127.0.0.1:18000/v1/models
```

## Tuning

If startup OOMs:

```bash
GPU_UTIL=0.80 MAX_SEQS=32 MAX_LEN=16384 ./launch_flash_next.sh
```

If other processes release GPU memory, try `GPU_UTIL=0.90`, `MAX_SEQS=128`.

Stop:

```bash
docker rm -f qwen38-flash-next
```

## Notes

- Model license: `qwen-community-1.0` (not Apache-2.0). Verify terms before
  redistribution.
- Do not kill other users' GPU processes on `dsswat27`.
- The model has thinking mode on by default; for annotation requests set
  `chat_template_kwargs: {"enable_thinking": false}` and use guided JSON.

## Annotation client

`annotate.py` sends only the visible causal input (`tool_interfaces`,
`user_request`, `observed_history`, `proposed_action`) and writes structured
JSON per candidate. It uses guided JSON and disables thinking.

```bash
# On dsswat27, after the server is healthy:
docker run --rm --network host \
  -v /var/local/danefedov/scripts:/scripts:ro \
  -v /var/local/danefedov/data/agent-safety:/data \
  --entrypoint python3 \
  registry.k.avito.ru/dockerhub-proxy/vllm/vllm-openai:v0.32.0 \
  /scripts/annotate.py \
    --input /data/candidates_2000.normalized.jsonl \
    --output /data/annotations_2000.jsonl \
    --base-url http://127.0.0.1:18000 \
    --model qwen3.8-flash-next \
    --limit 300 --concurrency 32
```

`prepare_candidates.py` normalizes raw candidates (adds `sources`, parses
`state`) so `sample_candidates.py` and `annotate.py` can consume them.

`annotate.py` reads its judge prompt from `--system-file` (default
`prompt.txt` next to it), so copy `prompt.txt` together with the script.

## Judge prompt

`prompt.txt` — единственный промпт в репозитории: тот, которым размечен
датасет (sha256 `5417c7fd1204fc6c08f18515570a37c6d5691b8e380d6ed29893596c14daafd6`).
Отклонённые кандидаты прошлого раунда в репозитории не хранятся; их измерения,
гипотезы и правило отбора — в `docs/DATA_PROMPT_TUNING.md`.

```bash
# Score the prompt on the dev split
docker run --rm --network host \
  -v /var/local/danefedov/scripts:/scripts:ro \
  -v /var/local/danefedov/data/agent-safety:/data \
  --entrypoint python3 \
  registry.k.avito.ru/dockerhub-proxy/vllm/vllm-openai:v0.32.0 \
  /scripts/eval_prompt.py \
    --gold /data/gold_cases.jsonl --split dev \
    --system-file /scripts/prompt.txt \
    --output /data/eval_dev.jsonl \
    --concurrency 16 --temperature 0.2 --seed 101

# Aggregate repeated runs and apply the selection rule
python aggregate_runs.py <dir> r1 r2 r3

# Inspect which cases changed between two runs
python diff_runs.py eval_r1_dev.jsonl eval_r2_dev.jsonl --only-changes
```

Full protocol, measurements and decision: `docs/DATA_PROMPT_TUNING.md`.

