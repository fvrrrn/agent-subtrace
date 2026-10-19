# Small backbone: CPU-only preparation handoff

Candidate has priority over 9B if remaining budget is tight. Fixed overall deadline:
`1792348785.243998`; no extension implied. No GPU work, jobs, downloads, installs,
training, confirmation/external scoring, tests, lint or commits were performed.
Only the new small-backbone preparer and these notes were written.

## Actually inspected on llm01

- Cached `Qwen/Qwen3.5-0.8B-Base` revision
  `dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68` exists under
  `/var/local/danefedov/experiments/agent-safety/models/huggingface/hub/models--Qwen--Qwen3.5-0.8B-Base/snapshots/`.
  `models/huggingface` is a symlink; inspection followed it, without downloading.
- Actual config: `Qwen3_5ForConditionalGeneration`, text hidden size **1024**,
  24 layers, 8 attention heads, 2 KV heads, vocab 248320, BF16; three linear-attention
  layers followed by full attention, repeated. Text max positions 262144, but this
  experiment retains the **16384** whole-dialog admission limit.
- Tokenizer: `Qwen2Tokenizer`; native template SHA-256
  `273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80`,
  **different from the 4B template**. All tokenizer files exist locally.
- Complete sole safetensors file: `model.safetensors-00001-of-00001.safetensors`,
  1,746,942,600 bytes. CPU header inspection found 488 tensors; final data offset
  plus header equals actual file length. Index contains 488 tensor entries referencing
  that sole file; index total tensor size 1,746,882,752 bytes. This establishes local
  materialization, not successful model loading or a CUDA gate. Preparation hashes
  all model/tokenizer files; full checkpoint hash has not yet been computed here.
- Actual prospective manifest has five lists: train_small 2009, train_full 6058,
  dev_select 301, dev_confirm 299, original_validation 425. The preparer explicitly
  selects only the four requested lists, NOT train_full. Old prefix-wise 0.8B
  experiments/features are incompatible and are never consulted.

## New preparer

`graph-safety/scripts/prepare_autoresearch_small_backbone.py`:

- CPU `prepare` first reconstructs original 4B labelled action message stops by exact
  native IDs/spans, then independently renders the **actual 0.8B native template**.
  No injected instruction. Messages, tools, group, source and per-action state hashes
  are preserved. Parent hashes and new token-prefix hashes remain distinct.
- Every assistant causal endpoint, including unlabelled history, must exactly match
  the full-dialog token prefix. No fallback, truncation or template surgery. Failure
  rejects the whole dialog with its key and explicit reason. This strict all-assistant
  gate may exclude more than a labelled-endpoint-only gate; accepted counts are unknown.
- Writes dialogs, exclusions, four requested split lists, admitted `common_splits.json`,
  matched action keys and audit with source/manifest/template/tokenizer/model-file hashes.
  These matched keys must also restrict 4B comparisons if any exclusions occur; do not
  compare different populations or silently replace the fixed requested split counts.
- Separate `cache` requires explicit `--authorize-gpu-cache`, supports `--shard` and
  `--shard-count`, refuses existing shard outputs. Frozen native `Qwen3_5TextModel`,
  BF16/SDPA, whole-dialog forward, right padding to128, trimmed `(tokens,1024)` features.
  New identity includes actual frozen weights hash, model files, template, input/source,
  split, code and environment/kernel versions. No 4B/legacy cache reuse.
- Serializing confirmation inputs/features is not confirmation scoring. Do not read
  confirmation metrics or use labels for model/prompt/threshold selection.

## Exact commands for main agent (NOT executed)

Run from the deployed immutable source snapshot root, containing `graph-safety/`.
Use a new output directory; do not overwrite another run. CPU preparation:

```bash
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME=/var/local/danefedov/experiments/agent-safety/models/huggingface PYTHONPATH=graph-safety/src /var/local/danefedov/experiments/agent-safety/envs/kev/bin/python graph-safety/scripts/prepare_autoresearch_small_backbone.py prepare --data /var/local/danefedov/experiments/agent-safety/datasets/whole-dialog-4b-v2 --root /var/local/danefedov/experiments/agent-safety/runs/autoresearch-sixhour-small08-v1/data
```

Only after main authorizes GPU use and finds an available assigned GPU, shard0 example
below uses GPU2 (replace with the assigned GPU). For two shards run shard1 separately
on another assigned GPU with identical arguments except device and `--shard 1`.

```bash
CUDA_VISIBLE_DEVICES=2 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME=/var/local/danefedov/experiments/agent-safety/models/huggingface PYTHONPATH=graph-safety/src /var/local/danefedov/experiments/agent-safety/envs/kev/bin/python graph-safety/scripts/prepare_autoresearch_small_backbone.py cache --authorize-gpu-cache --data /var/local/danefedov/experiments/agent-safety/runs/autoresearch-sixhour-small08-v1/data --root /var/local/danefedov/experiments/agent-safety/runs/autoresearch-sixhour-small08-v1/cache --shard-count 2 --shard 0
```

These are preparation/cache CLIs, not a training launch or authorization. Main must
apply its deadline wrapper; this script does not extend/enforce the global budget.

## Trainer blockers (not changed)

1. `train_whole_dialogs.load_data` asserts the 4B base/revision. Its `Head` hardcodes
   `LayerNorm(2560)` and `Linear(2560,256)`; longest-input gates also hardcode2560.
   Minimal small-head change is **LN1024 → Linear1024→256**, keeping exactly two
   residual Mamba256 blocks, final normalization and the selected event readout.
2. `train_autoresearch_heads.features` hardcodes `(tokens,2560)` and padded width2560;
   its gate at line241 does too. Its head comes through `train_curve_event_heads.Head`,
   so use a fresh appropriately dimensioned head rather than 4B head initialization.
3. That trainer requires five split lists and `train_full` membership for *all*
   original validation/source checks. The new preparation intentionally contains only
   the four authorized lists. Main needs an explicit compatible small-only loader;
   do not silently call the 2009-list the original 6058-list or launch full-data work.
4. Native loaders/gates/evaluation lock checks must accept the **new** pinned model,
   template and dimension, while retaining strict cache identity and admitted-key checks.
   Preserve simultaneous token BCE **plus** event BCE; do not turn this into event-only
   training. Event-loss weight/curve/readout follow the predeclared main protocol.
5. Actual model load, CUDA causal/reload/gradient/longest gates, feature throughput,
   peak memory and final train/replay compatibility remain **unmeasured**. AST parse
   passed; no imports of torch/model, GPU gates or preparation run were performed here.

Actual cost and admitted counts are **unknown**, not estimated from old prefix runs.
BF16 feature payload is `2048 * admitted_token_count` bytes before serialization overhead.
Main should measure an authorized gate and retain the final-evaluation reserve before
deciding whether this fits the original fixed six-hour deadline. No 9B files were edited.
