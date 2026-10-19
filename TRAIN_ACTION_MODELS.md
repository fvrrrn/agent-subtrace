# Legacy action-prefix training

This document describes the historical independent-prefix experiments, **not**
the user's current whole-dialog training method. Current entry points and
protocol: [WHOLE_DIALOG_TRAINING.md](WHOLE_DIALOG_TRAINING.md).
Research context and launch permissions: [../AGENTS.md](../AGENTS.md).

Entry point: `scripts/train_action_models.py`. Use the existing Kev/Mamba CUDA
environment with `graph-safety/src` on `PYTHONPATH`; dependencies are not installed
or changed by this script. No scheduler or import starts a run.

## Conditions

Base: pinned `Qwen/Qwen3.5-0.8B-Base`, not MiniCPM or Qwen3.8.

| Architecture | Trainable components | Objectives |
|---|---|---|
| `mamba` | one Mamba-2 + projection + binary head | exponential; cosine_rank |
| `mamba_lora` | same + base LoRA | exponential; cosine_rank |
| `mamba2_lora` | two residual Mamba-2 layers + projection/head + base LoRA | exponential; cosine_rank |
| `kev` | original Kev LoRA + native pointer head | noul_rank |

This is **seven trained conditions**, after the user's clarification about two
curves, and two baselines: original Kev and frozen Qwen + random one-layer
Mamba/head. The latter is an untrained classifier, not independent random scores
and not a randomly initialized Qwen backbone. It shares the one-layer training
heads' initialization seed.

### Loss and targets

For an annotated action with label y, prior target a and action token fraction u:

* Exponential: `y + (a-y)*exp(-ln(5)*u)`. No ranking term. The previous verified
  annotated action's exponential endpoint sets a; initial anchor is zero.
* Full cosine, no memory coefficient:
  `a + (y-a)*(1-cos(pi*u))/2`. Here a is the previous verified annotated hard label,
  initially zero. Add `0.1 * mean(relu(0.2 + z_early - z_late))` on verified unsafe
  action pairs. z is raw endpoint log-odds, not a sigmoid probability.
* Kev: native hard-label NOUL CE plus the same unsafe-pair ranking. Binary
  BCE on `YES_logit - NO_logit` is exactly two-option CE. Native option mapping
  is checked. The shipped inference temperature is retained; training and
  checkpoint-selection loss use raw logits. Kev has **no token-curve loss**.

Mamba curve BCE is averaged over tokens **within each labeled action**, then
over actions in the optimizer window. Context tokens are not labeled safe and
do not receive direct BCE. Gradients still pass through them. Ranking pairs are
sampled uniformly in an independent minibatch, avoiding rare-pair importance
spikes. Both endpoints receive gradients. No per-token monotonic penalty is added.

Pair preparation requires exact matching tools/request, the same split and
scenario group, exact earlier-history prefix and exact proposed action in the
next observed event (allowing only its later tool_result). Multiple matches are
not resolved arbitrarily. Opaque string histories are preserved as context but
receive no inferred pairs. Preparation reports coverage. Ranking training fails
if there are no admitted unsafe pairs. Temporal continuation is not severity gold.

Each example replays its own full causal prefix through Qwen/Mamba; state is not
carried between independent minibatch rows. This is **not** incremental streaming
or training on fully labeled trajectories. Earlier target anchors use only
verified earlier annotations; missing historical labels remain unknown. Curves
are artificial targets using full action length/label, not token-level safety gold.

## Speed choices

* **Frozen-base arm:** batched Qwen feature extraction once, then a shared BF16
  disk cache for both objectives and the random baseline. Features are loaded per
  minibatch, not all into RAM. Cache space estimate is written before extraction.
* **LoRA arms:** BF16 Qwen online, FP32 trainable Mamba/head under BF16 autocast,
  zero LoRA dropout, fused AdamW and official fused CUDA/Triton Mamba-2 scan.
  Token features cannot be cached while the base is changing.
* Length buckets, right padding to 128 and padded-token minibatch budget reduce
  wasted computation. No packing of unrelated recurrent sequences. An oversized
  single admitted example is not truncated to fit the minibatch budget.
* `--attention flash_attention_2` selects FlashAttention when installed/supported;
  default SDPA avoids adding dependencies. `--checkpointing` is opt-in: it saves
  memory at the cost of recomputation, and is not automatically faster on H100.
* Native Kev retains the established fp32/unmerged training loader with BF16
  autocast, plus fused AdamW. Its API remains single-example; no unsupported
  batched pointer interface is invented.

[Unsloth supports Qwen3.5 fine-tuning](https://www.unsloth.ai/docs/models/qwen3.5/fine-tune),
but this does **not** establish support or speedups for our custom token Mamba
readout or native Kev pointer loss. This code uses Transformers/PEFT + official
Mamba kernels, not an unverified Unsloth wrapper. No claim of maximum throughput
or measured speedup is made without a separate authorized CUDA measurement.
QLoRA is not the default for a 0.8B backbone on an 80GB H100: lower weight memory
is not automatically lower runtime.

## Explicit phases — examples only, not executed

From `graph-safety`, with the existing compatible environment activated:

```bash
export PYTHONPATH="$PWD/src"
export ROOT=/var/local/danefedov/experiments/agent-safety/runs/action-models-v1
export DATA=/var/local/danefedov/experiments/agent-safety/datasets/data-bundle-v1/training/records.json

# CPU preparation/tokenization, no training. Pinned tokenizer must be cached.
python scripts/train_action_models.py prepare --root "$ROOT" --records "$DATA"

# The following GPU phases require explicit launch authorization.
CUDA_VISIBLE_DEVICES=2 python scripts/train_action_models.py cache --root "$ROOT"
CUDA_VISIBLE_DEVICES=2 python scripts/train_action_models.py baselines --root "$ROOT"

# Repeat explicitly for each of mamba / mamba_lora / mamba2_lora,
# and each of exponential / cosine_rank. No automatic seven-run scheduler.
CUDA_VISIBLE_DEVICES=2 python scripts/train_action_models.py train \
  --root "$ROOT" --arch mamba_lora --objective cosine_rank

CUDA_VISIBLE_DEVICES=3 python scripts/train_action_models.py train \
  --root "$ROOT" --arch kev --objective noul_rank --epochs 2

python scripts/train_action_models.py compare --root "$ROOT"
```

Do not put these jobs on judges' GPU0/1 or stop foreign jobs. The examples above
do not check live GPU availability; verify it before an authorized launch.

## Outputs and limits

No truncation. Per-record length refusals, exact IDs/span boundaries, input
hashes, verified pairs, gradients, full frozen-parameter hashes, trainable counts,
configuration, package versions, epoch times, endpoint logits and probabilities
are retained. Failure on OOM/nonfinite values is not silently dropped/retried.
Existing run directories are not overwritten.

Checkpoints minimize **hard-label endpoint validation BCE**, not curve-fit loss
or AP. `compare` recalculates AP, ROC-AUC and threshold-0.5 metrics on common
successful validation keys, with label/input-hash checks. These are
**checkpoint-selection validation metrics, not an independent test**. Benchmark
evaluation is deliberately not scheduled by this training script. Native Kev
and Mamba have different renderers, so this is not a pure architectural ablation.

Default Mamba epochs=8, head lr=3e-4, LoRA lr=1e-5, rank=16/alpha=32,
weight_decay=.01, accumulation=4, batch_size<=4, token_budget=16384,
train_limit=8192 and eval_limit=65536. These are initial explicit settings, not
tuned or speed-benchmarked values. The original run was subsequently authorized
and started after CUDA auditing; see `artifacts/action_models_v1/README.md`.
Unsloth is not used or compatibility-tested in this path.

## Authorized additional hard-target controls

After the original run started, the user requested `token_hard` and
`endpoint_hard` on GPU2. Both use frozen **one-layer** Mamba/head, without LoRA
or ranking. The first applies hard 0/1 BCE to all annotated action tokens; the
second applies it only to the action endpoint. Unannotated/context tokens are
not assigned safe labels; endpoint gradients still traverse the full prefix.

`scripts/launch_hard_target_models.py` creates a **separate** experiment sharing
the prepared parent records/cache and original baselines via read-only usage of
symlinks. A dedicated CUDA loss-support/gradient audit precedes two concurrent
8-epoch jobs. Same initialization and settings as parent frozen-Mamba arms.
The original running source snapshot and protocol are not modified. See
`artifacts/action_models_hard_targets_v1/README.md` for the launched run.
