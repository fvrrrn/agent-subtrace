# Full-FT: runtime gate and bounded experiment

**Main-session launch:** PID2351438, GPU0/1,2epochs small,
`runs/autoresearch-sixhour-v1/fullft_small`, immutable
`runs/autoresearch-sixhour-fullft-launch/source`. Training deadline
1792343385.243998 (90minutes before the overall deadline).
Host preflight:925GB available, GPU0/1 ~76/75GB free; other jobs untouched.
Watcher `sh_14f2bc473001wV8RlR6s6SLfC2`, no polling. Gates and real Adam-step
fit remain unverified until runtime completion. Main corrected FSDP buffer_dtype
toFP32, preserving rotary buffers and consistency with FP32 full-state checks;
BF16 parameters/compute and FP32 master/gradient/Adam state are unchanged.
Original implementation description below predates launch.

Independent hypothesis: adapting the entire native Qwen representation may transfer
better than frozen features or low-rank adaptation. This is not evidence that it
does; the parent scheduler may decline the experiment for time/memory reasons.

Only new files: `scripts/train_autoresearch_fullft.py` and this note. No GPU jobs,
tests, lint, dependency installation, commits, or existing-file edits performed.
Runtime, installed FSDP compatibility, memory fit, and numerical behavior are
**unverified**. An AST parse is only a syntax check, not a runtime gate.

## Exact proposed launcher

Run from `/var/local/danefedov/experiments/agent-safety` using the existing environment
and an immutable source snapshot containing the new script and its existing imports:

```bash
CUDA_VISIBLE_DEVICES=0,1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
PYTHONPATH=runs/autoresearch-sixhour-fullft-launch/source/src:archive/legacy-shm/danefedov-qwen-mamba-v1/deps \
envs/kev/bin/torchrun --standalone --nnodes=1 --nproc-per-node=2 \
  runs/autoresearch-sixhour-fullft-launch/source/train_autoresearch_fullft.py \
  --root runs/autoresearch-sixhour-fullft \
  --data datasets/whole-dialog-4b-v2 \
  --splits runs/autoresearch-sixhour-v1/data/splits.json \
  --epochs 2 --deadline-unix 1792348785.243998
```

The source snapshot must include `src/graph_safety` as well as imported sibling
scripts; verify that the stated existing Mamba overlay is still the chosen runtime
environment (do not install or upgrade anything). The launcher source snapshot
must first be prepared by the parent; this task does
not create it or launch anything. Do not reuse a preexisting run root or extend
the authorized deadline. The fixed deadline may already have expired when read.
The parent must reserve additional time for final evaluation, not merely use the
trainer's 180-second persistence reserve as an experiment scheduling allowance.

## Fixed implementation

- Exactly the top-level five-list split manifest; `train_small` 2009 dialogs,
  `dev_select` 301. Other splits are neither scored nor used in gates. Original
  audited rows/IDs preserved, right-padding to128 via the original helper, no
  truncation. Saved input manifests contain only train_small/dev_select.
- Original `backbone(False)` loads locally pinned native Qwen3.5 TextModel. Enable
  all gradients, promote original BF16 pretrained values to FP32, enable native
  non-reentrant activation checkpointing, then layer-wise FULL_SHARD FSDP.
  This promotion does not recover precision absent from the source weights.
- FP32 sharded parameters and AdamW moments; BF16 compute, FP32 reductions and
  retained gradients. No BF16-master AdamW, PEFT, compile, or new dependencies.
  Existing native loading itself temporarily holds an unsharded GPU model.
- Same float32 Mamba head/seed20261017, replicated DDP after gates; endpoint,
  broad token curve, simultaneous token BCE +0.1 hard endpoint BCE. Model LR2e-6,
  head LR3e-4, AdamW weight decay0.01, combined backbone-global/head-once norm clip1.
- Global batch4: two ranks ×two synchronized microsteps; losses scaled2/window.
  Every real dialog appears once, including the final one-dialog window. Empty
  slots run an explicit zero-loss synchronization forward, not duplicated label
  supervision. No DistributedSampler padding or discarded examples. No no_sync
  accumulation of full unsharded gradients.
- Evaluation uses identical dev dialogs on both ranks to align all FSDP
  collectives. Only rank0 counts/writes them; this deliberately costs redundant
  head/evaluation computation for a smaller and auditable implementation.
  Full dev token logits, predictions, event BCE, AP, trapezoidal PR-AUC, ROC-AUC,
  binary metrics and low-FPR recall come from the existing metrics helper.
  Select by dev_select AP, ties lower event BCE, exact ties earlier epoch.

## Mandatory pre-training gates / saved states

All ranks execute causal future-ID perturbation, real disk full-backbone/head
gather/reload (parameters deliberately perturbed before loading), separate token
and event readout-gradient checks, loss-mask check, finite/nonzero global backbone
and both Mamba-block gradients, and longest TRAIN dialog backward <=16384 tokens.
No optimizer steps; probe gradients discarded and RNG restored before training.
No held-out examples or labels enter gates. Failure aborts without retries or
tolerance relaxation. Exact equality is required for causal and reload outputs.

Epoch checkpoints: rank0 CPU gathered full FP32 backbone and CPU head; per-rank
local sharded AdamW and RNG states, schedule progress, hashes. Full-state gather
is collective with rank0_only=True; reload is collective with rank0_only=False
and full CPU state loaded on **every** rank. Local optimizer states require the
same world size and wrapping policy; no portable optimizer resharding or resume
CLI is implemented. A saved complete checkpoint contains the necessary state,
but loading training state has not been runtime validated.

Protocol records source/dependency/model revision/template/data/input hashes,
actual lengths, package/kernel versions and peak per-rank CUDA allocation.
All exceptions retain rank-specific tracebacks; process group destroyed finally.
Global rank0-broadcast time guards precede setup, gate stages, training microsteps,
optimizer steps, epoch persistence, and each dev evaluation forward. They cannot
interrupt a running CUDA kernel/checkpoint or a stalled peer. Timeout/failure is
not automatically retried; torchrun must terminate surviving peers on rank error.

Budget stop persists diagnostic backbone/head/optimizer/RNG state, discards any
pending accumulation gradients, marks **nonresumable** (also when no gradients
are pending), preserves partial dev outputs, and never selects an incomplete dev
epoch. The 180-second reserve is a guard, not a guarantee that CPU gather/hash/
storage can finish in180 seconds; parent must measure persistence throughput.

## Memory / runtime blockers

For a nominal4B-parameter backbone (actual TextModel count may differ), FP32
masters are ~16GB globally, ~8GB/rank; two FP32 Adam moments ~16GB/rank; FP32
sharded gradients ~8GB/rank. Thus ~32GB/rank backbone persistent training state,
before BF16 active/all-gather buffers, checkpointed activations, Mamba state,
SDPA/recurrent workspace, allocator fragmentation, and replicated head. Native
FP32 pre-wrap allocation is ~16GB/rank, with a transient BF16 source allocation.
No claim that the longest dialog or this configuration fits80GB is made.
The no-step gates run before Adam moment allocation: passing the longest backward
is necessary but not sufficient to demonstrate training-step fit. The first real
optimizer step adds approximately16GB/rank of FP32 moments; require that headroom
and measure the real step before asserting feasibility.

CPU full-state output costs ~16GB/rank0; reload gate reads the full state on both
ranks (~16GB host RAM each, potentially shared host total~32GB), plus serialization
and full-state transformation temporaries. Gate checkpoint +each epoch need full
backbone storage, plus optimizer shard files (~32GB total moments per epoch).

Potential blockers are installed Qwen/FSDP support for its native hybrid linear
attention and checkpoint recomputation; actual layer wrapping/type layout;
native kernel gradient/dtype constraints; CPU-RAM/disk throughput; longest-dialog
activation memory; and completing gates +2009-dialog epochs +301-dialog dev
evaluations within the remaining wall budget. No installed API introspection or
GPU execution was performed. Only passing the actual mandatory gates can justify
proceeding, and parent scheduling remains required.
