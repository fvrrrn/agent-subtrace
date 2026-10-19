# Whole-dialog token safety training

Current research context and permissions: [../AGENTS.md](../AGENTS.md).

## Pipeline

1. `scripts/assemble_data_bundle.py --output ...`: byte-prefix snapshots of
   completed judge calls; strong consensus labels plus published StepSafe action
   labels. Protected benchmarks remain outside grouped95/5 train/validation.
2. `scripts/prepare_whole_dialogs.py`: exact event-prefix trie, maximal real
   dialogs, native `tokenizer.apply_chat_template`, final-token action spans.
   Alternative branches remain separate. Shared prefix labels are supervised
   once. Actor roles/identities remain visible; labels and judge rationales do not.
3. `scripts/train_whole_dialogs.py`: frozen-feature cache, CUDA gate, training.
4. `scripts/launch_whole_dialogs.py`: two parallel GPU queues, validation comparison.

## Model and objectives

Pinned Qwen3.5-4B-Base `1001bb4d826a52d1f399e183466143f4da7b741b`.
Last hidden states → LayerNorm2560 → Linear2560→256 → two residual
pre-norm Mamba2 blocks → LayerNorm256 → Linear256→1.

**One dialog is one forward**, not one row per action prefix. All labeled
actions in the dialog receive token-level BCE in the same computation graph.
Qwen and Mamba are causal; later observations cannot affect earlier scores.
State resets between dialogs only. Full backpropagation through the sequence;
no truncated BPTT, per-action replay, chunk detach or benchmark-driven selection.

Linear artificial targets interpolate from the preceding observed action label
(initially0) to the current label. Unknown context has no BCE label. Token loss
is averaged within an action, then across actions in a dialog; accumulation4
dialogs. These are constructed targets, not measured token-level harm.

Four arms, **two epochs each**, same initial head and seed20261017:

| Arm | Qwen | Loss |
|---|---|---|
| frozen_linear | frozen | linear-token BCE |
| frozen_linear_mono | frozen | BCE + unsafe ranking |
| lora_linear | LoRA | linear-token BCE |
| lora_linear_mono | LoRA | BCE + unsafe ranking |

Ranking:0.1×mean relu(0.2+early−late), raw endpoint logits of adjacent labeled
unsafe actions in the **same** dialog forward. No pairing across a labeled safe
action, no separate model calls for pairs. This is a prior, not a calibrated
severity or future-harm probability.

LoRA rank16/alpha32/dropout0/all-linear; frozen base weights otherwise unchanged.
Fused AdamW: head lr3e-4, LoRA lr1e-5, weight_decay0.01, grad_norm1.
BF16 Qwen, FP32 head with BF16 autocast, SDPA, non-reentrant gradient
checkpointing for LoRA. Native Mamba2 CUDA/Triton kernels.

## Data and causality

Current immutable action snapshot:20233 actions, after acquisition deduplication.
Final prepared v2:7083 dialogs /19645 labeled actions;6658train and425validation
dialogs,18712/933 labeled actions. 2501 multi-action dialogs,259 with multiple
unsafe annotations. Published labels and machine silver retain their limitations.

588 actions excluded:16 auxiliary generated-benign-only,280 unstructured events,
234 whole-dialog length overflow,6 request/history mismatch,52 native-template
causal-prefix rewrites. Limit16384, maximum admitted15723, no truncation.
Native Qwen template can change earlier reasoning formatting when a later user
turn appears; such inputs are rejected rather than changing the official template
or silently supplying future-dependent formatting. Action endpoints are checked
against template-rendered causal prefixes, in characters and exact token IDs.

Some source `user_request` fields contain `Authored counterfactual scenario: ...`
metadata, not the actual request. Those headers are not rendered; the real user
message from history is retained. Source records and labels are not overwritten.
Private annotations never become visible messages. Prefix-sharing selection,
admission and metadata handling are recorded before training, not chosen by scores.

## Caches and audits

Frozen cache is one full-dialog feature tensor per input, shared across frozen
arms. Identity includes token IDs, model revision/frozen-weight hash, native
template hash and kernel environment. No old0.8B/prefix cache reused. LoRA
encodes each complete dialog online; it never reads stale frozen features.
This is a disk feature cache, not a claim of implemented incremental serving.

CUDA gate verified suffix causality, gradients of both Mamba blocks/head/LoRA,
loss support, frozen weights and reload, including backward on the longest
dialog. Validation selects minimum hard endpoint BCE from the two epochs;
reports AP, trapezoidal PR-AUC, ROC-AUC, Accuracy/F1 at0.5. No new benchmark
evaluation is automatically attached to this training schedule.

Completed run `runs/whole-dialog-4b-v3`, source sibling `-launch/source`.
All four conditions finished two epochs and selected epoch2 by validation BCE.
Former scheduler3530435 and watcher `sh_14bc17405001v1QuP1K8qolgsZ` are no
longer active. Results are copied to `artifacts/whole_dialog_4b_v3`; current
status and the separately locked epoch1 LoRA external evaluations are in
[../AGENTS.md](../AGENTS.md).
Data remains v2. Its completed7083-entry cache is reused read-only. Version2
training failed before its first optimizer step because torch.__version__ was
serialized as TorchVersion, rejected by weights_only loading. Version3 allows
only that known metadata type in the safe loader; future caches serialize a
plain string. Cached forward/backward passed on first and longest dialogs.
Version1 was stopped after a few cache entries, before any training, for an
independent request-field audit. Version2 preserves all matching token sequences;
the CUDA gate probe and longest dialog are byte-identical, so that gate is reused.
