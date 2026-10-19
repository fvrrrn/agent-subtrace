# Frozen Qwen -> causal Transformer -> binary head

User-requested replacement of the extra Mamba block. Permanent server root:
`/var/local/danefedov/experiments/agent-safety/runs/action-models-transformer-v1`.
Immutable source/launch: sibling `action-models-transformer-v1-launch`.
Launcher PID2179373, GPU2. Existing teacher services and student runs unchanged.

Frozen cached Qwen token features -> LayerNorm1024 -> projection1024→256 ->
sinusoidal positional encoding -> one pre-norm causal Transformer block
(4 heads ×64, FF width256, GELU, residual connections, dropout0) ->
LayerNorm256 -> Linear256→1. Native PyTorch SDPA uses `is_causal=True`;
right padding follows real tokens and cannot influence their outputs.
No LoRA or Mamba; no additional dependencies. Attention has quadratic sequence
compute, unlike the additional Mamba; this is not a speed-equivalent replacement.

Reuse unchanged 15218train/848validation examples and frozen-Qwen BF16 cache.
Linear token target was selected by previous validation BCE, not StepGuard.
Four epochs; same seed, optimizer, batching and no ranking. Select checkpoint
by hard endpoint validation BCE, then evaluate on cached StepGuard3017test and
161dev. Compare AP, trapezoidal PR-AUC, ROC-AUC and threshold0.5 metrics against
Mamba-linear, direct linear head, original Kev and random Mamba. No threshold
tuning or external checkpoint selection. Report common keys and state hashes.

Before training: finite nonzero gradients, same-shape future-suffix invariance
and in-memory reload checks. Audit gradients are discarded; training reinitializes.
One seed, previously viewed benchmark, not exactly parameter-matched architectures.
Live/wait/copy handle `sh_14afb59180017VE28w6chsUfjO`; do not poll completion.
