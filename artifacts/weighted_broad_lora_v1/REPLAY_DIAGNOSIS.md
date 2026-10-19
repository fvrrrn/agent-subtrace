# Native Triton autotuning caused the replay divergence

## Evidence

The full original workflow was replayed (CPU admission, head construction and
warm-start loading, online Qwen + PEFT insertion, frozen-weight hashing, eval
mode, validation order). No optimizer was constructed by diagnostic runs.

1. Head state SHA before backbone setup, after setup and after repeated forwards:
   `811f5a25c7a9aa7bb5c70701427bab7ffa29fc8a168732af6bb23b597573a4eb` throughout.
2. On the failing 3200-token dialog, online hidden features equal frozen cache
   bitwise; zero-LoRA and disabled-LoRA outputs also equal bitwise. Padded inputs,
   dtype BF16, strides and storage offsets match. All pointers are 256-byte aligned.
3. Ten consecutive identical-input forwards in the full process consistently
   reproduce one mismatched token, max error .00390625. Its five action endpoints
   match the original reference exactly. This is **not within-process random noise**.
4. A separate head-only process using the exact captured input and head matches
   the original reference in ten of ten forwards. Module captures first diverge
   at `blocks.0`; input normalization, projection and first pre-Mamba norm agree.
5. The processes selected different native Triton autotuner configurations for
   BMM/chunk cumulative sum/chunk state. They share scan and state-passing choices.
6. Controlled **A/B/A in one process**, identical fixed weights/input:
   - original full-process config A exactly reproduces the failed output;
   - recorded head-only config B exactly reproduces the original saved output;
   - restoring A restores the identical failed output.
   Head state hash remains unchanged, and endpoints are exact in all three.

This demonstrates a causal kernel-selection explanation for the captured failure.
The first run's .015625/4-token failure was not captured with kernel metadata;
the controlled experiment does not reconstruct its exact historical kernel choice.

## Execution change

v3 pins the five **observed native forward** choices from B using an immutable,
hashed manifest. No precision change, tensor rounding, reference regeneration,
parameter update or tolerance relaxation. Backward autotuners are not pinned;
we do not claim global deterministic training.

The runtime still requires exact replay of **all original token logits on all
301 dev dialogs**, then separate token/event gradients, causal/reload and longest
backward gates. Only after those pass may the eight locked continuation cells train.
The same forward lock applies to every cell. Prior failed runs remain unchanged.

## Artifacts

- `diagnostic_v3/diagnostic.json`: same-workflow input/state/repeated-forward checks.
- `diagnostic_headonly_v1/diagnostic.json`: exact captured head-only replay and module deltas.
- `diagnostic_kernel_choice_v1/diagnostic.json`: controlled A/B/A causal intervention.
- `diagnostic_v2/diagnostic.json`: another full-process replay matched all 301
  dialogs without a pin; its kernel choices were not logged due to an instrumentation
  omission. This observation motivated recording and fixing autotuner choices.
- Server `runs/weighted-broad-replay-diagnostic-v3/failure_capture.pt`, `modules.pt`:
  exact online/cache hidden tensors, head parameters, inputs and raw logits.

Diagnostics intentionally end with `DIAGNOSTIC_COMPLETE_NO_TRAINING`, recorded by
the reused trainer as a failed status. That intentional stop is not a training run.
