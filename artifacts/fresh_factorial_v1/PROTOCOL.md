# Fresh shared-init factorial v1 — prospectively bounded four conditions

Authorized 2026-10-19: continue research and experiments. Require all previous
8+3 outcomes complete, validate status, checkpoint hashes and all 886 action IDs.
No duplication or modification of those runs. Exactly four new conditions,
two epochs each: frozen/LoRA × step/broad, uniform token BCE and hard endpoint
event BCE simultaneously. Same Instruct revision, native template, last hidden
layer, two residual Mamba, optimizer, seed 20261017, dialog order and splits.
No attention, full fine-tuning, class weighting, new data or annotation.

Choose the event coefficient from **the three additional follow-up cells only**
by dev AP, ties lower unweighted event BCE, then total measured time. This is
the requested selection rule, not choosing from the control plus three. Their
winner is on_event1 (AP .52302177), hence coefficient 1.0. The existing control
has higher AP .53306592 at coefficient .1; disclose this instead of describing
1.0 as the globally best coefficient. Positive weight is nevertheless OFF in
all four fresh cells as required by this factorial's ordinary uniform BCE.
The earlier pos-weight gain in recall is not a ranking improvement.

Each cell begins with the exact same freshly seeded Mamba/head tensors and fresh
zero-output LoRA adapters. Frozen uses the same zero-adapter kernels with frozen
parameters. No selected head/checkpoint weights are loaded. Save initial tensors
and their content hashes and require exact hash equality across all four cells.
The prior trained-head replay has no valid reference here and is explicitly
inapplicable, not falsely reported as passed. Keep strict causal/reload zero-error,
both-loss gradients, loss masks, longest backward and frozen-weight gates.

Head initialization happens at the original trainer point before backbone loading;
the initial-state recording consumes no random draws. Native Mamba forward kernel
configurations remain pinned. Preserve all token logits, input IDs, model/template/
source/kernel hashes, checkpoints and optimizer/RNG state, two-epoch training costs.

At lock time save prior metric/hash evidence and all four names before reading
any new-cell metrics. GPU3 sequential with ownership check per cell, detached
regular-file stdout/stderr, independent live tqdm viewer, pidfd completion.
Any failure stops the finite queue without retry or scope expansion.

AP is primary; separate trapezoidal PR-AUC and ROC-AUC. Report P/R/F1/Accuracy,
FP/FN at .5, descriptive recall at FPR≤1%/5%, source slices and score quantiles.
No thresholds fitted; no confirm, oldval, external scores or new annotations.
One seed and dev-selected research, not held-out evidence. Compare the four fresh
cells directly. Comparison with prior continuation changes initialization and
total training exposure and cannot identify a pure loss or warm-start effect.

Before launch a bounded CPU diagnostic uses existing train/dev inputs and saved
dev logits: class mass under mean-action/dialog reduction, length/source slices,
false negatives, exact causal endpoint conflicts vs first-token/isolated-action
ambiguity, and analytic endpoint-logit gradient opposition. It does not identify
parameter-space gradient interference, evidence spans or causal harm attribution.
