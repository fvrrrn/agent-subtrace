# Transformer hard-token supervision

User requested a step-function target instead of a smooth curve. Same frozen
Qwen cache and causal Transformer/head (660993 parameters) as transformer-v1.
BCE target is the hard current-action label0/1 at every annotated action token;
no loss on history or padding, no ranking, no LoRA, no endpoint-only variant.
This is not a claim that the unannotated history has per-token safety labels.

Four epochs, same15218train/848validation, seed/init, fused AdamW and batching.
Choose checkpoint only by original validation endpoint BCE, then evaluate
StepGuard3017test and161dev. Compare AP/trapezoidal PR-AUC/ROC-AUC on matched
keys/labels/state hashes with Mamba token-hard, original Kev/random, and prior
Transformer-linear. Mamba token-hard was selected among8epochs; Transformer gets
4epochs. One seed, previously viewed benchmark, no threshold tuning.

CUDA preflight checks causal suffix invariance, reload, nonzero finite block/head
gradients, hard-token direct loss support and identical initialization to prior
Transformer-linear. Audit gradients discarded before actual training.

Root `/var/local/danefedov/experiments/agent-safety/runs/action-models-transformer-hard-v1`.
Immutable source: sibling `action-models-transformer-hard-v1-launch/source`.
Launcher PID2260345, GPU2; existing workers/services unchanged.
Live/wait/copy `sh_14b071cef001fB6RKpOAxaQ6LB`; do not poll completion.
Actual objective is explicit token_hard despite inherited curve descriptions
in the common launcher's provenance fields. No new architecture search.
