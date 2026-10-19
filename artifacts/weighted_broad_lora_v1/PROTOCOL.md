# Locked continuation factorial

Authorization: «Бюджет забей. Также сделай ablation по поводу кривой и остального».
No previous deadline or proposed 90-minute cap applies. Finite eight cells,
two epochs each: frozen/LoRA × broad/step × uniform/adaptive token BCE.
Every cell also uses 0.1 endpoint event BCE, seed 20261017, accumulation four
whole dialogs, full BPTT, head AdamW lr 3e-4, adapter lr 1e-5, wd .01, clip 1.
New optimizer in every cell; common selected weighted-broad epoch-2 head SHA
7f3d0d3950313f2f7b39ab1b7a92b4933c3f072b885f0025ebfe42701a9b2137.
Frozen cells continue training the head for two epochs too.

Instruct Qwen/Qwen3.5-4B revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a,
last layer, original two residual Mamba blocks and endpoint head. LoRA all-linear
r16 alpha32 dropout0, non-reentrant gradient checkpointing. No added attention.
Native 2310 dialogs: 2009 train-small and 301 dev-select; no truncation,
StepGuard, confirm, external evaluation or threshold fitting.

Adaptive loss imports the exact prior loss_parts function: population std of
sigmoid logits over all known action tokens in the dialog, beta=clamp(std/.35,0,1),
detached softmax(beta*p) within action, weighted token BCE, mean actions.
T=1. Broad profile unchanged; step uses the action label at every action token.
Uniform control is mean token BCE. Both loss terms are always simultaneous.
Online features are recomputed for LoRA; frozen cached features never feed LoRA.
Backbone outputs on masked padding are zeroed before Mamba, matching the frozen
cache representation; real-token gradients remain intact. Frozen cells retain
zero-initialized, requires_grad=False LoRA modules for matched execution kernels;
their zero B matrices are audited after both epochs.

v3 numerical execution lock: five native Mamba forward Triton configurations
are pinned to the observed head-only replay manifest, included and hashed in
immutable source as `WEIGHTED_LORA_KERNELS.json`. Controlled A/B/A on identical
captured input/head reproduced and removed the failure by changing only these
native configurations. No arithmetic precision, outputs, tolerance or weights
are changed. Every cell must still reproduce ALL original selected dev token
logits exactly before training; the original reference is never regenerated.
This fixes process-dependent autotuner selection, not within-process stochastic
forward noise. Backward kernels retain native behavior; no global determinism
claim is made. All eight cells use the same forward lock.

Before updates: exact online replay of all 301 selected dev token arrays,
zero-adapter warm-start, separate token/event gradient checks for both Mamba
blocks/head/adapters, causal suffix/reload checks and longest-training backward.
Frozen backbone hashes before/after; preserve optimizer/RNG, token/event IDs and
raw logits, source/input/template/checkpoint hashes, environment and runtime.
Live tqdm streams through launcher to shell. Errors stop the queue, no retries.

Selection by dev AP, ties lower event BCE then earliest epoch. Report all eight
AP/trapezoidal PR-AUC/ROC-AUC, P/R/F1/Accuracy/FP/FN at .5 and BCE on identical
IDs; paired LoRA, weighting and curve contrasts. One seed, dev-selected.
Common initialization already underwent weighted-broad training and dev selection:
these are continuation adaptations, not unbiased fresh-init loss ablations.
