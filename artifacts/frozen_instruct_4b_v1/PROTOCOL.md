# Frozen Instruct 4B v1

Explicit new authorization: Qwen3.5-4B post-trained counterpart of Base, frozen
backbone plus the existing two residual Mamba blocks and classifier. Five arms,
two epochs each: A_broad/A_linear/A_step use the existing endpoint head and
token-curve BCE + 0.1 event BCE; B uses hard-Y forecasting plus auto-pool;
C uses hard-Y forecasting plus multiscale subsequence features. No added
attention, LoRA, full fine-tuning, StepGuard data or generated rationale.

Official model: `Qwen/Qwen3.5-4B`, revision
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`. The official model card explicitly
identifies post-trained weights and names `Qwen/Qwen3.5-4B-Base` as its base.
Config: `qwen3_5`, text hidden size 2560, matching the existing head input size.

Only original train_small (2009 dialogues) and dev_select (301) are admitted.
Every labeled action must preserve exact native causal-prefix token equality
against its whole-dialogue rendering. Any rejection blocks GPU cache/training;
no prefix fallback, template replacement, truncation or silent exclusion.
The preparation gate precedes weight download to avoid wasting the budget if
native template behavior makes this protocol impossible.

Training stop: Unix 1792345185. Original global deadline: 1792348785;
one hour reserved. Selection: maximum dev-select AP, ties lower event BCE;
threshold 0.5 only. Preserve AP, trapezoidal PR-AUC, ROC-AUC, P/R/F1/Accuracy,
FP/FN, BCE, inputs/IDs/lengths, exceptions, kernel environment and hashes.

Server run: `runs/frozen-instruct-4b-v1`; immutable sibling source:
`runs/frozen-instruct-4b-v1-launch/source`. New cache required. GPU ownership
check found other users' processes on 0/1/2 and GPU 3 free; do not touch them.
CUDA causal/reload/gradient/mask/longest-backward/frozen-cache gates are mandatory
before training, but have not yet run. This document is authorization/protocol,
not a claim that GPU encoding or any of the five trainings has completed.
