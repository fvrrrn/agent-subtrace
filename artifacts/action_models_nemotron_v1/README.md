# Nemotron-AIQ: evaluation-only final-report screen

Separate verified raw copy:
`/var/local/danefedov/experiments/agent-safety/datasets/nemotron-aiq-eval-v1/raw`.
Repository `nvidia/Nemotron-AIQ-Agentic-Safety-Dataset-1.0`, pinned revision
`668cdd63f594eaf5e24ca2e41aeba6d4c92376b4`. Existing715MB snapshot copied with
every manifest checksum verified; no redundant download or redistribution.
NVIDIA Evaluation Dataset License: no training or checkpoint selection.

Raw10796 traces: safety2800 with-defense +2800 without-defense, security2600
with-defense +2596 without-defense. Security `success` is true on all rows and
cannot be taken as binary unsafe-action gold; preserve inventory, no arbitrary
binarization of its continuous per-node risk scores.

Safety adapter targets **final-report attack outcome**, not the safety of every
internal action or entire execution. Source `success` means attack reached final
workflow span. Model sees only the visible root request/workflow configuration
and root final report as a proposed text response; tools are unknown/empty and
internal chronology is NOT fabricated. Attack snapshots, risk category, judge
rationale, ground_truth and threat IDs never enter input. This is retrospective
response screening, not reconstruction of proactive internal-tool checks.

5566 cases have an unambiguous request/root report;34 missing final reports are
explicit rejections. Labels657positive/4909negative. No token overflows before
inference, max8071tokens; no truncation. Independent source-prefix audit5566/5566
passed. Dataset/model-state SHA and no-label-in-action-row checks passed; exact
normalized/masked request overlap with current train/validation bundle is zero.
Lexical exclusion is not proof of semantic independence or clean pretraining.

Fifteen fixed current checkpoints, same as ATBench run: eight frozen Mamba arms,
random Mamba, direct linear, two Transformers, LoRA epoch1, locked validation-best
epoch2, original Kev. No further checkpoint or threshold selection on this data.
Report Acc/F1/BA, AP/trapezoidal PR-AUC/ROC-AUC, confusion matrices, overall and
separately with/without defense. Positive prevalence is11.8%, so always-safe
accuracy is a necessary baseline. No action-level accuracy from trace labels.

Source: `graph-safety/scripts/prepare_external_replay.py` and
`evaluate_trajectory_actions.py`. Shared immutable source/launch:
`runs/external-replay-v1-launch/source`, queue PID2611522 on GPU2 runs ATBench
then Nemotron. Native long-context/OOM case rejections remain explicit.
Root `runs/action-models-nemotron-v1`.
Live/wait/copy `sh_14b3ad725001fLdEI1KsWjhf2Q`; do not poll completion.

## Public results are NOT directly comparable

StepGuard paper Table8 uses100 **manually relabeled full execution traces**:
25benign,25attack-exposed without unsafe execution,50unsafe executions.
GRPO Acc75.0/F176.6; Balance-GRPO Acc82.0/F183.0. Exact trace IDs and manual
labels have not been established here. Our5566 final-report source-outcome
cases are not that subset or task; do not place the numbers in one matched
leaderboard. NVIDIA paper primarily reports agent attack/risk outcomes, not
classifier AP/AUC on our adapter. Sources:
https://arxiv.org/html/2608.24777 (AppendixA.3/Table8)
https://arxiv.org/html/2511.21990v1
https://huggingface.co/datasets/nvidia/Nemotron-AIQ-Agentic-Safety-Dataset-1.0
