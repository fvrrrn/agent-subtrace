# Current Qwen-action models on TS-Bench-Harm

Evaluation only, user authorized. Root:
`/var/local/danefedov/experiments/agent-safety/runs/action-models-tsharm-v1`.
Immutable code: `runs/action-models-tsbench-v1-launch/source`.
Code: `graph-safety/scripts/evaluate_action_tsharm.py` and `launch_action_tsbench.py`.
Queue PID2314969 on GPU2 evaluates Harm then Dojo. Pending evaluator PID2314970
blocks through pidfd on that queue and original training scheduler1659888, then
adds successfully completed final LoRA/Kev arms, without changing training.

Pinned ToolSafe snapshot, both harmful_steps525 and benign_steps206; all731 rows
attempted, preserving duplicates/source order, instruction/env/history/current
action. Causal model renderer unchanged; metadata and score do not enter input.
Gold original score preserved:0safe206,0.5intermediate315,1unsafe210.
Official ToolSafe strict mapping0->0,0.5/1->1 is primary; loose0/0.5->0,1->1
is secondary. Definite-only416 rows is supplemental, not full-benchmark result.
Source mapping code and file hashes recorded. No label adjudication or tuning.

Ready14: original Kev, random Mamba, eight frozen Mamba objectives, LoRA epoch1,
direct linear head, Transformer-linear and Transformer-token-hard. All learned
checkpoints are fixed by original validation, except explicitly fixed LoRA1.
Final four scheduled LoRA conditions and native trained Kev added only after
their existing training completes. Older Kev trained on ToolSafe is excluded.
No generative judge restarted; no new training or extra architecture search.

All731 inputs admitted; exact normalized/masked-request and input overlap against
the16937-row train/validation bundle is zero. Earlier bundle exclusions protect
all TS-Bench. This is lexical protection, not proof of semantic independence or
unknown pretrained-model data cleanliness. Corpus has been viewed historically.

One frozen-Qwen BF16 feature cache per benchmark; hash must match original base.
LoRA evaluated online using its adapter, native Kev uses shipped temperature.
No truncation; native admission failures recorded separately. Compare on common
successes with source-state/label matching, retain excluded keys. AP,
trapezoidal PR-AUC, ROC-AUC, Accuracy/F1 at0.5; metrics by input file as well.
No threshold calibration or checkpoint selection on TS-Bench. One seed.

Ready Harm+Dojo live/copy `sh_14b0f2423001FFoYWsDXYnpb9A`;
pending final Harm live/copy `sh_14b0f51cc0012Ay5wlb1ZAlYMV`.
Do not poll completion. Heavy records/features remain on permanent server storage;
manifests, selected checkpoints, predictions and comparisons are copied locally.
