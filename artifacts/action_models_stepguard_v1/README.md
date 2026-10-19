# Fixed-checkpoint StepGuard transfer evaluation

User-authorized evaluation, not training. Server root:
`/var/local/danefedov/experiments/agent-safety/runs/action-models-stepguard-v1`.
Source snapshot: sibling `action-models-stepguard-v1-launch/source`.
Evaluator: `graph-safety/scripts/evaluate_action_stepguard.py`.
Process PID at launch: **1858143**, GPU2 alongside ongoing jobs.

Compare original Kev-0.8B with frozen-Qwen + one-layer Mamba exponential.
Mamba checkpoint fixed to epoch3 selected by the original 848-action validation
BCE, head SHA256 `d40e2bbde6bd07af2cc968b107c6b3b86fa9adb9101d586ecfb156833b494b51`.
No re-selection, fine-tuning, inversion or threshold adjustment.

Main result: existing 3017 external_test actions (1484 unsafe / 1533 safe).
Auxiliary result: 161 external_dev actions (83 unsafe / 78 safe), separately.
This is the entire normalized benchmark-excluded StepGuard pool, not extra
training. Preserve published causal states, excluding private rationale and
current results. Identical Mamba renderer/segment tokenization to its training;
native Kev retains its original NOUL renderer and shipped temperature.

Maximum Mamba input is 9482 tokens, budget65536, no truncation. Native Kev also
checks its actual input length independently. Report all rejections and metrics
only on common successful keys with matching source state hashes and labels.
AP/ROC-AUC are primary; threshold0.5 metrics are also saved. No new Qwen token
cache is written: this evaluation extracts frozen features online.

StepGuard was viewed in earlier research and an old in-domain pilot trained on
a subset; none of its weights are reused. This is exploratory external transfer,
not untouched confirmation or an isolated Mamba-vs-Kev architecture ablation.
Concurrent GPU2 workloads prevent isolated throughput comparisons.
