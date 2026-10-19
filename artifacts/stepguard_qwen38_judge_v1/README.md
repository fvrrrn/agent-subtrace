# Existing Qwen3.8-27B judge on StepGuard

Client PID2065533; existing GPU1 endpoint `http://127.0.0.1:8111`, unchanged
Qwen3.8-27B-FP8 with native MTP and thinking enabled. Concurrency2 shares service
with active corpus annotation. No new model, restart or paid API.

Root: `/var/local/danefedov/experiments/agent-safety/runs/stepguard-qwen38-judge-v1`.
Launcher/source: sibling `stepguard-qwen38-judge-v1-launch`.
Code: `graph-safety/scripts/evaluate_stepguard_teacher.py`.

Main3017 external_test and separate161 external_dev, original causal action
states only. Teacher protocol extended with a structured real-valued
`unsafe_score` in0–1. This is a **self-reported generative confidence**, not native
class probability/token likelihood or a calibrated probability. Report AP,
trapezoidal PR-AUC and ROC-AUC, plus score>=0.5 decisions.

Every normally completed finite score counts, including categorical uncertain.
No evidence-quality/teacher-consensus selection. Errors, uncertainty, score vs
verdict conflicts and one allowed 4096-token retry after initial2048 length
completion are reported separately. No semantic or HTTP retries. Full raw replies,
request hashes, usage and errors are preserved. No client-side input truncation.

Reference predictions are frozen copies of the seven completed student/baseline
conditions in the initial StepGuard comparison. Compare on common successful
keys, checking source-state hashes/labels. A shared Qwen teacher helped label
some student training data; this is not an independent human ceiling.
