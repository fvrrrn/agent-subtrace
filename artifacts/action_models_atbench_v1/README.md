# Fixed causal ATBench evaluation and Complete audit

Pinned **full ATBench**, not root test.json/ATBench500:
`AI45Research/ATBench@4476ef92ed8f85c8d58d8a5b9dfdf55aa7893138`,
`ATBench/test.json`, SHA256
`80c534b5f3517c872b528a2e6b64e34495206968edd184f4aed563d39f0cca09`.
Canonical dataset `datasets/atbench-action-replay-v1`.

All1000 trajectories considered. Shared admission rejects25 trajectories with
an oversized action prefix, no truncation;975 cases/4334actions remain:
2814tool calls,1520Complete events, max22205tokens among admitted prefixes.
Native encodings/OOM may reject additional whole cases, recorded by model.
Labels remain trajectory-only; **no action metrics from broadcast trace labels**.

Primary: maximum over all action scores, including Complete. Sensitivity:
tool-call-only maximum, excluding tool-less cases explicitly. Also report the
all-event maximum on exactly those tool-containing cases for matched comparison.
Complete events remain in the visible history of later actions. Record peak
event kinds and changed class decisions; never infer that every Complete is safe
or select the favorable aggregation after outcomes.

Each proposed action includes source thought/action only, no current result.
Visible history preserves original chronology, including user updates. Tool pool
retains injected descriptions/parameters but excludes private _original_description
and _source. No labels/reasons/risk categories in inputs. All4334 admitted action
prefixes independently reconstructed and audited before inference. Normalized/
masked request overlap with current training bundle is zero; not semantic proof.

Fifteen checkpoint snapshots locked before benchmark scores. Same current bundle
training; eight frozen Mamba objectives, original Kev/random, direct linear,
Transformer-linear/token-hard and LoRA1/validation-best2. Unfinished LoRA/Kev arms
are not represented as completed models. No training, calibration or ATBench
checkpoint/threshold selection;0.5 retained, TS-Harm fitted thresholds not used.
Matched original Kev and hard-label trained Mamba controls are the primary
comparators. ATBench previously examined; this is exploratory transfer.

Acc/F1/BA plus AP/trapezoidal PR-AUC/ROC-AUC on common successful trajectories,
with all exclusions. This is recorded replay, not prevented-harm measurement.
Published full-trajectory classifiers observe outcomes our causal action checks
cannot; do not call that a matched protocol or pure architecture comparison.

Code `graph-safety/scripts/evaluate_trajectory_actions.py` and
`prepare_external_replay.py`. Immutable source `runs/external-replay-v1-launch/source`:
prepare version retained as evaluate_trajectory_actions.py, run version as
evaluate_trajectory_actions_run.py adds per-split/Complete diagnostics and runtime
environment audit. Trainer/heads unchanged. Root `runs/action-models-atbench-v1`.
Queue PID2611522 onGPU2, ATBench then Nemotron; judges/other training unchanged.
Live/wait/copy `sh_14b3aa6ed001GkY6obigiMr8cZ`; do not poll completion.
