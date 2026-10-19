# Fixed-checkpoint StepGuard comparison of ready action models

Server root: `/var/local/danefedov/experiments/agent-safety/runs/action-models-stepguard-all-v1`.
Pipeline PID **1958244**, GPU2. Source snapshot:
`runs/action-models-token-curves-v1-launch/source`.

1. Preserve exactly epoch1 Mamba+LoRA exponential; evaluate online in batches of
   one to limit memory alongside the running native Kev training. Snapshot hashes
   are checked before scoring. Later training cannot replace this copy.
2. Build one shared frozen-Qwen StepGuard feature cache. Evaluate the completed
   cosine+ranking, hard-token and endpoint-only heads plus the random-head
   baseline. Original Kev and exponential scores are reused unchanged.
3. Train linear/quadratic/sqrt/plain-cosine heads, four epochs each, at most two
   concurrent jobs on GPU2; then evaluate their validation-selected checkpoints.
4. An independent evaluator waits by pidfd for the existing native Kev training
   process to finish successfully, snapshots its selected checkpoint and adds it.

Main split:3017 StepGuard external_test; separate dev:161. Same prepared causal
inputs as the previous StepGuard evaluation. New frozen heads use the shared
cache; online LoRA uses its own changed backbone. BF16 batch shapes can affect
scores numerically; no cross-batching bitwise equivalence is claimed. AP/AUC and
threshold0.5 metrics use common successful keys with label/state-hash checks.
No threshold fitting, training on StepGuard or external checkpoint re-selection.

`comparisons/initial.json`, `with_curves.json`, `with_kev.json` preserve successive
comparison stages. Only prediction files with a completed metrics JSON enter a
snapshot. StepGuard is previously viewed; this is exploratory transfer. No
unfinished LoRA eight-epoch condition is substituted for the requested epoch1.

The original running jobs and judges are not modified or stopped. Completion
copies omit the large records/features; checkpoints, manifests and predictions
are retained locally and on permanent server storage.
