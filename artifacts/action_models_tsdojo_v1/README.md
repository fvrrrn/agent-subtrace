# Fixed best-model shortlist on TS-Bench-Dojo

User authorized evaluation after TS-Bench-Harm in the same GPU2 queue.
Root `/var/local/danefedov/experiments/agent-safety/runs/action-models-tsdojo-v1`.
Shared immutable source/launch `runs/action-models-tsbench-v1-launch`.

All1220 published steps: banking87, slack101, travel177, workspace855.
Original labels are binary:352unsafe/868safe; strict and loose coincide.
Same causal proposed-action protocol, preserved source rows, no truncation.
The benchmark tests action safety under prompt injection, not edge recovery,
token-level risk annotations or prevented harm in a live agent environment.

Five models fixed before Dojo outputs: LoRA exponential epoch1 (prior StepGuard
AP leader), frozen Mamba quadratic (prior frozen AP leader), Mamba cosine_rank
(prior frozen AUC leader), Mamba linear (original validation BCE winner),
original Kev. No selection on Harm or Dojo and no new teacher/model deployment.
This is an exploratory shortlist chosen using a different previously viewed
evaluation set, not an independently preregistered model selection study.

Report overall action-weighted Accuracy, F1, AP, trapezoidal PR-AUC, ROC-AUC and
separate four-domain metrics, all at unchanged threshold0.5. Report common-key
admission/exclusions, input/label hashes, pinned source/checkpoint revisions.
Same bundle overlap audit as Harm runs before inference. No training, threshold
optimization or changes to existing runs. Initial pipeline PID2314969; live/copy
`sh_14b0f2423001FFoYWsDXYnpb9A`, do not poll completion.
