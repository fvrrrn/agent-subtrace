# Four-epoch frozen-Mamba token curves

Server: `/var/local/danefedov/experiments/agent-safety/runs/action-models-token-curves-v1`.
Shares original prepared records, Qwen cache and baselines read-only. One-layer
Mamba/head, no LoRA/ranking, same seed/init/optimizer/data as earlier frozen arms.

Four predeclared target fractions: linear `u`, quadratic `u²`, square-root `√u`,
and full cosine `(1-cos(pi*u))/2`. Target = `a+(y-a)*fraction`, with `a` the
previous uniquely verified annotated hard label, initially zero. These are
artificial token targets, not token safety gold; unannotated context stays
unsupervised. All endpoints reach the hard action label.

Exactly **four epochs** each. Checkpoint selection minimizes original validation
hard endpoint BCE, never StepGuard. Two concurrent jobs on GPU2, in two waves,
after initial StepGuard/LoRA evaluation frees memory. New heads are then evaluated
on StepGuard by the separate all-model pipeline, without external re-selection.

Before launch, CUDA loss-support/gradient and shared-initialization audits passed.
One generalization typo in the audit used the old `token_hard` condition name;
the failed preparation/source are archived as
`action-models-token-curves-v1-preflight-failed`. No training had started there;
the fix changed only the audit lookup, not targets/data/loss.
