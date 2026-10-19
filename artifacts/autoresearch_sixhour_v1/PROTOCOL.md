# Six-hour autoresearch: prospective protocol

User authorized four H100 GPUs, subagents, frozen/LoRA/full fine-tune and
six wall-clock hours. Start/deadline are in server
`runs/autoresearch-sixhour-v1/authorization.json`:
1792327185.243998 /1792348785.243998. Time includes setup, training, validation,
confirmatory evaluation and reporting. Improvement is not guaranteed.

## Non-negotiable constraints

- Token-level BCE **plus** hard event BCE simultaneously, no event-only stage.
- Native causal whole dialogs; no label/rationale/future outcome as features,
  no truncation or silent failed-case exclusions. State resets per dialog.
- Existing immutable corpus/cache/checkpoints retained; stopped annotation
  never resumed. New annotations are not silently incorporated.
- Protected ATBench/R-Judge/StepGuard/ToolSafe never enter train or adaptive dev.
- No commits, dependency upgrades, tests or lint/format runs.
- Source/model/template/cache/input hashes, IDs, actual lengths, environment,
  costs, failures and optimizer/checkpoint state retained.

## Data and selection

An independent data agent audits actual family/source metadata and freezes
small/full train, source-held-out `dev_select`, untouched `dev_confirm`,
and the old easy validation as a diagnostic. No semantic independence claim
from mere hashes or source names. Split limitations are mandatory disclosure.
Do not change split after examining candidate results.

Fast experiments use one fixed grouped small-training subset. All candidates
use identical fully evaluated dev cases. Primary search criterion is
**dev_select AP** (trapezoidal PR-AUC and ROC-AUC reported separately), ties
lower hard event BCE, then lower compute. Checkpoint decisions may be made
from saved per-epoch outputs; hard event BCE defaults in old trainers must
not be confused with this experiment-level OOD criterion.

Report precision/recall/F1/Accuracy/confusions at0.5 and recall at FPR<=1%/5%;
do not use accuracy or a threshold fitted on a test benchmark for search.
Final threshold is fitted only to dev_select (max min(P,R), ties F1/recall/
higher threshold) then locked before dev_confirm or external evaluations.
Calibration is not established just by a sigmoid or BCE.

## Search budget

Start with a small shortlist, not a dense grid: endpoint control, MIL,
NULL and one statistically motivated signal readout. One event-loss weight
control only if component gradients reveal a concrete issue. At most two
follow-ups on one idea; retain all losing runs.

Compare one frozen-backbone system-instruction probe with native-template
reencoding and separate immutable feature identity, if causal preparation
passes. LoRA is prioritized over full fine-tune given the wall budget.
No automatic full fine-tune unless measured throughput/memory admit a
meaningful comparison within the deadline and confirmation reserve.

Up to two finalists receive larger-data training and untouched confirmation.
External benchmarks are a final report, not an adaptive idea-selection loop.
ATBench is already viewed and must be described as such. Prioritizing any
observed benchmark winner is post-hoc, not a new held-out discovery.

## Delegation / resource ownership

Data, implementation, literature and prompt-preparation agents have disjoint
file ownership; GPU work is centrally scheduled. Four-GPU allocation applies
only to our verified jobs; other processes/data must not be changed.
Wait on child processes/pidfds or completion notifications, not completion
polling. Avoid starting a job that cannot finish before the deadline.

## Deliverable

All-candidate table, fixed data manifests, heads/adapters, exact inputs and
scores, hypothesis decisions, threshold/calibration caveats, resource ledger
and practical next-step recommendation. No claim of causal localization,
prevented harm, statistically significant superiority or validated guard
without the corresponding evidence.
