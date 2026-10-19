# Kev-4B: one typed-choice LINK per candidate pair

User requested a single `choice`, rather than two independent binary questions,
to distinguish result and instruction dependencies. One full exploratory run
on the same 571 raw pinned R-Judge records; no pilot, candidate prefilter, extra
model, threshold fitting or trial repetitions.

## Fixed relation classification

Each earlier-event/current-action pair gets exactly **one four-option choice**:

- `NONE`: neither direct operational input nor followed instruction.
- `USES_RESULT`: direct result/data/identifier/resource use only.
- `FOLLOWS_INSTRUCTION`: the action is initiated/directed/changed by an
  instruction from that event's observations, without also using operational
  result data/resources from it.
- `BOTH`: both dependencies hold.

Instruction text alone is not operational-result use, and an instruction's
presence alone is not evidence that it was followed. Following an untrusted
instruction **does** create a dependency; LINK does not grant authorization.
Errors may supply observations. Shared topic/time or indirect-only sources
are not direct edges. Full exact questions/criteria are saved in calls.jsonl.

Add an edge iff the **native winning class is not NONE**. Do not sum the three
positive class scores and threshold that sum: a uniform four-class output would
otherwise look like .75 edge confidence. Native tie-breaking follows insertion
order, with NONE first. No class-score threshold is added or tuned. This differs
from the previous noul >=.50 rule, so the run changes relation categories,
wording/API and decision rule together, not the API alone.

`link_labels` and `link_distributions` retain every predicted pair type and its
four-class distribution. In typed-choice mode `link_scores` is the **winning
class's score**, not P(edge); it does not decide admission. A BOTH pair adds one
parent node, not two duplicates. SAFETY ancestors follow the union of all three
positive relation types. Types are stored for analysis, not added as extra
SAFETY hints.

## Unchanged controls and runtime

Same Kev-4B revision/base/head, shipped temperature, verified source/weights,
fp32 unmerged CUDA, native probs() shared-prefix path and observing head hook as
the first graph run. Same chronological node conversion, initial environment
inputs, always-visible profile/user context, exact SAFETY renderer/question,
causal action checks, .50 safety threshold and max-over-actions aggregation.

The full-prefix control is measured alongside graph SAFETY in the same run.
The earlier full-trajectory baseline is reused unchanged as a separate post-hoc
reference. Only LINK differs from the prior graph protocol; inputs include no
private labels, evaluator goals or annotator explanations. Strict row/branch
budgets remain8192/512, no truncation/retries. No edge or action gold exists.

This variant follows inspection of earlier R-Judge failures. It is an
**exploratory iteration on an already inspected benchmark**, not an untouched
held-out confirmation of a chosen graph method. Gold trajectory labels do not
enter inference, but that does not erase benchmark-informed prompt development.
No graph accuracy, early detection or prevention claim is supported. Training
contamination and probability calibration remain unknown.

## Remote execution

`llm01`, physical H100 GPU3; separate task directory
`/home/danefedov/kev4b-typed-choice-graph-20261013`. Existing isolated
environment/pinned immutable source/data are reused; prior artifacts stay
unchanged. No other model job is launched with this experiment.

```bash
CUDA_VISIBLE_DEVICES=3 \
HF_HOME=/dev/shm/danefedov-kev-rjudge-20261013/huggingface \
HF_HUB_DISABLE_IMPLICIT_TOKEN=1 \
TRITON_CACHE_DIR=/dev/shm/danefedov-kev-rjudge-20261013/triton \
TRITON_F32_DEFAULT=ieee \
/dev/shm/danefedov-kev-rjudge-20261013/venv/bin/python -u \
  /home/danefedov/kev4b-typed-choice-graph-20261013/rjudge_kev_graph.py \
  --link-mode typed-choice
```

Separate `results/*_rjudge_kev4b_typed-choice_graph/` stores exact requests,
native outputs/encodings/logits, predicted relation types, ancestor graphs,
action/trajectory scores, counts/metrics/errors, environment, hashes and source
snapshots. Label-type frequency is descriptive, not edge accuracy. Completed
outputs/log are copied locally. No tests, formatter or linter are run.

## In-progress density snapshot requested by the user

One-time audit of **225 completed records**, not completion polling, no new
model calls or changed thresholds. Snapshot:
[20261013_typed_choice_partial_density.json](results/20261013_typed_choice_partial_density.json).
It retains exact record keys and the source JSONL-prefix hash. Compare noul on
the **same 225 keys**; exclude 21 no-candidate cases from completeness ratios.

| Structural measurement | Noul, same keys | Typed choice, partial |
|---|---:|---:|
| Weighted direct-edge density | 66.4% | 74.1% |
| Directly complete graphs | 140/204 (68.6%) | 170/204 (83.3%) |
| Reachability-complete graphs | 172/204 (84.3%) | 173/204 (84.8%) |
| Weighted ancestor coverage | 76.5% | 74.8% |

Typed choice currently selects more direct edges but does not necessarily
retain more ancestors: edge placement and transitive paths matter. Full
reachability stays high. This is **not final full-dataset performance**, and
high density alone is not proof that the predicted edges are wrong. The
current SAFETY comparison uses the graph as a history filter, so a fully
retained prefix gives it the same input as the control. No top-k or density
penalty was introduced in response to this diagnostic.

## Completed full run (supersedes the partial estimates)

[Report](results/20261013T062003057516Z_rjudge_kev4b_typed-choice_graph/report.md),
[summary](results/20261013T062003057516Z_rjudge_kev4b_typed-choice_graph/summary.json),
[density](results/20261013T062003057516Z_rjudge_kev4b_typed-choice_graph/graph_density.json),
[paired diagnostics](results/20261013T062003057516Z_rjudge_kev4b_typed-choice_graph/paired_diagnostics.json).
571/571 successful without truncation/errors; all1461 full-prefix step scores
exactly match the prior noul control at native precision.

Graph TP215/FP79/TN191/FN86, BA71.08%, F172.27%, AUROC.74805 versus matched
full-prefix TP213/FP81/TN189/FN88, BA70.38%, F171.60%, AUROC.74493. Ten decisions
change: seven corrections and three new errors; small exploratory gain, no
reliable-benefit claim. All three earlier instruction-origin regressions recover
their alarm, though one does so via a shortcut while still missing E1->E2.

Direct density75.2%; directly complete442/528=83.7%, reachability-complete
447/528=84.7%, ancestors77.5%. Increased direct edges barely change total
ancestor retention. SAFETY token saving6.55%; graph+LINK tokens2.10x/full and
observed time1.96x (non-isolated). Total825.84s; maxrow1952/branch243.
Outputs/log/source snapshots copied locally. No further graph variant, model
size, threshold adjustment or repeated inference was started.
