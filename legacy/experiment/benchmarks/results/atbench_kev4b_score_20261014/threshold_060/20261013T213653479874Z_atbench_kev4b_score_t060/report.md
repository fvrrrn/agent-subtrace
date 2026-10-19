# Kev-4B ATBench: score

Status: interrupted_by_user; attempted 114/1000; complete 111; context refusals 3; failures 0.

LINK: two-level score (no dependency / direct dependency), score >= 0.60; all prior nodes are scored in one request per current event. No edge types or argmax gate.
SAFETY receives the full prefix plus an untyped ancestor-subgraph hint. F1 uses the fixed .50 SAFETY threshold; ROC-AUC uses the trajectory maximum of the VIOLATION score, not LINK scores.
Full-prefix control scores are reused from the frozen completed baseline, matched by key and input hash.

| N | TP | FP | TN | FN | F1 | Recall | Specificity | BA | AUROC |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 111 | 37 | 30 | 27 | 17 | 0.61157 | 0.68519 | 0.47368 | 0.57943 | 0.56693 |

Cached baseline on the same 111 keys: F1=0.54206; BA=0.55799; ROC-AUC=0.5334632878492528. No new control inference.

All 1000 records are considered. Context refusals are explicit, without truncation; compare arms on common successful keys.
Native state limit 65536; branch limit 512; rows over 16384 use the official efficient-kernel context, logged per call.
Only tool names/descriptions/parameter schemas and recorded messages enter inputs; labels, taxonomy, reason, _source and _original_description do not. Tool interfaces are global, not separate graph nodes.
Exploratory score-protocol change after inspecting ATBench; one fixed threshold grid, no adaptive expansion.
Only trajectory labels exist. No edge accuracy, per-action recall, causality or prevented-harm claims.
Exact requests/encodings/logits/errors: calls.jsonl; per-action scores/graphs: records.jsonl.
