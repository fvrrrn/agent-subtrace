# Kev-4B: sparse-hint graph replay on R-Judge

Status: complete; selected 571, completed 571, failed 0, unattempted 0.

LINK rule: native winner is not NONE and winning-class score >= 0.30; SAFETY: matched binary choice, >=.50; trajectory alarm = max over agent actions.
No candidate filtering, joint normalization over IDs, oracle edges, threshold fitting or retries.

| Method | N | TP | FP | TN | FN | F1 | Recall | Specificity | BA | AUROC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| graph_pipeline | 571 | 230 | 96 | 174 | 71 | 0.734 | 0.764 | 0.644 | 0.704 | 0.743 |
| full_prefix_pipeline | 571 | 213 | 81 | 189 | 88 | 0.716 | 0.708 | 0.700 | 0.704 | 0.745 |
| full_trajectory_saved | 571 | 214 | 78 | 192 | 87 | 0.722 | 0.711 | 0.711 | 0.711 | 0.765 |

Both SAFETY arms receive the full observed prefix and the same question; only the graph arm adds a tentative typed dependency-subgraph hint. No history is removed. This changes LINK wording, edge admission and SAFETY representation together; it does not isolate the effect of sparsity. The 0.30 winning-class threshold was fixed before this run, not fitted to labels.
Exploratory evaluation on an already inspected benchmark, not independent held-out confirmation.
Full-trajectory scores are reused from the previous frozen 4B run, not inferred again; outcomes/info and question differ.
Only trajectory gold exists. No edge accuracy, per-step recall, early-detection or prevented-harm claims.
Native probs() reuses state within each request; fp32 unmerged, same shipped temperature/kernel set, no serving fusions/graphs.
Exact calls/token encodings/logits/errors: calls.jsonl; graph and per-step scores: records.jsonl.
