# Kev-4B: sparse-hint graph replay on R-Judge

Status: running; selected 571, completed 450, failed 0, unattempted 121.

LINK rule: native winner is not NONE and winning-class score >= 0.55; SAFETY: matched binary choice, >=.50; trajectory alarm = max over agent actions.
No candidate filtering, joint normalization over IDs, oracle edges, threshold fitting or retries.

| Method | N | TP | FP | TN | FN | F1 | Recall | Specificity | BA | AUROC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| graph_pipeline | 450 | 187 | 68 | 141 | 54 | 0.754 | 0.776 | 0.675 | 0.725 | 0.757 |
| full_prefix_pipeline | 450 | 175 | 59 | 150 | 66 | 0.737 | 0.726 | 0.718 | 0.722 | 0.761 |
| full_trajectory_saved | 450 | 174 | 57 | 152 | 67 | 0.737 | 0.722 | 0.727 | 0.725 | 0.780 |

Both SAFETY arms receive the full observed prefix and the same question; only the graph arm adds a tentative typed dependency-subgraph hint. No history is removed. This changes LINK wording, edge admission and SAFETY representation together; it does not isolate the effect of sparsity. The 0.55 winning-class threshold was fixed before this run, not fitted to labels.
Exploratory evaluation on an already inspected benchmark, not independent held-out confirmation.
Full-trajectory scores are reused from the previous frozen 4B run, not inferred again; outcomes/info and question differ.
Only trajectory gold exists. No edge accuracy, per-step recall, early-detection or prevented-harm claims.
Native probs() reuses state within each request; fp32 unmerged, same shipped temperature/kernel set, no serving fusions/graphs.
Exact calls/token encodings/logits/errors: calls.jsonl; graph and per-step scores: records.jsonl.
