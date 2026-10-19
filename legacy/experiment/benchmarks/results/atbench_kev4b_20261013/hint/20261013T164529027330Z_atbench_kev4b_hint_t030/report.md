# Kev-4B ATBench: hint

Status: complete; attempted 1000/1000; complete 975; context refusals 25; failures 0.

Same causal action checks, SAFETY question, .50 decision threshold and max-over-actions aggregation.
LINK: independent typed choice per prior node; non-NONE winner with returned score >= .30.
baseline: full prefix, no LINK/hint; filter: original typed LINK and ancestors only; hint: evidence-focused LINK and full prefix with tentative typed ancestor-subgraph hint.
The two graph arms use their own graphs and different pre-existing LINK prompts, not a single shared graph.

| N | TP | FP | TN | FN | F1 | Recall | Specificity | BA | AUROC |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 975 | 250 | 230 | 273 | 222 | 0.52521 | 0.52966 | 0.54274 | 0.53620 | 0.53589 |

All 1000 records are considered. Context refusals are explicit, without truncation; compare arms on common successful keys.
Native state limit 65536; branch limit 512; rows over 16384 use the official efficient-kernel context, logged per call.
Only tool names/descriptions/parameter schemas and recorded messages enter inputs; labels, taxonomy, reason, _source and _original_description do not. Tool interfaces are global, not separate graph nodes.
Protocol/thresholds fixed from R-Judge before ATBench inference; no ATBench prompt/threshold fitting.
Only trajectory labels exist. No edge accuracy, per-action recall, causality or prevented-harm claims.
Exact requests/encodings/logits/errors: calls.jsonl; per-action scores/graphs: records.jsonl.
