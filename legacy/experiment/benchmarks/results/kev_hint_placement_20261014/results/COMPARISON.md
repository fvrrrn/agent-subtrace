# Kev-4B: identical hint in state versus question

10 stratified attempts (5 safe / 5 unsafe); 10 common complete cases; 62 action checks.
One shared graph; identical full chronological observations, hint block, base safety prompt and settings.
Fixed generic prompts before inference; no score-based prompt/threshold tuning. Exploratory, previously inspected ATBench.

| Placement | TP | FP | TN | FN | F1 | BA | ROC-AUC | SAFETY s/action | Cache hits |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| state_hint | 4 | 2 | 3 | 1 | 0.7273 | 0.7000 | 0.8000 | 0.2259 | 0 |
| question_hint | 2 | 1 | 4 | 3 | 0.5000 | 0.6000 | 0.8400 | 0.0629 | 52 |

Changed trajectory decisions: 3. Question/state SAFETY time ratio: 0.2784.
Request time includes native encoding, queue/model/readout and raw-logit audit, excludes JSON artifact writes and wait_idle captures.
Same in-process native Server as HTTP serving, no network overhead. Four GPUs partition cases, not conditions; arms alternate order per action.
Startup warmups excluded. New shape captures logged separately; elapsed shard times retain their overhead.
Serving precision/bucketing can change scores as part of the placement system; this is not a pure semantic prompt ablation.
No claim of statistical reliability, edge accuracy, action-level accuracy or prevented harm from ten trajectory-labelled cases.
