# jaredpalmer/kev-0.8b full-trajectory baseline on R-Judge

Status: complete. Selected 571; completed 571; failed 0.
Model: `jaredpalmer/kev-0.8b` @ `bf75a6a8848ea6960ff2ed108d9ed44c2941174f`; Kev source `85ccf56a0afb952323c621055b02876e8498d60c`.

One native choice per full trajectory, threshold .50. No graph, generation, fine-tuning or calibration.
Same source state/question as Laya and GLiClass, verified against saved GLiClass input hashes.

| Measurement | N | TP | FP | TN | FN | F1 | Recall | Specificity | BA | AUROC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Kev: all | 571 | 49 | 83 | 187 | 252 | 0.226 | 0.163 | 0.693 | 0.428 | 0.496 |
| Kev: common | 569 | 48 | 82 | 187 | 252 | 0.223 | 0.160 | 0.695 | 0.428 | 0.497 |

All 571 raw pinned-repository records attempted, including initial environment observations.
Native rounded probabilities drive the reported metrics; unrounded scores/logits also retained.
Private label/risk_description/goal/scenario/attack_type are never sent to the classifier.
No truncation is permitted. Native admission and requested 8192/512 token limits are audited.
Only trajectory-level safety discrimination is measured, not edges, steps or harm prevention.

## Outcome and matched comparison

Kev-0.8B completes the task mechanically, but has essentially chance-level risk
ranking: AUROC .496 on all 571 records, .497 on the common 569. At the fixed
.50 threshold it detects only 49/301 unsafe records and flags 83/270 safe
records. BA=.428, F1=.226. It does not collapse to one answer, but is not a
useful frozen classifier for this workload with these inputs/questions.

Same 569 successful records, full-trajectory question/state strings, fixed .50:

| Model | Recall | Specificity | F1 | Balanced accuracy | AUROC |
|---|---:|---:|---:|---:|---:|
| Kev-0.8B | .160 | .695 | .223 | .428 | .497 |
| Kev-4B | .713 | .710 | .723 | .712 | .766 |
| Kev-27B v2 | .823 | .985 | .897 | .904 | .942 |
| Laya | .183 | .866 | .281 | .525 | .541 |
| GLiClass | 1.000 | .000 | .690 | .500 | .630 |

All-record score medians: safe .3004, unsafe .3257. Ranges .0688–.9649 and
.0686–.7901 overlap strongly, and safe p90=.83488 exceeds unsafe p90=.5561.
The low fixed-threshold recall is not merely evidence that a useful ranker
needs calibration: empirical ranking itself is weak. No threshold search,
reversal of labels, prompt tuning or repeat inference was performed.

## Completed-run audit

571/571 successful, no truncation/errors, max combined input 1,786, state 1,674,
branch 112 tokens. Source state/question hashes verified before inference for
all 569 previous GLiClass completions. Pointer/head/adapter/base revisions and
artifact hashes verified; native shipped temperature 2.3510958125672174 unchanged.
Official CUDA fp32 unmerged LoRA forward, same pinned code/kernel package set as
the 4B baseline. Native rounded scores drive metrics; unrounded outputs/logits,
actual kernel environment, token encoding and source snapshots are retained.

Physical H100 GPU **2**, concurrent with the separate Kev-4B graph job on GPU 3
at the user's explicit request. Download/load 24.96 s, inference loop 147.16 s,
total 172.12 s (~2.9 min). CPU/I/O and GPU 2 may be shared with other workloads;
this is not an isolated speed comparison. No jobs launched by us share a GPU.

Remote task `/home/danefedov/kev08-rjudge-baseline-20261013`; model outputs/log
and pinned source snapshot copied to the local workspace. Isolated cache/env
in `/dev/shm/danefedov-kev-rjudge-20261013` is temporary. Only the requested
0.8B full-trajectory baseline was run; no graph evaluation of 0.8B is included.
