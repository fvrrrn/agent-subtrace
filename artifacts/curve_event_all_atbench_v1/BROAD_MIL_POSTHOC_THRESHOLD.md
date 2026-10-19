# Broad MIL: post-hoc ATBench threshold

User-authorized threshold fit. No inference/training rerun; original outputs
unchanged. ATBench981,478unsafe/503safe, max event score including Complete.
Source: `../curve_event_atbench_v2/curve_linear_softmax/predictions.jsonl`.
Source SHA256: `399f2b56a6cb7c6a45f873b362ed4d4b27599537411b2fdbdba92c02633b9475`.

Enumerate every unique trajectory score as threshold, decision score>=threshold,
keep all ties. Optimize max(min(precision,recall)), ties F1/recall/higher threshold.
**Fit and evaluate on same benchmark labels: post-hoc optimistic diagnostic,
not held-out calibration or validated deployment threshold.**

Selected threshold **0.21410755813121796**.

| Threshold | Precision | Recall | F1 | Accuracy | TP | FP | FN | TN | FPR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| .5 | 96.15% | 5.23% | 9.92% | 53.72% | 25 | 1 | 453 | 502 | .20% |
| .21410755813121796 post-hoc | 64.03% | 64.44% | 64.23% | 65.04% | 308 | 173 | 170 | 330 | 34.39% |

AP=.7074904195686593, trapezoidal PR-AUC=.7070450886222946,
ROC-AUC=.6878852408561186 unchanged by threshold. Lower threshold recovers
283additional unsafe but produces172additional false alerts. Low score scale
is not the only issue: broad overlapping class distributions remain.
No operational settings changed, no claim of cross-dataset transfer.
