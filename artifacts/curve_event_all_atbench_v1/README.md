# ATBench: все linear/step и attention конфигурации

User-authorized scoring, не обучение и не подбор порога. Запущен
`runs/curve-event-all-atbench-v1`, immutable sibling `-launch/source`,
launcher PID1976768. На GPU2/3 пары linear NULL/MIL, step NULL/MIL,
затем отдельно linear NULL+attention-before на GPU2.

Validation-selected checkpoints зафиксированы в `locked/`:

| Name | Training shape | Event readout | Attention | Epoch |
|---|---|---|---|---:|
| linear_null | linear local i/N | NULL | no | 2 |
| linear_mil | linear local i/N | MIL | no | 2 |
| step_null | all-one unsafe | NULL | no | 2 |
| step_mil | all-one unsafe | MIL | no | 3 |
| linear_null_attention | linear local i/N | NULL | causal before Mamba | 2 |

Те же ATBench981 (478unsafe/503safe),4365actions,19исключений по длине
сохранены. Native template/IDs/spans,656whole-dialog/325causal-prefix fallback,
max64773tokens, без обрезки. Read-only frozen Qwen BF16 cache, без загрузки
Qwen/перекодирования. Все raw token/NULL logits,IDs,spans сохраняются на
сервере в `*/tokens/`; небольшие predictions/protocol/locked/checkpoints/
costs/comparison копируются локально. Ошибки сохраняются, retries не назначены.

Primary — max trained native event score всех событий включая Complete.
Sensitivity — tool-only и matched all-events на одинаковых980tool-cases.
Gold только траекторный. Threshold0.5 фиксирован; никакого fit по ATBench.
AP/trapezoidal PR-AUC/ROC-AUC, P/R/F1/Acc/confusion, class-conditional
trajectory score quantiles/mean/fraction>=.5 для всех моделей.

В общую таблицу также входят сохранённые outputs broad NULL/MIL из
`curve-event-atbench-v2` и прежнего frozen-linear epoch2 endpoint;
они повторно не оцениваются. Source/checkpoint/reference hashes фиксируются,
при анализе сверяются exact common case IDs/labels/input hashes/action states.

Ранее просмотренный subset/causal replay, не официальный full1000 completed
trajectory evaluator. Выбор моделей только validation, scores uncalibrated.
Различия не изолируют event-loss; attention добавляет capacity и compute.
Не заменять primary лучшим tool-only sensitivity.

Watcher/copy `sh_14ef736300018fhk0h0C1b9hEY` завершён. Все5новых моделей
оценили981/981cases,returncodes0. Артефакты скопированы локально.
Статус завершения — `status.json`; тяжёлые tensors остаются на сервере.

## Primary:981all-events cases,478unsafe/503safe

| Model | AP | PR-AUC trapezoid | ROC-AUC | Safe median | Unsafe median |
|---|---:|---:|---:|---:|---:|
| Previous frozen endpoint | .6016 | .6005 | .6323 | .1144 | .1859 |
| Broad NULL | .5092 | .5077 | .5058 | .3811 | .3688 |
| Broad MIL | .7075 | .7070 | .6879 | .1723 | .2674 |
| Linear NULL | .6015 | .6007 | .6003 | .2818 | .3724 |
| Linear MIL | .4710 | .4692 | .5235 | .1896 | .2001 |
| Step NULL | .5090 | .5071 | .4868 | .3917 | .3596 |
| Step MIL | .5674 | .5657 | .5912 | .2086 | .2604 |
| Linear NULL+attention before | .6364 | .6357 | .6217 | .3315 | .4121 |

Fixed native event-score threshold.5:

| Model | Precision,% | Recall,% | F1,% | Accuracy,% | TP | FP | FN | TN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Previous frozen endpoint | 58.33 | 7.32 | 13.01 | 52.29 | 35 | 25 | 443 | 478 |
| Broad NULL | 51.77 | 33.68 | 40.81 | 52.40 | 161 | 150 | 317 | 353 |
| Broad MIL | 96.15 | 5.23 | 9.92 | 53.72 | 25 | 1 | 453 | 502 |
| Linear NULL | 62.62 | 26.99 | 37.72 | 56.57 | 129 | 77 | 349 | 426 |
| Linear MIL | 50.00 | .21 | .42 | 51.27 | 1 | 1 | 477 | 502 |
| Step NULL | 52.34 | 28.03 | 36.51 | 52.50 | 134 | 122 | 344 | 381 |
| Step MIL | 58.97 | 9.62 | 16.55 | 52.70 | 46 | 32 | 432 | 471 |
| Linear NULL+attention before | 69.43 | 33.26 | 44.98 | 60.35 | 159 | 70 | 319 | 433 |

Attention vs linear NULL: AP+.03494,PR-AUC+.03504,AUC+.02137,
P+6.81pp,R+6.28pp,F1+7.26pp;30more TP,7fewer FP. Но recall33.26%,
FN319/478 и FP70/503(FPR13.92%) всё ещё плохой operating point.
Broad MIL сохраняет лучший primary ranking, не лучший recall при.5.
Step не устранил проблему. Scores не откалиброваны, no threshold fitting.

## Matched tool-case sensitivity980

| Model | AP all-events | AP tool-only | AUC all-events | AUC tool-only |
|---|---:|---:|---:|---:|
| Previous frozen endpoint | .6016 | .6504 | .6317 | .6815 |
| Broad NULL | .5092 | .6354 | .5048 | .6402 |
| Broad MIL | .7075 | .7031 | .6873 | .6783 |
| Linear NULL | .6015 | .6964 | .5995 | .6957 |
| Linear MIL | .4710 | .6715 | .5225 | .6955 |
| Step NULL | .5090 | .6317 | .4858 | .6137 |
| Step MIL | .5674 | .6347 | .5904 | .6665 |
| Linear NULL+attention before | .6364 | .6764 | .6209 | .6683 |

Набор событий сильно меняет вывод: внимание улучшает all-events, но
tool-only AP/AUC ниже, чем у linear NULL baseline. Нельзя объявить общий
выигрыш attention/retrieval или убрать Complete ради лучшего числа.
Все PR-AUC/operating points/distributions трёх modes в `comparison.json`.
