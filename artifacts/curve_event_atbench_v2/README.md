# ATBench curve/event: исправленный запуск

Те же checkpoints NULL epoch2/MIL epoch4 из4epoch-v2, те же981cases/
4365actions и read-only frozen Qwen cache. Protocol/locked hashes сохраняются.
Primary=max event score включая Complete, sensitivity=tool-only; сравнение
с прежним frozen-linear endpoint. Class-conditional trajectory quantiles,
fraction>=.5 и AP/trapezoidal PR-AUC/ROC-AUC/P/R/F1/Acc/confusion при.5,
без подбора порога.19прежних length exclusions, без обрезки.

v1 import error до первого case сохранён отдельно. v2 удаляет лишний импорт,
не меняет inference или inputs/checkpoints. Source `-launch/source` immutable.
Server `runs/curve-event-atbench-v2`, launcher PID1645643, GPU2/3.
Watcher/copy `sh_14ec863820014dHVvqVnpb83A9` завершён.981/981 у обоих,
scoring без ошибок. Артефакты скопированы локально.
Тяжёлые raw/NULL token logits/IDs/spans остаются в server `*/tokens/`;
остальные артефакты копируются сюда. Статус — `status.json`.

Ранее просмотренный ATBench subset/causal replay, не официальный full1000
retrospective evaluator. Gold только траекторный. Scores не считаются
калиброванными вероятностями; эффект event-loss отдельно не изолирован.

## Результаты: primary all-events981

| Модель | Safe median | Unsafe median | Unsafe p90 | AP | PR-AUC trapezoid | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Прежний frozen endpoint | .1144 | .1859 | .4742 | .6016 | .6005 | .6323 |
| Curve + NULL | .3811 | .3688 | .6935 | .5092 | .5077 | .5058 |
| Curve + MIL | .1723 | .2674 | .4539 | .7075 | .7070 | .6879 |

| Модель | P | R | F1 | Acc | TP | FP | FN | TN | FPR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Прежний | 58.33% | 7.32% | 13.01% | 52.29% | 35 | 25 | 443 | 478 | 4.97% |
| NULL | 51.77% | 33.68% | 40.81% | 52.40% | 161 | 150 | 317 | 353 | 29.82% |
| MIL | 96.15% | 5.23% | 9.92% | 53.72% | 25 | 1 | 453 | 502 | .20% |

NULL поднял шкалу обоих классов, но почти не разделяет их. MIL лучше
ранжирует primary, однако unsafe score остаётся преимущественно ниже.5.
Больший score сам по себе не означает улучшение детекции или калибровки.

Tool-only980 sensitivity (478unsafe/502safe), AP/PR-AUC/AUC:
NULL=.6354/.6342/.6402; MIL=.7031/.7026/.6783;
прежний=.6504/.6494/.6815. Результат NULL чувствителен к набору событий;
это не повод исключать Complete из primary. Primary981 и sensitivity980
не полностью matched, не объявлять разницу чистым эффектом Complete.
Точные distributions/confusions в `comparison.json`.
