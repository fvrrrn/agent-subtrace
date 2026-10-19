# Отдельный эксперимент: attention ДО Mamba

Пользователь исправил «после» на **до Mamba**. Никакое after-Mamba обучение
не запускалось. Этот run **завершён**,4эпохи,returncode0; выбрана
linear NULL по hard event validation BCE, не по ATBench.

Server `runs/curve-event-attention-before-v1`, immutable sibling `-launch/source`.
Автоматически стартует отдельным процессом после завершения shapes-v1,
GPU2,4эпохи fresh same seed, не fine-tune выбранного head checkpoint.

```
frozen cached Qwen -> LN/proj2560→256
 -> x + Linear(causal SDPA(QKV(LN(x))))
 -> two residual Mamba2 blocks -> final LN -> selected token/event readout
```

Один preLN residual attention layer,4heads×64,dropout0, без FFN и добавленных
позиционных embeddings. Qwen states уже контекстные; причинность дополнительно
ограничена `is_causal=True`. Нет gold-risk bias/teacher forcing/нового predictor;
attention видит всю разрешённую прошлую историю, включая безопасные permissions.
Два loss прежние: выбранная token-curve BCE плюс0.1event BCE, одинаковые
данные/optimizer/число эпох/validation selection с выбранным baseline.

Сохраняется исходная инициализация общих Mamba/readout параметров; attention
инициализируется дополнительно. Это сравнение добавленной capacity, не matched
parameter/compute budget. Event-loss не гарантирует истинные dependencies;
attention weights не считаются объяснением или token gold.

До train — прежние CUDA gates плюс ненулевые QKV/output gradients,
causal suffix/reload и longest backward. При провале train не запускается.
Heads/optimizer/RNG, оба loss, все validation token/NULL logits сохраняются.
Baseline и attention выбирают checkpoints по hard event validation BCE.
Сравнение будет validation-selected, не независимый тест.

Общий notification/copy `sh_14ecfe4a8001XfaZoSEdLolgpl` завершён,
артефакты локально. Внешние benchmarks для новых условий пока не назначены.

## Итог validation933(97unsafe), threshold.5

Оба selected checkpoints epoch2.

| Условие | BCE | AP | PR-AUC trapezoid | ROC-AUC | P,% | R,% | F1,% | Acc,% | FP/FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Linear NULL baseline | .175321 | .7182 | .7165 | .9440 | 77.42 | 49.48 | 60.38 | 93.25 | 14/49 |
| Attention before Mamba | .172993 | .7053 | .7036 | .9380 | 76.12 | 52.58 | 62.20 | 93.35 | 16/46 |

FPR1.67%→1.91%. BCE немного улучшился, recall+3.09pp,F1+1.82pp;
3дополнительных TP и2дополнительных FP. AP/PR-AUC/ROC-AUC ухудшились,
поэтому общего выигрыша нет. Recall52.58% всё ещё пропускает46из97unsafe.
Один seed, added capacity, validation reused for configuration/checkpoint
selection. Нельзя утверждать подтверждённую пользу внимания к причинным
зависимостям или улучшение ATBench. Полные значения: `comparison.json`.
