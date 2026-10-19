# Два loss одновременно: token-curve BCE + event BCE

Это новый user-authorized запуск **НЕ event-only** и не замена токеновой цели.
Основная цель в обоих условиях — BCE по выбранной широкой токеновой кривой.
Дополнительно0.1event-BCE, на каждом training step того же forward.

```
g(u) = (0.9 + 1.2u - 1.5u²)/1.14, u = i/N
target_i = y * g(i/N)
L_action = mean_action_tokens BCEWithLogits(z_i,target_i)
           +0.1 * BCEWithLogits(z_event,y)
L_dialog = mean_labeled_actions L_action
```

`g` — выбранная peak-scaled смесь Uniform/Beta, не плотность с суммой1.
Safe targets0, unsafe — g, не прежний linear0→1. Контекст и неизвестные
действия не получают токеновых targets, но участвуют в причинной памяти.
Нет отдельного event-only этапа, LoRA, attention или mono-ranking.

## Два условия из PDF

- **№2, curve_null, GPU2:** uniform-location NULL/witness;
  `z_event = logmeanexp(z_action_tokens) - null_readout(h_action_end)`.
  Дополнительная Linear256→1 NULL-head инициализирована нулём.
- **№3, curve_linear_softmax, GPU3:** `s=sum(sigmoid(z)²)/sum(sigmoid(z))`.
  Event-logit считается эквивалентной устойчивой формулой
  `LSE(2logsigmoid(z))-LSE(logsigmoid(z)+logsigmoid(-z))`, без clipping.

Одинаковые fresh Mamba/token-readout init, seed20261017, два residual Mamba2,
FP32 weights/BF16 autocast, native CUDA/Triton. **Две эпохи**; fused AdamW
lr3e-4/wd.01, clip1, accumulation4 whole dialogs. Кривые не имеют token gold;
высокие ранние targets могут быть непредсказуемы до прочтения различающего
продолжения. Оба компонента loss и их градиенты сохраняются отдельно.

## Данные, кеш, проверки

Прежние immutable whole-dialog-v2 данные:7083dialogs,6658train/425validation,
18712/933 labeled actions. Limit16384 без обрезки, те же588исключений.
Qwen3.5-4B-Base frozen; полный старый BF16 cache переиспользуется read-only,
Qwen **не загружается и не кодируется повторно**. Новые ответы остановленных
судей не добавлены в эти данные: сравнение использует один и тот же допущенный
корпус и точные IDs, чтобы не перестраивать кеш и не смешивать изменения данных.

Один диалог = один forward с loss всех известных action spans, без unroll
префиксов. Gate перед обучением проверяет ненулевые readout-градиенты **обоих
loss отдельно**, градиенты обоих Mamba, causal suffix/reload, loss masks и
longest15723 backward. Audit gradients discarded; train стартует заново.

Checkpoint выбирается по **hard event validation BCE**, не по benchmark.
Сохраняются epoch1/epoch2 heads, selected head, AP/PR-AUC/ROC-AUC/Acc/F1/P/R,
event scores и token endpoints, все validation token logits/NULL logits/spans.
Это один seed и validation-selected сравнение, внешние оценки автоматически
не запускаются. Никакие пользовательские attention-идеи в модели не включены.

Server `runs/curve-plus-event-v1`, immutable sibling `-launch/source`;
launcher PID1506894. GPU2/3 параллельно сначала проходят gate, затем train.
Watcher/copy **sh_14eb4e931001VBr4OesPEg6Qo1** завершён; оба обучения успешны,
returncodes0/0, по2эпохи. Оба selected checkpoints — epoch2.
Тяжёлый Qwen cache остаётся на сервере; артефакты копируются сюда по окончании.

## Итог validation

Общие933действия,97unsafe/836safe. Порог0.5, без подбора.

| Условие | AP | PR-AUC (trapezoid) | ROC-AUC | BCE | Precision | Recall | F1 | Accuracy | FP/TN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Curve + NULL | .6896 | .6881 | .9298 | .180797 | 76.79% | 44.33% | 56.21% | 92.82% | 13/823 |
| Curve + linear-softmax MIL | .6171 | .6151 | .9009 | .218367 | 85.71% | 18.56% | 30.51% | 91.21% | 3/833 |

NULL сильнее по ранжированию/BCE и recall, но пропускает54из97unsafe;
MIL пропускает79из97. FPR=1.56%/0.36%. Высокая Accuracy не означает
достаточную детекцию. Один seed, выбор по этой validation; внешние benchmarks
не запускались. Без matched curve-only контроля эффект добавления event-loss
не изолирован. Точные метрики/выбор: `comparison.json`, histories/checkpoints
и validation token logits — `arms/`.
