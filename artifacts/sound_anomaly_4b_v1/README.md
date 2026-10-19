# Результаты B/C: обучение завершено

Scheduler PID 3040382 завершился успешно, `status.json`: complete,
finished_unix=1792336351.7128527. Оба условия прошли CUDA gates и две эпохи
на GPU 2 последовательно после завершения A. Повторов не было.
Отбор по dev-select AP выбрал эпоху 2 в обоих случаях.

301 диалог, 886 действий: 739 safe и 147 unsafe. Порог 0.5 фиксирован.

| Условие | AP | Трапецеидальная PR-AUC | ROC-AUC | Event BCE | P, % | R, % | F1, % | Acc, % | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A, новый контроль | .50919 | .50879 | .79414 | .48854 | 100 | 2.04 | 4.00 | 83.75 | 0 | 144 |
| B, forecast + auto-pool | .37736 | .37331 | .76431 | .43415 | 50 | 1.36 | 2.65 | 83.41 | 2 | 145 |
| C, forecast + subsequences | .43952 | .43756 | .76599 | .46153 | 100 | 4.76 | 9.09 | 84.20 | 0 | 140 |

**Вывод:** B/C не улучшили основной критерий AP относительно нового контроля A.
C сильнее B по AP, но при 0.5 пропускает 140 из 147 unsafe действий.
Более низкий BCE B/C не означает лучшего ранжирования или решения проблемы
низкого recall. A→B — многофакторное сравнение, не чистая loss-абляция.
Все результаты validation-selected, один seed; независимой внешней оценки нет.

AP по эпохам: B .30418 → .37736; C .39430 → .43952.
Recall при FPR≤1% / ≤5%: B 4.08% / 21.77%, C 14.97% / 25.17%.
Медианы safe/unsafe scores: B .03341/.12198; C .02052/.08552.

## Проверки и расходы

- Причинность и reload: точная ошибка 0 у обоих условий, включая event-выход.
- Раздельные token/event градиенты к обоим блокам Mamba и соответствующим
  головам конечны и ненулевые; прямой loss вне action spans равен 0.
- Реальный train backward на 15 322 токенах пройден без обрезки.
- Shared core SHA-256 B/C совпадает с A:
  `711d8ffcbe2187d642f304b43634df9c311fa3d2d8577528829a35722c3501d8`.
- Параметры: B 1 526 323; C 1 551 026.
- Две эпохи с validation: B 133.42 с, C 147.08 с.
- Peak allocated memory longest gate: B 666 095 104 байта,
  C 666 292 736 байт; это память head-only gate, не backbone inference.
- Qwen не загружался; существующий BF16 cache читался с проверкой identity.

Выбранные checkpoint SHA-256:
- B: `3f47143d366ed738c5347f9d736421d11764476cd2f6e2cf35593463bcd60703`.
- C: `ec108ed084cff49e00090824c94cc6503af4b7bfe3a4ece240625b6eb356257d`.

Полные metrics, logs, raw event logits, selection, history и gate скопированы
локально. IDs, spans, raw token logits, состояния Mamba, checkpoints и
optimizer/RNG states находятся на сервере в `runs/sound-anomaly-4b-v1/`.
Source: `runs/sound-anomaly-4b-v1-launch/source`; source manifest сохранён
рядом со snapshot. Поле tensor `null_logits` — legacy-имя состояний Mamba,
не NULL logits; это явно описано в каждом `tensor_schema.json`.
