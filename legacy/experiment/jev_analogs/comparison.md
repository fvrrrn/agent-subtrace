# Jev-like модели: исходный алгоритм

Протокол: original XML prompts, один LINK-choice по ID прошлых событий,
порог рёбер .20, один SAFETY-choice, порог нарушения .50. Без RRSI и дообучения.
Данные: исходные 10 действий, 3 violations, 7 allowed, 8 gold-рёбер.
Каждая модель строит свой граф. Full-prefix — тот же safety-вопрос без графа.

## Завершённые измерения

| Модель | Граф: найдено нарушений / 3 | Граф: ложных блокировок / 7 | Full: найдено / 3 | Full: ложных / 7 | Правильные / предсказанные рёбра |
|---|---:|---:|---:|---:|---:|
| Laya, исходный reference | 0 | 0 | 0 | 0 | 1 / 14 |
| GLiNER2.5-Decide | 1 | 3 | 0 | 2 | 4 / 10 |
| Von | 1 | 0 | 3 | 4 | 2 / 8 |
| OpenDecider nano | 0 | 0 | 0 | 0 | 2 / 12 |

Kev-4B отменён по запросу пользователя во время загрузки/инициализации, до
первого inference. Нет предсказаний и метрик; отмена не считается провалом качества.
Дополнительных моделей после указания пользователя не запускали.

## Артефакты

- Laya: [исходный report](../results/20261012T222228823123Z/report.md).
- GLiNER2.5-Decide: [описание adapter](gliner/README.md),
  [summary](gliner/results/summary.json), [trace](gliner/results/trace.json),
  [token audit](gliner/results/input_audit.json).
- Von: [summary](encoders/results/von/summary.json),
  [trace](encoders/results/von/trace.json), [report](encoders/results/von/report.md).
- OpenDecider nano: [summary](encoders/results/nano/summary.json),
  [trace](encoders/results/nano/trace.json), [report](encoders/results/nano/report.md).
- Версии, mapping и запуск encoder-моделей: [encoders/README.md](encoders/README.md).

Для GLiNER проверено побайтовое совпадение всех 9 входов LINK и 10 full-prefix
входов с исходным Laya trace: совпадают state и question. Входы graph-SAFETY
закономерно различаются, поскольку зависят от собственного предсказанного графа.
Нативный GLiNER serializer сохраняет исходный текст и добавляет финальную точку.
Все 29 вызовов выполнены на MPS, без обрезки/разбиения; максимум 1073 токена.

Прежнее сравнение ModernBERT/GLiClass в ../competitors/ использовало compact/H0,
поэтому его числа не смешиваются с этой таблицей как одинаковые условия.
Перечень найденных семейств и первичные источники: [CATALOG.md](CATALOG.md).

Von и nano завершили по 29 вызовов на MPS, без обрезки (максимум 874 и 908
токенов соответственно). Von с графом ловит одно нарушение без FP, но baseline
ловит все три ценой четырёх FP. Nano разрешает все действия. По одной маленькой
траектории нельзя установить общую иерархию моделей или калибровку порогов.
