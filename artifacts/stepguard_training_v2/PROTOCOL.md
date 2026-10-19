# StepGuard training v2 — отдельный протокол допуска

Статус: **BLOCKED_PROVENANCE_NOT_TRAINING_RELEASE**, CPU-аудит 18 октября 2026.
Новая пользовательская авторизация разрешает StepGuard в новой версии train,
включая прежние 3017 + 161 примеров. Она не меняет старые версии данных и не
разрешает включение официальных test или переупакованных защищённых benchmarks.

## Фактический релиз

- HF: `ninty-seven/stepguard_data`, revision
  `aadeae009800d8fbcd145575f1d1a6b37abb0950`.
- Только `sft3k/train` (3000) и `rl4k/train` (4000). В pinned manifest нет
  официальных dev/test. Наши `external_test` / `external_dev` — не официальные splits.
- Всего 7000 строк: 3500 action и 3500 trajectory; 3500 safe / 3500 unsafe.
- Action: SFT 1500 (750 / 750), RL 2000 (1000 / 1000); всего 1750 safe / 1750 unsafe.
- 3500 action-строк содержат **3178 уникальных точных causal-входов**, 2710
  request-family proxies. Все прежние 3017 + 161 ключей найдены в официальном train;
  их входы и исходные метки совпадают точно. 322 повтора нельзя выдавать за новые
  уникальные действия. Группы повторов сохраняются в дополнительном аудите.
- Лицензия Apache-2.0 относится к правам авторов релиза; README отдельно сохраняет
  права третьих лиц на включённые материалы. LICENSE и все файлы сверены по SHA-256.

## Семантика и исключение утечек

Action label оценивает текущее предлагаемое действие в контексте запроса,
истории и инструментов. Это не harmfulness запроса и не gold для всей истории.
В кандидатах сохранены точные `tool_interfaces`, `user_request`, `observed_history`,
`proposed_action`; guard-инструкция и ответ судьи удалены из модельного входа.
Контекст не обрезается, роли внутри непрозрачной истории не придумываются.

Trajectory label оценивает завершённую траекторию, содержащую результаты действий.
Его нельзя broadcast на токены или отдельные действия. В 1500 SFT trajectory-ответах
есть `UnsafeStep`; это индекс действия, не token evidence. Из-за outcome-зависимой
семантики это не автоматический допуск выбранного causal action как action gold.
3500 trajectory-строк сохраняются в инвентаризации, но не превращаются молча в
action-training records.

В схемах релиза нет token labels, offsets или размеченных evidence spans.
`Analysis` — объяснение внешнего разметчика, не вход и не token gold.
`risk_source` не является бинарной unsafe-меткой: unsafe и наличие риска различаются.

## Происхождение и блокировка

Публичные manifests прямо сообщают, что пути удалены. У SFT все 3000 значений
`source`, `sampling_family`, `paper_pool_source` равны `None` в сводке. RL содержит
только агрегаты внутренних пулов (`quality_score23`, `relabel_raw`, `v3_intent_cf`,
`pure_benign`), без связи строки с исходным source/family. Поэтому точные исходные
family groups и отсутствие переупакованного ToolSafe/ATBench/R-Judge не установлены.
Ссылка из manifest на `docs/experiments/current_data.md` не дала публичного файла:
проверенный raw URL репозитория StepGuard вернул HTTP 404.

Для 3500 action-строк проверены normalized-request и request-family совпадения
с защищённым ToolSafe: **0 найдено**. По IDs, causal-input hashes, normalized requests
и request-family hashes с frozen dev_select/dev_confirm: **0 найдено**.
Читались только manifests/metadata holdout, не их примеры или labels.
Сравнение по подлинным source groups невозможно: они не опубликованы.
Нулевые точные/нормализованные совпадения не доказывают независимость источников.
Для trajectory-строк этот overlap-аудит не заявляется.

**Допущено 0 действий.** Это карантин до выяснения происхождения, а не отмена
пользовательского разрешения на StepGuard. Кандидаты помечены `admitted: false`;
готового `train`/`dialogs.json` нет, прежние benchmarks остаются защищены.
Нужен row-level source/family mapping либо отдельное обоснование происхождения,
позволяющее исключить защищённые переупаковки. Не заменять неизвестные семьи
хешем запроса с утверждением, что это точные исходные семьи.

## Правила последующей подготовки после разрешения блокировки

1. Проверить всё официальное train, без выбора по предсказаниям/ошибкам моделей.
2. Любые подтверждённые protected-source или holdout overlaps исключить из нового
   train целыми связанными семьями; сохранить все IDs и основания.
3. Обе стороны одинаковых causal-входов с разными метками карантинировать.
   В текущих 3500 action-строках найдено 0 таких групп. Исходные метки неизменны.
4. Повторы объединять только с сохранением всех origins и без умножения action loss.
   Объединять последовательность в whole-dialog только при доказанном точном
   продолжении; не склеивать альтернативные ветви или неизвестные actor roles.
5. Использовать существующий whole-dialog протокол: один диалог — один forward,
   известные action spans; без результатов проверяемого действия, rationale,
   будущего или gold во входах; без обрезки. Native inputs готовить отдельно для
   каждого backbone после допуска и фиксировать tokenizer/template/input hashes.
6. Любые train-использованные StepGuard 3017 + 161 больше **не внешняя held-out
   оценка**. Старые результаты сохраняются как исторические; новые нельзя подавать
   как внешнее подтверждение. ATBench, R-Judge и ToolSafe остаются protected.
7. Loss-сравнение предшествует final full FT; token и event losses одновременно,
   только Mamba + head, финальный full FT ровно 1 эпоха. Этот CPU-аудит не запускает
   обучение. Общий deadline `1792348785`, резерв финальной оценки 3600 секунд;
   время не продлевается из-за подготовки.

## Артефакты

- Скрипт: `graph-safety/scripts/prepare_stepguard_training_v2.py`.
- Сервер: `/var/local/danefedov/experiments/agent-safety/datasets/stepguard-training-v2/`.
- `audit/inventory.jsonl`: все 7000 строк, hashes, labels, причины карантина.
- `audit/candidates_NOT_ADMITTED.jsonl`: 3500 полных action-входов без ответов судьи.
- `audit/inventory_summary.json`: точные input/source/script hashes и счётчики.
- `audit/contradictions.json`: обе стороны конфликтов (пусто).
- `audit/old_evaluation_coverage.json`: точное соответствие 3178 прежних записей.
- `audit/duplicate_groups.json`: точные группы повторов и все source keys.
- Тяжёлые action inputs остаются на сервере; локально скопированы аудиты.

Outputs validation-аудита другого агента, старые splits и source snapshots
не редактировались. GPU, API разметки, tests, lint и commits не запускались.
