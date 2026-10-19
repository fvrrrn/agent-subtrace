# Agent Subtrace — композиционная безопасность агентных траекторий

Исследование причинной оценки безопасности действий с учётом истории,
происхождения данных и разрешений. Текущий подход: Qwen → два residual Mamba →
потокенный score. Обучение по целым диалогам, одновременно token-loss и event-loss.
Это replay записей, а не измерение предотвращённого вреда.

**Корень Git — этот каталог (`graph-safety/`).** Remote:
https://github.com/fvrrrn/agent-subtrace. Коммиты и публикация выполняются отдельно.

## Что где находится

| Путь | Содержимое |
|---|---|
| [AGENTS.md](AGENTS.md) | Правила работы, ограничения и исследовательский контекст; старые PID не считать живыми |
| [WHOLE_DIALOG_TRAINING.md](WHOLE_DIALOG_TRAINING.md) | Подготовка целых диалогов, архитектура и исходный протокол |
| [TRAIN_ACTION_MODELS.md](TRAIN_ACTION_MODELS.md) | Более ранний prefix-wise этап, не текущий whole-dialog метод |
| `src/` | Библиотечный код |
| `scripts/` | Подготовка данных, обучение, gates, оценка и сборка отчётов |
| `artifacts/` | Метрики, отчёты, протоколы, manifests, логи и сохранённые версии экспериментов |
| `manifests/`, `protocols/`, `annotations/` | Идентичности данных, протоколы и контроль разметки |
| `knowledge/`, `prompts/` | Исследовательские материалы и шаблоны |
| [legacy/experiment/](legacy/experiment/) | Переносимые копии ранних графовых экспериментов, baselines и результатов |
| [docs/EXPERIMENT_INDEX.md](docs/EXPERIMENT_INDEX.md) | Полный указатель локальных каталогов экспериментов и их отчётов |
| [manifests/repository_collection.json](manifests/repository_collection.json) | Реестр копий и тяжёлых файлов с SHA-256; первоначальное поле `excluded_preserved_files` предшествует включению LFS |

## Эксперименты и результаты

Сводка на 19 октября 2026. Числа разных validation/splits не образуют общий
leaderboard. В частности, старую validation нельзя сравнивать напрямую с более
сложным source-held-out dev-select.

| Этап | Где читать | Основной результат / ограничение |
|---|---|---|
| Первые графы LINK→SAFETY, Kev и конкуренты | [legacy/experiment/benchmarks/](legacy/experiment/benchmarks/), [legacy/experiment/results/](legacy/experiment/results/), [legacy/experiment/evolution/](legacy/experiment/evolution/) | Устойчивый общий выигрыш графа не установлен; сохранены отрицательные результаты и аудиты |
| Ранние action/prefix модели | Каталоги `artifacts/action_models_*`, полный список в указателе | Не являются обучением непрерывного состояния по всему диалогу |
| Whole-dialog frozen/LoRA ± mono | [whole_dialog_4b_v3](artifacts/whole_dialog_4b_v3/) | На прежней validation AP frozen .6918, LoRA .8991; один seed, выбор эпохи по validation |
| Внешняя оценка whole-dialog моделей | [whole_dialog_benchmarks_v1](artifacts/whole_dialog_benchmarks_v1/) | Dojo, Harm, StepGuard train subset, ATBench, R-Judge; LoRA оценивалась на отдельно выбранной эпохе 1 |
| Endpoint/max/top10/tail8 | [ALL_MODELS_TOKEN_COMPARISON.md](artifacts/whole_dialog_token_summary_v1/ALL_MODELS_TOKEN_COMPARISON.md) | 96 строк; универсального лучшего readout нет, перенос порога слабый |
| Общий порог и published baselines | [whole_dialog_shared_threshold_v1](artifacts/whole_dialog_shared_threshold_v1/) | Подбор по тем же benchmark-меткам — post-hoc, не held-out калибровка |
| Broad + NULL/MIL, 2 и 4 эпохи | [curve_plus_event_v1](artifacts/curve_plus_event_v1/), [curve_plus_event_4epoch_v2](artifacts/curve_plus_event_4epoch_v2/) | Одновременные token/event losses; same-seed rerun не был побитово идентичен |
| Linear/step × NULL/MIL и attention-before | [curve_event_shapes_v1](artifacts/curve_event_shapes_v1/), [curve_event_attention_before_v1](artifacts/curve_event_attention_before_v1/) | Attention не дал общего выигрыша по ranking |
| Проверка новых readouts на ATBench | [curve_event_all_atbench_v1](artifacts/curve_event_all_atbench_v1/), [curve_event_atbench_v2](artifacts/curve_event_atbench_v2/) | Broad MIL AP .7075, но recall@.5 всего 5.23%; subset 981/1000, не официальный full-trace leaderboard |
| Source-held-out autoresearch | [autoresearch_sixhour_v1](artifacts/autoresearch_sixhour_v1/) | Dev-select: 301 диалог / 886 действий / 147 unsafe. Base LoRA small AP .52707; full-train .55278 |
| Остановленный FullFT и аудит | [FULLFT_TRAINING_AUDIT.md](artifacts/autoresearch_sixhour_v1/FULLFT_TRAINING_AUDIT.md), [диагностика](artifacts/autoresearch_sixhour_v1/fullft_validation_diagnostics/) | Epoch1 AP .42209; остановлен пользователем, автоматически не возобновлять |
| Frozen Base A/B/C | [sound_anomaly_4b_control_v1](artifacts/sound_anomaly_4b_control_v1/), [sound_anomaly_4b_v1](artifacts/sound_anomaly_4b_v1/) | A AP .50919, B .37736, C .43952; низкий recall |
| Frozen Instruct A/B/C и кривые | [frozen_instruct_4b_v3](artifacts/frozen_instruct_4b_v3/) | A broad/linear/step AP .47718/.48286/.50798; общего выигрыша Instruct нет |
| Adaptive weighting × last/three-layer | [streaming_weighted_multilayer_v1](artifacts/streaming_weighted_multilayer_v1/) | Лучший last broad AP .51855, recall@.5 10.88%; три слоя не дали общего выигрыша |
| Warm-start ablation: frozen/LoRA × step/broad × uniform/adaptive | [weighted_broad_lora_v1](artifacts/weighted_broad_lora_v1/), [recovery outputs](artifacts/weighted_broad_recovery_v1/) | Все 8 завершены; лучший AP .53307, LoRA step uniform. Общий старт уже обучен на weighted broad |
| Class weighting × сила event-loss | [streaming_followup_v1](artifacts/streaming_followup_v1/), [v2 outputs](artifacts/streaming_followup_v2/) | Три дополнительных условия завершены; class weighting ranking не улучшил |
| Fresh-init frozen/LoRA × step/broad | [fresh_factorial_v1](artifacts/fresh_factorial_v1/) | Четыре новых двухэпоховых условия; запуск подтверждён. `server_snapshot/` — копия состояния, не обещание завершения всей очереди |

### Данные, разметка и research

- [corpus_v1](artifacts/corpus_v1/), [data_bundle_v1](artifacts/data_bundle_v1/):
  происхождение данных, исключения, версии корпуса. Разметка остановлена;
  69 754/80 000 ответов не являются завершённым training release.
- [validation_admissibility_v2](artifacts/validation_admissibility_v2/): очистка
  только подтверждённых конфликтов; dev-select не изменён, oldval 421/917.
- [stepguard_training_v2](artifacts/stepguard_training_v2/): официальный релиз
  содержит train; происхождение строк недостаточно установлено для допуска в train.
- [stepguard_method_audit_v1](artifacts/stepguard_method_audit_v1/): статья, код,
  различия протоколов и streaming research.
- [fresh_factorial_v1/RESEARCH.md](artifacts/fresh_factorial_v1/RESEARCH.md):
  последующие гипотезы и источники; `diagnostics/` — измеренные проверки.
- [READOUT_IDEAS.md](READOUT_IDEAS.md), [HISTORY_ATTENTION_IDEA.md](HISTORY_ATTENTION_IDEA.md):
  идеи; их наличие не означает, что все варианты запускались.

## Как читать результаты

Сначала открыть README/PROTOCOL конкретного запуска, затем `selection.json`,
`metrics.json`, `status.json`/`complete.json` и manifests. Исходные failed runs
сохраняются: старый `running` или существующий checkpoint не доказывает завершение.

AP, трапецеидальная PR-AUC и ROC-AUC — разные метрики. Для решений читать также
precision/recall/F1/Accuracy и FP/FN. Один seed, выбор по dev и просмотренные
benchmarks не дают независимого подтверждения качества. Потокенные кривые —
искусственные обучающие цели, не gold evidence spans и не калиброванные вероятности.

## Что входит в Git и где тяжёлые артефакты

Отчёты, небольшие метрики, конфигурации и исходники хранятся в обычном Git.
Для `.pt`, `.pth`, `.safetensors`, `.bin`, `.zip`, `.jsonl`, `.pdf` и остальных
локальных файлов от 1 MiB настроен **Git LFS** в `.gitattributes`. Данные сохраняются
в LFS, а в обычном Git — небольшие указатели. Это включает локальные checkpoints
и optimizer states; server-only данные автоматически не скачиваются.

Перед работой установить Git LFS и выполнить `git lfs install`; после клонирования
для получения бинарных данных — `git lfs pull`. Хранение и скачивание LFS имеют
отдельные квоты GitHub. Настройка и staging не означают публикацию: до коммита и
push файлы остаются только локально. Размеры, исходные пути и SHA-256 первичной
сборки записаны в [реестре](manifests/repository_collection.json); его старое поле
`excluded_preserved_files` не означает текущее исключение локального файла из LFS.

Тяжёлые server-only features, token tensors, checkpoints и полные датасеты:
`llm01:/var/local/danefedov/experiments/agent-safety/`. Точные `runs/...`, revision,
input/template/weight hashes нужно брать из протокола соответствующего запуска.
Реестр локальной сборки не является полным инвентарём сервера.

Исторические документы/snapshots сохранены без переписывания; старые пути
`graph-safety/...` означают путь от прежнего родительского каталога `paper/`.
Рабочие материалы из `method.md`, `train/`, `todo/`, сторонние окружения и модели
не переносились. Исходный `experiment/` сохранён вне репозитория как архив.

Для повторной локальной сборки: `python scripts/collect_repository_results.py`
из окружения с `tqdm`. Скрипт не запускает обучение, оценку или публикацию и не
меняет Git/LFS policy. Новые большие файлы других типов нужно явно добавлять
через `git lfs track --filename <path>` до `git add`.
