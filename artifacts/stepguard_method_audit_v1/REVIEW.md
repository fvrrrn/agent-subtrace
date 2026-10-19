# StepGuard и наш классификатор: проверка постановки до новых обучений

18 октября 2026. Прочитаны статья arXiv:2608.24777v1, включая приложения,
публичный recipe, SFT launcher, Balance-GRPO, prompts и static adapters.
GitHub revision: `93b580bc5fa1fb7ace16302bfb91563e4307ac1a`.
Точные источники и SHA-256: `sources.json`, локальные копии: `sources/`.
Новые обучения, GPU inference, подбор порогов и новые benchmark оценки не запускались.

## Главный вывод

Мы пока не воспроизвели обучение StepGuard и не провели matched сравнение,
позволяющее обвинить Mamba или бинарную классификацию. Одновременно различаются
backbone, supervision, данные, оптимизация, вычисления при inference и target
на траекторных benchmarks. Наиболее обоснованный приоритет — проверить
соответствие данных и целей задаче, затем численную корректность; очередная
замена pooling сама по себе эти различия не устраняет.

Это не доказательство, что rationale, балансировка или Instruct по отдельности
исправят наш результат. Их вклады в нашем pipeline ещё не изолированы.

## 1. Что действительно делает StepGuard

| Аспект | StepGuard | Наш текущий подход |
|---|---|---|
| Backbone | Qwen3-4B-Instruct-2507 | Qwen3.5-4B-Base |
| Выход | Генерация Analysis, Judgment, RiskSource, для trajectory UnsafeStep | Скалярные logits на выходах Mamba, event readout |
| SFT loss | Cross-entropy по токенам целевого ответа судьи | BCE по искусственным целям на токенах проверяемого действия + 0.1 event BCE |
| Обучающие задачи | 50% action, 50% trajectory | Action labels в whole-dialog обработке |
| Баланс | 50% safe, 50% unsafe в итоговом 7K релизе | 11.35% unsafe в small, 13.30% в full |
| Источники | StepGen: пары с общим префиксом, Refuse/Aware и benign tool reuse | Разнородные native/silver; 88.2% unsafe full происходят из StepSafe |
| SFT | Full FT, 2 эпохи, LR 2e-5, cosine, warmup 0.1, batch 16 | Остановленный FullFT: fresh Mamba/head, backbone LR 2e-6, head 3e-4, постоянный LR, batch 4 |
| Дополнительная стадия | 4K RL prompts, 100 updates Balance-GRPO после SFT | Нет аналогичной стадии |
| ATBench/R-Judge | Отдельный trajectory prompt, целая запись | Максимум causal action scores, включая Complete |

Источники: статья §3–5, C.1, F, G.2; recipe и run_sft.sh; наши
`FULLFT_TRAINING_AUDIT.md` и `DATA_SEMANTICS_REVIEW.md`.

**Token-level CE у авторов не означает supervision опасности каждого входного
токена.** Они учат генерировать ответ, включая объяснение. Наши token curves —
другая задача. Внешнее объяснение может быть training target отдельного
generative метода, но его нельзя подмешивать в вход классификатора.

Instruct backbone и генерация объяснения дают другой исходный навык и дополнительное
вычисление перед Judgment. В runtime authors сообщают 195.5 generated tokens/call
(Table 7); это не среднее для всех static benchmarks. Вывод о превосходстве
reasoning над single-forward классификацией без matched ablation невозможен.

## 2. Почему дело не только в GRPO

Table 11: отдельный SFT-only checkpoint, 3000 примеров, уже даёт опубликованные
Acc/F1: Dojo 91.3/86.8, Harm 73.8/78.2, ATBench 64.1/69.9, R-Judge 87.4/88.5%.
Это не финальный checkpoint Table 1 и не наши измерения.

Table 4: SFT → Balance-GRPO улучшает aggregate Acc/F1 79.7/81.0 → 82.2/82.1;
safe accuracy 69.3 → 78.4%, unsafe accuracy 91.1 → 86.4%.
Здесь RL главным образом исправляет operating balance, а не создаёт навык с нуля.
На независимых 100 Nemotron traces эффект больше (Table 8), но этот набор
не равен нашей прежней Nemotron адаптации.

Аblations дают конкретные указания на роль данных:
- Table 2: intermediate-prefix supervision повышает trajectory average
  Acc/F1 80.4/82.0 → 83.8/83.4; benign tool reuse повышает Harm F1 69.2 → 75.2.
- B.2: добавление 1K harmful-intent примеров даёт +7.8 п.п. Harm F1;
  это отдельный diagnostic checkpoint, не основной результат.
- Table 13: исходный retained pool сам несбалансирован по типам риска:
  62.9% indirect prompt injection. Нельзя описывать его как равномерный по 8 типам.

## 3. Что конкретно не так или не доказано у нас

### Данные и перенос

В full train всего три source attribution: StepSafe 13156 действий / 1980 unsafe,
StepShield 3593 / 246, LLM_contrastive_v2 128 / 19. Small содержит 449 исходных
групп на 2009 диалогов: число диалогов не равно числу независимых механизмов.
Модель может использовать стиль источника, формат и семейство сценария.
Это правдоподобная причина OOD gap, а не доказанный механизм по одному счётчику.

Уже подтверждены 19 противоречивых causal-input групп / 39 действий full train,
из них 3 группы / 6 действий small. Различаются только скрытые label-bearing
scenario metadata, правильно удалённые из входа. Одинаковому доступному входу
невозможно присвоить оба правильных ответа. Масштаб дефекта недостаточен, чтобы
без дополнительного анализа объяснить весь провал. Старые releases не менять.

StepGuard данные пока не были включены: наш audit нашёл 3178 уникальных action
inputs, отсутствие официального dev/test и отсутствие точных обнаруженных
пересечений. Статья описывает синтетическое происхождение и дополнительный
embedding/tool-filter audit — это свидетельства в пользу происхождения, которые
нужно учитывать. Но они не восстанавливают удалённые row-level family IDs.
Неизвестность provenance не является доказательством contamination.

### Семантика loss

Broad shape задаёт unsafe target от примерно 0.789 до 0.526 через пик 1, safe=0.
Это позиционный prior, не наблюдаемая вероятность вреда и не token evidence.
Одинаковые ранние префиксы safe/unsafe альтернатив не могут заранее отличаться
по ещё не прочитанному продолжению. B/C hard-label forecasting устраняют форму,
но также не дают настоящих token evidence labels.

**Важное уточнение прежнего объяснения про endpoint conflict.** Нельзя утверждать,
что broad + 0.1 event заставляет последний score оставаться около 0.526.
В коде token BCE усредняется по N токенам действия. Для unsafe независимый
endpoint logit минимизирует BCE с эффективной целью

`p* = (g(1)/N + 0.1)/(1/N + 0.1) = (0.52631579 + 0.1N)/(1 + 0.1N)`.

При N=1/10/100/1000 это 0.5694/0.7632/0.9569/0.9953. Это аналитический optimum
свободного logit для одного unsafe примера, не предсказание обученной общей сети.
Конфликт и искусственное supervision остаются, но очень низкие внешние scores
нельзя объяснить только значением broad на последнем токене.

### Обучение и проверка

CPU-аудит immutable FullFT source не нашёл ошибки scaling/accumulation/clipping
или выбора endpoint; Adam moments FP32, стартовая голова совпадает с LoRA.
Это не доказательство полной численной корректности.

Train BCE 0.190 — онлайн-среднее по диалогам; dev 0.593 — по действиям.
С диалоговым усреднением dev=0.480. Разрыв остаётся, но train не был повторно
оценён фиксированным checkpoint. Называть это точным generalization gap неверно.

FullFT начинался со случайной головы, а не с обученной frozen head. Возможная
дестабилизация backbone не проверена; clip norms в training loop не логировались.
Train/eval equivalence и trained-checkpoint replay ещё не подтверждены.
Однако epoch1 validation считалась живой FSDP моделью: ошибка будущего внешнего
loader не может быть причиной уже сохранённого AP=0.42209.

Завершившийся paired validation audit подтвердил все 886 одинаковых action keys,
labels, spans и сохранённых input hashes четырёх условий. Все 3544 token endpoints
побитово совпадают с event logits; sigmoid расходится не более чем на 1.2e-7.
Независимый пересчёт воспроизвёл AP FullFT/LoRA/A/original frozen:
.42209/.52707/.50919/.49634. Это не новый inference replay.

FullFT повышает scores обоих классов: FP=215 против 71 LoRA, recall=72.79%
против 51.02%. Recall при FPR≤1% при этом падает с 19.05% до 4.76%.
Следовательно, строго монотонная глобальная калибровка не исправит ранжирование.
Выраженный дефицит на AgentDoG FineGrainedTaxonomy: AP .40981 против .63834,
FP 81 против 20. Срезы по длинам и позициям смешаны с source/domain, причинность
не доказана. В семи детерминированных token-case все три unsafe FullFT endpoints
выше средних logits своих действий: это не поддерживает объяснение общим
подавлением unsafe endpoint, но не является репрезентативной локализационной оценкой.

Нельзя говорить «классификатор ничего не выучил»: старый LoRA имел validation
AP=0.8991, PR-AUC=0.8988, ROC-AUC=0.9839 (другой split, epoch2).
На Dojo старый epoch1 endpoint: AP=0.7113, PR-AUC=0.7125, ROC-AUC=0.8878,
но recall при FPR≤5% только 41.76%. Проблема — перенос и рабочая шкала,
а местами также слабое ранжирование. Эти модели не тождественны новому FullFT.

## 4. Несопоставимость benchmark результатов

1. ATBench: авторы обучают и оценивают trajectory judgment; мы action judgment
   с максимумом по событиям. Это разные targets, даже если final Complete у нас
   уже видит почти всю историю. Нельзя свести различие к «мы совсем не видим
   результаты»: прошлые outcomes доступны поздним causal endpoints.
2. Максимум не обучен как trajectory classifier и не является автоматически
   калиброванной вероятностью: с числом событий растёт возможность ложной тревоги.
   Формула `1-(1-f)^K` описывает только независимые ошибки, не наши зависимые scores.
3. Наш ATBench 981/1000: исключены 19 unsafe по длине. Harm primary=731 strict,
   у StepGuard=416 definite-only. Dojo номинально одинаковые 1220 и action target,
   поэтому trajectory mismatch не объясняет весь Dojo gap.
4. Авторы публикуют Accuracy и unsafe-class F1, не AP/PR-AUC/ROC-AUC.
   Сравнивать наш AP=0.42 с их Accuracy=0.95 нельзя; prevalence у наборов различается.
5. Статья Table 12 использует 564 R-Judge записи в similarity audit; наши 571.
   Это не доказательство, что весь основной inference был ровно на 564.

## 5. Что показал независимый просмотр публичного evaluator

- `ts_bench.py` действительно исключает gold score=0.5. Summary исключает
  unparsed/failed cases и confidence≤0; ATBench summary также считает только
  `counted_in_metrics`. Точные знаменатели основной статьи без predictions
  неизвестны. Наличие exclusion не доказывает существенного завышения качества.
- `at_bench.py` trajectory mode передаёт историю и final agent action, а
  last_action mode оставляет только safe cases: нельзя выбрать default
  last_action и получить эквивалент основной таблицы.
- ATBench adapter хранит risk labels и возможные analyses в context.memory,
  но прочитанный `StepGuardTrajProfile.build_prompt` их не подставляет. Наличие
  gold в metadata само по себе не является утечкой; здесь её не обнаружено.
- Prompt renderer передаёт names и первую строку descriptions инструментов,
  без полного parameter schema. Заявление «их преимущество — больше tool schema»
  не подтверждается текущим кодом.
- `docs-open/evaluation.md` документирует исправление R-Judge: сохранены все
  3098 turns / 571 records, изменились inputs 157 records (16 multi-group и
  141 с прежде пропущенным post-action feedback). Historical paper input не
  верифицирован, влияние на scores неизвестно. Нельзя приписывать исправленной
  версии старые published scores.
- Текущий `static_core.example.yaml` использует ATBench-Pro и дополнительные
  suites, а не точную Table 1. Нужна явная фиксация benchmark/config, не запуск
  текущего example под видом воспроизведения статьи.
- Recipe подтверждает warmup=0.1; launcher не задаёт held-out eval split,
  хотя §F говорит о selection на held-out static validation. Идентичность этого
  validation и точную selection procedure по просмотренным файлам не установили.
- Balance-GRPO code отличается от формулы §5: код использует сглаженные accuracy,
  повышает вес слабого класса И снижает сильного с clipping; статья явно описывает
  одностороннее повышение через max. Направление коррекции совпадает, формулы не
  идентичны. Для reproduction нужен pinned code, а не свободная интерпретация RL.

## 6. Что проверить прежде, чем снова обучать

| Проверка | Состояние |
|---|---|
| Статья, приложения, recipe, SFT launcher, prompt и adapters | Прочитаны, snapshot и hashes сохранены |
| Семантика action/trajectory, Harm mapping, Complete, admission | Различия установлены; matched IDs с авторами отсутствуют |
| Наши train source balance и точные causal contradictions | Подтверждены существующим исчерпывающим структурным аудитом |
| FSDP scaling, clipping, FP32 Adam и endpoint code | CPU-аудит пройден в ограниченном смысле: дефект не найден |
| Recompute всех 886 dev событий и paired source/length diagnostics | Завершено; метрики и 3544 endpoints подтверждены, причина обучения ещё не установлена |
| Один и тот же trained checkpoint: train/eval и FSDP/evaluator logits | Открыто, нужен отдельный численный replay без обучения |
| Fixed-checkpoint train vs dev loss с одинаковыми весами | Открыто; онлайн train history недостаточно |
| Детерминированные decoded inputs: запрос, permissions, roles, action span, outcomes | Код просмотрен; полного нового round-trip всех inputs не выполняли |
| Одинаковые causal input + task + данные для discriminative/generative методов | Такого matched эксперимента пока нет |

До закрытия диагностических пробелов новые обучения и финальные оценки поставлены
на паузу по уточнению пользователя. Это не команда уничтожать checkpoints или
автоматически останавливать уже запущенные отдельные процессы. Confirm остаётся
нетронутым. Следующее обучение должно проверять одну сформулированную причину,
а не менять одновременно backbone, данные, target и pooling. Возможные matched
контроли — предложения для отдельного согласования, не запущенные эксперименты.

## Локальные источники наших утверждений

- `../autoresearch_sixhour_v1/FULLFT_TRAINING_AUDIT.md`.
- `../autoresearch_sixhour_v1/DATA_SEMANTICS_REVIEW.md`.
- `../autoresearch_sixhour_v1/fullft_validation_diagnostics/README.md` и `run_20261018_v3/`.
- `../stepguard_training_v2/PROTOCOL.md`.
- `../whole_dialog_shared_threshold_v1/PUBLISHED_COMPARISON.md`.
- `../whole_dialog_token_summary_v1/ALL_MODELS_TOKEN_COMPARISON.md`.
- `../../scripts/train_autoresearch_heads.py`, `train_curve_event_heads.py`,
  `prepare_whole_dialogs.py`, `train_sound_anomaly_heads.py` (текущий локальный код;
  для выводов об исполненном FullFT использован отдельный immutable-source аудит).
