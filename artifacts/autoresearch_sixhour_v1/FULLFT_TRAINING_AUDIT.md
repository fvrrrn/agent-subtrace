# CPU-аудит остановленного FullFT Qwen 4B

Дата: 18 октября 2026. Проверка только чтением исходников, JSON и CPU-метаданных сохранённых tensors. GPU forward/backward, обучение, возобновление, тесты, установка пакетов и подбор порогов не выполнялись. Confirm и внешние оценки не читались.

## Вывод

**Конкретной ошибки в распределённом масштабировании loss, накоплении градиентов, clipping или выборе endpoint не найдено.** Сохранённые Adam moments действительно FP32; стартовая голова побитово совпадает с all-layer LoRA. Это не доказывает численную эквивалентность FSDP и LoRA и не устанавливает причину низкого AP.

Обнаружена существенная несопоставимость исходного сравнения train/dev event BCE: train усредняется по диалогам и измеряется во время изменения весов, dev — по действиям после эпохи. При одинаковом с train диалоговом усреднении сохранённый dev BCE составляет **0.48037107**, а не 0.59279523. Разрыв с train 0.18998575 остаётся большим, но его нельзя целиком объяснять переобучением или использовать как доказательство ошибки training/evaluation.

Полный fine-tune начат с новой головы, а не с обученного frozen checkpoint. Gate проверяет восстановление начальных весов внутри того же FSDP-графа; он не проверяет обученный checkpoint, optimizer reload, однопроцессный evaluator или совпадение train/eval forward. Эти пробелы — реальные ограничения доказательств, а не доказанные дефекты вычисления.

## Источники и воспроизводимость проверки

Серверный корень: `/var/local/danefedov/experiments/agent-safety`.

- Run: `runs/autoresearch-sixhour-v1/fullft_small`.
- Авторитетный source: `runs/autoresearch-sixhour-fullft-launch/source/scripts/`.
- Далее `F` означает `train_autoresearch_fullft.py`, `L` — `train_autoresearch_lora.py`, `W` — `train_whole_dialogs.py`, `C` — `train_curve_event_heads.py`, `H` — `train_autoresearch_heads.py` в этом source.
- SHA-256 F: `957e2a672ad57a568aa0f019009fd7e67c7eb5a3c48bc15beed0e00e52962165`.
- Сверены с `protocol.json` хеши F, L, W, C, H и `prepare_whole_dialogs.py`: все совпали. Текущий рабочий source не использован вместо snapshot.
- Прочитаны `protocol.json`, `gate.json`, `history.json`, `status.json`, `epoch1/status.json`, `STOPPED_BY_USER.json`, `inputs.json`, сохранённые dev-select predictions.
- CPU: сравнение небольших `gate_reload/head.pt`; два `epoch1/training_rank*.pt` открыты через `FakeTensorMode` + `mmap=True`, без материализации большого optimizer/checkpoint. Backbone tensors не загружались.
- Runtime protocol: PyTorch 2.8.0+cu128, Transformers 5.17.0, Mamba 2.3.2.post1, Triton 3.7.1. Дополнительно прочитаны установленные исходники FSDP и Qwen. Их неизменность со времени запуска отдельным хешем не доказана.

## 1. Распределение данных и масштабирование градиентов

F:390–428 задаёт одинаковую перестановку `Random(SEED + epoch)` на обоих ranks, окна по четыре диалога, два microsteps. Индекс строки равен `micro * 2 + rank`: rank 0 получает позиции 0, 2, rank 1 — 1, 3. Это не два независимых sampler и не повторное обучение на всём train каждым rank.

Для реального окна размера W каждый локальный loss умножается на `2/W`. FSDP и DDP усредняют градиенты по двум ranks; поэтому после двух backward получается `sum(g_i)/W`. FSDP применяется только к backbone, DDP — только к голове: один параметр не усредняется последовательно обоими wrappers. `no_sync` отсутствует: коммуникация выполняется на каждом backward, но это само по себе не ошибка накопления.

Для 2009 диалогов: 502 полных окна + одно окно из одного диалога = **503 optimizer steps за эпоху**. В последнем окне один настоящий loss умножен на 2, три синхронизационных forward используют `order[0]` и `z.sum() * 0`. В конечном результате настоящий пример имеет вес 1, dummy — 0. Дополнительного Adam step для dummy нет. Наличие реального backward через все параметры сохраняет порядок collectives. Dropout/изменяемые forward buffers могли бы сделать такой dummy не совсем нейтральным к состоянию, но такой эффект в этом запуске не установлен.

Метаданные обоих сохранённых rank optimizer подтверждают `optimizer_steps=503`, `dialogs_applied_this_epoch=2009`, `training_epoch_complete=True`. Остановка после 596 steps означает ещё 93 законченных шага второй эпохи; оценка эпохи 1 не относится к этим позднейшим весам. Частичная эпоха не является второй завершённой эпохой.

## 2. Clipping и проверка градиентов

F:124–143 вызывает `model.clip_grad_norm_(inf, norm_type=2)`, затем прибавляет квадрат нормы DDP-head **один раз** и умножает обе группы градиентов на `min(1, 1/(norm+1e-6))`.

В прочитанном PyTorch `fully_sharded_data_parallel.py:1160–1197` FSDP суммирует квадраты локальных sharded norms между ranks, а unsharded-параметры не считает повторно. `max_norm=inf` даёт коэффициент 1 при конечной норме. Следовательно, это получение глобальной нормы backbone, а не первое скрытое clipping. В коде нет удвоения нормы головы и нет локальной нормы, ошибочно выдаваемой за глобальную.

F:117–121 проверяет конечность всех имеющихся gradients на обоих ranks. Gate проверяет наличие gradient у каждого непустого flat shard и ненулевые gradients projection/readout/обоих Mamba blocks. Это не доказывает наличие ненулевого gradient у каждого исходного параметра внутри flat shard, правильность величины каждого gradient или distributed-equivalence с последовательным контрольным backward.

**Ограничение логирования:** возвращённая `clip_combined` норма в training loop не сохраняется. Поэтому утверждения «backbone подавлял голову clipping» или «градиенты взрывались» не подтверждены. Gate norms 26.51 и 29.62 — до обучения и на отдельных диалогах, не статистика optimizer steps.

## 3. Цели, маски, padding и веса примеров

- L:38–46: mean token BCE внутри действия, затем mean по действиям диалога; event BCE также mean по действиям. Итог `token + 0.1 event`.
- L:15 импортирует `event_logit` из H, а не из C. H:43–46 явно возвращает `values[-1]` для `endpoint`. Имя конструктора головы `curve_linear_softmax` **не означает**, что FullFT обучался с MIL-readout.
- C:67–73: unsafe token target `(.9 + 1.2u - 1.5u²)/1.14`, safe target 0. Endpoint unsafe token target около 0.5263 конкурирует с hard event target 1; это согласованная искусственная цель, общая с контрольными условиями, не ошибка только FullFT.
- W:82–93: padding справа до кратности 128; attention mask равен 1 только на фактическом input. `use_cache=False`. F:65–70 обрезает выход до фактической длины. Loss использует только известные action spans; history и padding не объявляются safe.
- F:load_rows проверяет input hash, длины, непересечение spans, бинарные labels, разделение train/dev по group. Сериализация input повторно в этом аудите не декодировалась; причинность всех 2310 диалогов не перепроверялась.
- В model forward передаются только IDs и attention mask. Gold используется в loss и выборе gate-примера, не как модельный вход. Teacher/cached hidden features не вызываются: F.forward всегда идёт в W.encode. Импорт функции cached features не равен её исполнению.

По сохранённым `inputs.json`:

| Split | Диалогов | Действий | Unsafe | Unsafe по действиям | Mean доли unsafe по диалогам |
|---|---:|---:|---:|---:|---:|
| train_small | 2009 | 5647 | 641 | 11.3512% | 11.4506% |
| dev_select | 301 | 886 | 147 | 16.5914% | 13.6331% |

В train каждый action получает вес, обратно пропорциональный числу размеченных действий своего диалога; длинные actions не получают большего веса только из-за числа токенов. Class weighting/oversampling отсутствуют. Это не event-balanced objective и не токеновое среднее по всему корпусу.

## 4. Что означает train/dev разрыв

F training totals делятся на число диалогов; validation event BCE собирается отдельно по каждому действию (F:validate). Кроме того, train loss — онлайн-значения до очередного optimizer step, dev — один фиксированный checkpoint.

Существующие dev logits позволяют без inference пересчитать одинаковое диалоговое усреднение: `mean_dialog(mean_action(softplus(z)-y*z)) = 0.480371073949699`. Официальный action-weighted dev BCE = 0.5927952284552578. Разница весов объясняет 0.11242415 абсолютной разницы двух dev агрегатов, **не устанавливает причину оставшегося разрыва** с train 0.18998575.

Token BCE уже диалогово усреднён на обеих сторонах: train 0.26030898, dev 0.47016470; остаётся различие онлайн/фиксированных весов и train/dev состава. Нельзя назвать это чистой оценкой generalization gap фиксированного checkpoint без отдельной оценки train этим checkpoint.

## 5. FP32 optimizer, learning rates, initialization

F:run сначала создаёт голову после `seed(20261017)`, затем загружает native BF16 backbone и вызывает `.float()` до FSDP. Это FP32 master storage с исходными значениями, уже квантованными в BF16 при загрузке, а не восстановление исходной более высокой точности pretrained весов.

FSDP: `param_dtype=bf16`, `reduce_dtype=fp32`, `buffer_dtype=fp32`, `keep_low_precision_grads=False`, `use_orig_params=False`; head — FP32 под BF16 autocast. `optimizer_fp32` проверяется при создании и после каждого optimizer step.

Независимая CPU-проверка metadata обоих epoch1 optimizer:

- 33 backbone parameter entries и 28 head entries на rank;
- у всех 61 entries `exp_avg` и `exp_avg_sq` имеют dtype FP32;
- backbone LR `2e-6`, head LR `3e-4`, betas `(0.9, 0.999)`, eps `1e-8`, weight decay `0.01`;
- AdamW `foreach=False`; scheduler и warmup отсутствуют, LR постоянны;
- decay применяется и к bias/norm/специальным параметрам, поскольку отдельных no-decay groups нет. Это явная настройка, а не найденная причина результата.

Начальная голова FullFT и `lora_screen_best/gate_reload/head.pt`: одинаковые keys, **все tensors побитово равны**, 1 526 065 элементов state. Значит, расхождение с LoRA нельзя объяснять другой стартовой головой. С frozen A побитовое совпадение отдельно не проверялось.

LoRA snapshot использует тот же seed, head construction, данные, splits, broad/endpoint objective и порядок диалогов; LR adapters `1e-5`, fused AdamW, BF16 native backbone без общего backbone-autocast. FullFT меняет все backbone parameters, использует FSDP и другой optimizer backend. Это не один лишь переключатель числа обучаемых параметров. FullFT **не продолжает обученную frozen голову**; гипотеза о дестабилизации backbone случайно инициализированной головой правдоподобна, но не доказана.

## 6. Train/eval kernels, buffers и checkpoint gates

FullFT forward охвачен одним BF16 autocast и при train, и при validation; validation выполняется живой FSDP-моделью сразу после checkpoint save, без внешнего evaluator/reload. Поэтому ошибка будущего однопроцессного загрузчика **не может объяснить уже записанный dev epoch1**.

W.encode всегда задаёт `use_cache=False`. В прочитанном Qwen source `modeling_qwen3_5.py:574–650` переход к single-token recurrent decode требует cached states; обычный whole-dialog путь вызывает chunk delta rule. Прямого переключения chunk/recurrent по `self.training` в этом участке нет. Attention использует training-dependent dropout (`:812`); фактическое значение конфигурации dropout отдельно не извлекалось. Kernel hub dispatch, влияние grad-enabled режима и checkpoint recomputation на численную эквивалентность этим чтением не исключены.

F:146–225 gate:

1. Один training probe в eval: изменение будущих IDs не меняет прошлые logits (`causal_error=0`).
2. Сохранение full FP32 state + head, преднамеренное изменение параметров, strict reload в ту же FSDP-обёртку: logits совпали (`reload_error=0`).
3. Отдельный train backward: обе loss дают ненулевой gradient readout; выходной loss-mask корректен; gradients конечны.
4. Longest training dialog, 15 322 токена: backward прошёл.
5. RNG восстановлен, gradients очищены, optimizer steps в gate = 0.

**Gate не покрывает:** сравнение train/eval logits на одном input; обучение/шаг Adam с последующим reload; восстановление optimizer и RNG; non-FSDP evaluator; logits всех примеров/длин; эквивалентность распределённого backward контрольному; побитовое сравнение rank predictions. Head DDP wrapping происходит после gate.

F:checkpoint сохраняет floating state tensors только после assert FP32, включая persistent buffers, и отдельные optimizer shards. Неперсистентные buffers в state_dict не входят по правилам PyTorch. Настройка FSDP buffer dtype FP32 не равна доказательству, что каждый вычислительный путь и внешний загрузчик воспроизводят все buffers. Epoch1 dev прошёл на живой модели; проверка будущего evaluator должна оставаться отдельным требованием.

## 7. Что установлено и что остаётся гипотезой

**Подтверждено:** остановка пользователем, один завершённый epoch1 checkpoint, 503 steps в нём и 596 всего перед остановкой; ухудшение AP относительно LoRA/frozen A; высокий FP при 0.5; разные веса train/dev event BCE; fresh head вместо warm-start; отсутствие логов clipping; ограниченное покрытие gate; FP32 Adam; правильная по чтению формула distributed scaling.

Epoch1: AP 0.42209, трапецеидальная PR-AUC 0.42145, ROC-AUC 0.79076; P/R/F1/Accuracy при 0.5 = 33.23/72.79/45.63/71.22%, TP=107, FP=215, FN=40, TN=524. Это одновременно слабее ранжирование и другая шкала, а не просто «модель всегда safe». Приведённые в поручении ориентиры: LoRA AP 0.52707, frozen A AP 0.50919; этот аудит не пересчитывал их полный сравнительный отчёт.

**Не доказано:** catastrophic forgetting, чрезмерный LR, clipping как причина, повреждение рекуррентного backward, precision train/eval mismatch, label leakage или недостаточная длительность обучения. Нет оснований называть остановку OOM либо обещать улучшение от второй эпохи.

## 8. Самые дешёвые следующие диагностики

Ниже предложения, не команды возобновить обучение.

1. На уже сохранённых dev-select logits сопоставить модели на одинаковых action IDs, пересчитать также диалогово взвешенные BCE и срезы по источникам/длинам. Это CPU-only; параллельный validation audit занимается этой частью.
2. При необходимости CPU-only сравнить небольшие сохранённые head states с frozen A и metadata/hash manifests. Это уточнит matched initialization без inference.
3. CPU-only, если нужен анализ сдвига весов: последовательно читать отдельные tensors через mmap и сравнить epoch1 с gate checkpoint по слоям, нормам и доле изменений после BF16 округления. Не загружать несколько полных копий; такой анализ показывает изменение весов, но не доказывает причину ухудшения.
4. Только с отдельным разрешением на GPU-диагностику: один фиксированный train/dev input, одинаковый checkpoint, сравнение logits в train/eval и FSDP/предназначенном evaluator; отдельно проверить один distributed accumulation gradient против контрольного. Это существенно дешевле новой эпохи, но в текущем CPU-only поручении не выполнялось.
5. Любой будущий разрешённый запуск должен сохранять preclip backbone/head norms и коэффициент clipping, чтобы проверять гипотезу о дисбалансе, а не выводить её из отдельных gate norms. Новый training, warm-start или LR sweep не назначены данным аудитом.
