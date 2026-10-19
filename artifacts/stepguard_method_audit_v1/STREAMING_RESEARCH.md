# Потокенные цели: свежие работы и применимость

Обзор 18 октября 2026, без новых запусков. Полные тексты проверены:
- SingProbe, 31 августа 2026: https://arxiv.org/html/2608.30703v1 (§2.2.2, §2.3, §4.1).
- StreamGuard / Predict, Don't React, 5 апреля 2026: https://arxiv.org/html/2604.03962v1 (§2, §3.1, §3.3).
- Constitutional Classifiers++, 8 января 2026: https://arxiv.org/html/2601.04603v1 (§5.1–5.2).

## Вывод

Потокенный выход стоит сохранить. Нужны различимые семантики: обнаружение
нарушения по уже видимому префиксу и прогноз unsafe завершения текущего действия.
Ни одно не равно intrinsic вредности изолированного токена. End-label не
идентифицирует единственную правильную кривую внутри действия.

## Weak-label обучение без заданной формы

CC++ использует causal sliding-mean logits и softmax-взвешенную сумму
ПОТОКЕННЫХ BCE: sum softmax(zbar/tau)_t * BCEWithLogits(zbar_t,Y).
Это не BCE от взвешенного среднего вероятностей, как event auto-pool B;
и не обычная равномерная hard-label token BCE из B/C. При tau→0 акцент на
высокорисковых позициях: на safe примере они дают сильный штраф ложной тревоги,
на unsafe — положительное свидетельство без фиксированной позиционной кривой.
Параметры статьи M=16, tau=1; training SWiM, inference EMA. Их асимметрия
и train/inference mismatch должны быть явно отражены при адаптации.

SingProbe адаптирует силу weighting по разбросу predicted probabilities:
alpha=softmax(beta*p/T), beta=clip(std(p)/tau,0,1), затем sum alpha*tokenBCE.
В начале почти uniform, при различимых scores — фокус на высоком риске.
Это мотивировано шумными пиками при инициализации. Std в формуле берётся по
валидным позициям batch; нормировка alpha — внутри примера.
Наличие/отсутствие stop-gradient в весах требует проверки implementation;
это меняет градиенты, нельзя молча выбрать вариант и назвать reproduction.
Weak-label weighting сам по себе не доказывает истинную localization и не
исключает shortcut. Поиск evidence по всему action используется при training;
каждый inference logit остаётся causal.

Для нас возможна адаптация: прежний frozen backbone → Mamba → z_t;
weighted token BCE только по current action + одновременно endpoint event BCE.
State Mamba сохраняется по диалогу, нормировка весов и smoothing boundaries
привязаны к action. Это предложение, не реализация. Не добавлять attention.
Сначала сравнить именно loss на прежних features, чтобы не менять всё сразу.

## Признаки нескольких слоёв

CC++ показал пользу нескольких/всех слоёв; SingProbe по умолчанию берёт три
слоя (shallow/middle/deep), с attention-based head. У нас только last_hidden_state.
Можно отдельно проверить concat трёх слоёв → прежняя projection → Mamba, без
attention. Это не воспроизведение SingProbe. Нужен новый cache; другие checkpoints
и loss не менять в том же сравнении. Исходные модели авторов и задачи отличаются
от отдельного frozen Qwen4B, читающего агентные traces: перенос не гарантирован.

## Forecasting с содержательными soft targets

StreamGuard определяет V(prefix)=E_{continuation~q}[judge(completed response)].
Target строится несколькими rollout продолжениями; intermediate targets мягкие,
terminal target — hard gold. Это supervised value fitting, не обязательный GRPO.
Референс использует 4 generator модели × 4 rollouts: разметка дорогая.
В нашем применении horizon разумно ограничить завершением текущего tool action,
сохраняя доступную историю и tools. Никакие вызовы tools не исполняются.
q и judge определяют смысл score: это не безусловная вероятность реального вреда.
Нужны syntactically valid action continuations; отказ склонного к отказам generator
может искусственно занизить targets. Новая разметка требует отдельного разрешения.

Важное исправление прежнего обсуждения: hard terminal Y для каждого префикса
не является математически неверным forecasting автоматически. BCE в ожидании
оценивает conditional frequency при репрезентативной выборке продолжений.
Но это шумный one-sample target под распределением нашего корпуса и не метка
«нарушение уже видно». Rollouts меняют распределение/дисперсию supervision.

## Ограничения опубликованных результатов и наша оценка

- CC++ — CBRN jailbreaks, в том числе каскад, не ToolSafe/ATBench; production
  refusal rate не равно benchmark FPR. Авторы отмечают ограничение оценки,
  считающей любой trigger до конца успешной защитой, даже если он поздний.
- SingProbe — response safety/hallucination, не доказательство качества на
  permissions/provenance tool-use. Его onset benchmark преимущественно
  sentence-prefix annotated, не intrinsic token gold.
- StreamGuard response_loc: 813 unsafe-only, precision=100% конструктивно;
  в 91% unsafe onset в первой sentence, OnTime измеряется до конца sentence.
  Высокий F1 там не доказывает малый FPR или точный момент детекции на наших traces.
- На action tasks нельзя автоматически считать все prefixes safe-action безопасными
  для отдельной задачи onset detection: поздняя коррекция может менять смысл.
- Для streaming диагностики нужны false alarm до различающего фрагмента,
  задержка после него и recall к концу действия при фиксированном action-level FPR.
  Action AP/PR-AUC/ROC-AUC и P/R/F1/Acc/FP/FN остаются; boundary оценка требует
  независимых prefix labels, которых нынешний корпус не даёт.

Приоритет: (1) адаптивный weighted token loss, (2) отдельная проверка нескольких
слоёв, (3) при разрешённом бюджете prefix/value distillation. Не запускать по
этому обзору обучение или разметку автоматически.
