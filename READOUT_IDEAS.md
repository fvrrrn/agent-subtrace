# Event readout без жёсткой формы кривой: идеи из смежных областей

Статус: обзор и предложения, **не реализовано/не измерено**. Новых model calls,
обучения, калибровки или readout-сетки этот документ не назначает.

## Проблема постановки

У нас есть action-label, но нет token-label. Контекстуализированное Mamba
состояние h_i уже содержит прошлые токены/события: токен не независимое наблюдение.
Риск prefix, локальное свидетельство и probability целого события — разные
выходы. Если p_i уже оценивает один и тот же Y по возрастающим prefixes,
усреднение/перемножение p_i не является корректным объединением независимых
свидетельств. Если p_i — локальный activation, нужен обученный MIL readout,
а не интерпретация его как калиброванной prefix-risk probability.

Сходство Dice с заданным g отвечает «похожа ли форма», но не свободно решает
«опасно ли событие». Текущий линейный target, как и другой фиксированный hump,
не становится token-gold от гладкости. Возможные совместные свидетельства
могут находиться в разных местах; hi может представлять их композицию.

## Проверенные источники

1. Wang et al., *A Comparison of Five Multiple Instance Learning Pooling
   Functions for Sound Event Detection with Weak Labeling*, ICASSP2019:
   https://arxiv.org/html/1810.09050v3
   Сравнивают event tagging и frame localization; показывают, что лучшая
   event-classification не гарантирует лучшую локализацию. Linear softmax
   сбалансировал localization errors в их опытах; это не доказательство
   превосходства на наших текстах. Есть ограничения learned-attention pooling.
2. McFee et al., *Adaptive pooling operators for weakly labeled sound event
   detection*,2018: https://arxiv.org/html/1804.10070v2
   Learnable auto-pool учитывает разные длительности; фиксированная мягкость
   и длина bag влияют на pooling. Не решает автоматически domain shift.
3. Ilse, Tomczak, Welling, *Attention-based Deep Multiple Instance Learning*,
   ICML2018: https://proceedings.mlr.press/v80/ilse18a.html
   Event probability по contextual feature pooling, а не обязательно pooling
   уже сжатых scalar probabilities. Attention contribution не token-safety gold.
4. Rajpurkar, Jia, Liang, *Know What You Don't Know: Unanswerable Questions for
   SQuAD*, ACL2018: https://arxiv.org/abs/1806.03822
   Нужно отделять «где возможный ответ» от «поддерживается ли вообще ответ».
   Аналогия: опасно выглядящий span ещё не означает unsafe действие.
   NULL/witness ниже — наша адаптация идеи, не дословная архитектура статьи.
5. Chan, Walther, *Detection with the scan and the average likelihood ratio*,
   Statistica Sinica2013: https://arxiv.org/abs/1107.4344
   Неизвестные положение/протяжённость сигнала — nuisance parameters;
   max-scan и average likelihood ratio имеют разные свойства по масштабам.
   Теоремы для Gaussian noise нельзя перенести на коррелированные Mamba logits.

## Кандидат A: простой shape-free MIL baseline

Linear softmax / Lehmer pooling:

`s = sum(p_i²)/sum(p_i)`, нулевой вектор отображается в0.

Для p>=0 он в[0,1]. На binary token-activations любая непустая группа единиц
даёт1, все нули дают0. Нулевой padding не меняет s. Нет prior по позиции,
заданного k или ширины. Это **новая семантика идеала** (binary activations),
не обещание1 на произвольной прежней soft-кривой.

Градиент `ds/dp_i=(2p_i-s)/sum(p)`. При BCE positive-event boosts большие
activations и может подавлять малые, в отличие от среднего, толкающего все
токены вверх. Но это не coordinatewise-monotone detector: рост малого p_i
может уменьшать s. Ненулевой фон на длинном сообщении всё ещё разбавляет
короткий сигнал; один ложный пик тоже остаётся проблемой. Не выдавать его
за окончательный оптимальный readout. Наши текущие probabilities обучались
на curve-loss, поэтому post-hoc pooling и training-through-pooling различаются.

## Кандидат B: наличие нарушения против NULL, неизвестная позиция свидетельства

Из Mamba features вычислять raw evidence-logits r_i и отдельный
`r_null = f(h_end)` для гипотезы «unsafe current action не подтверждается».
NULL видит доступную историю и полностью прочитанное candidate action,
но не его tool-result/будущие события. Позиции — только current action span.

Ввести фиксированные location weights pi_i>=0,sum pi=1, по умолчанию1/N:

```
A = sum_i pi_i * exp(r_i)
s = A / (A + exp(r_null))
z_event = logsumexp(r_i + log(pi_i)) - r_null
L_event = BCEWithLogits(z_event, y)
```

Это дискриминативная categorical mixture «NULL или один выбранный witness».
Выбранный latent witness не означает, что в тексте только один опасный токен:
масса может распределяться между несколькими представителями составной угрозы.
`s in [0,1]`; если NULL доминирует,s→0; если evidence доминирует,s→1.
Форма evidence не фиксирована. Суммирование — по альтернативным местам
свидетельства, **не noisy-OR независимых token risk probabilities**.

`pi=1/N` устраняет чисто детерминированный рост s от N при одинаковых r_i,
но не гарантирует robustness к новым шумным пикам/длинам/domain shift.
Большой конечный сигнал всё равно конкурирует с multiplicity/NULL: это
нужно измерять и калибровать на отдельном dev, не на benchmark labels.

Удобная отдельная условная localization-кривая:
`alpha_i = pi_i exp(r_i)/A`, sum alpha=1. Само наличие опасности — s,
расположение поддерживающих activations — alpha. Soft curve prior можно
обсуждать для alpha как слабое ограничение, а не заставлять им event-probability.
Но alpha зависит от нормировки по полному сообщению, **не causal online
probability в момент i**, и не доказанная интерпретация/разметка токенов.

Для фиксированных pi:
`dL/dr_i = (s-y)*alpha_i`, `dL/dr_null = y-s`.
Нет дополнительного множителя sigmoid(r_i)*(1-sigmoid(r_i)) перед pooling;
это отличается от вторичного event-loss через усреднённые token sigmoids.

Ограничение идентифицируемости: message-label сам по себе не заставит alpha
находить настоящие causal witness spans. Модель может решать y через NULL
или ложные корреляции. Нужны отдельные контрфактические/локализационные проверки;
learned attention не объяснение по определению. Даже если bag-probability
обучена BCE, calibration/перенос не гарантированы.

## Кандидат C: multiscale signal detection

На токеновом trace искать превышение безопасного background на интервалах
разной позиции и длины. Локальное evidence сравнивать не с одним hump,
а с safe null distribution соответствующей длины/масштаба; затем учитывать
число просмотренных интервалов. Priors задают ожидаемую длительность/частоту,
не точную форму каждого positive trace. Это идея из scan/ALR detection.

Без background calibration длинные сообщения получают больше ложных пиков.
Коррелированные h_i требуют covariance-aware или empirical null; независимый
Gaussian z-test непригоден по умолчанию. p-value/1-p-value не probability unsafe.
Перевод detection statistic в calibrated event probability требует отдельного
обучения/проверки. Полный quadratic scan не нужен без согласования масштаба;
возможны multiscale intervals, но ничего не реализовано.

## Что не стоит делать

- Не умножать `(1-p_i)` от нынешних prefix-risk scores как независимые события.
- Не переносить свойства audio/cancer detectors на safety без проверки.
- Не обещать истинную token-localization только от event-label BCE.
- Не смешивать предварительную raw activation curve с calibrated event-probability.
- Не обучать/подбирать новые головы, pooling или prior weights автоматически.

Предварительный приоритет: A как минимальный контроль, B как более осмысленная
вероятностная постановка без фиксированной формы; C — отдельное статистическое
направление, не обязательный новый слой/запуск. Все варианты пока гипотезы.
