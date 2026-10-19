# Широкие позиционные профили с ненулевыми краями

Графики: [prior_profiles.png](prior_profiles.png) — candidate peak-scaled
shapes слева и area-normalized densities справа; [manual_token_examples.png](manual_token_examples.png)
— все три ручных примера с фактическими Qwen-позициями. SVG также сохранены.
Построены `plot_token_position_priors.py` без модели/обучения; matplotlib
запущен в отдельном temporary uv environment, проектные зависимости не менялись.

Рабочая гипотеза для проверки, не установленный результат: **широкий гладкий
горб/плато над ненулевым фоном, с допустимой асимметрией и нефиксированным
положением максимума**. Начальный уровень зависит от видимой истории; конец
не обязан быть максимальным или обнуляться. Первый простой prior-кандидат —
смесь uniform/left-Beta/middle-Beta, дающая высокий старт и широкий пик:
`g(u)=(.9+1.2u-1.5u²)/1.14`; g0≈.789,g.4=1,g1≈.526.
Это не предписание выдавать score.789 на первом токене любого unsafe
сообщения и не доказательство универсального пикаu.4. Контрпример3 показывает
опасные свидетельства в конце; для семантической локализации нужна зависимость
от содержания. Высокий фон — предположение об uncertainty/provenance prior,
а не вручную обнаруженная опасность нейтральных слов. Пиковая нормировка не
создаёт калиброванную вероятность. Weighted-readout по этомуprior, новая
кривая обучения и дополнительные эксперименты не запускались.

CPU-only исследование трёх **сконструированных** английских предложений,
не benchmark/обучение и не эмпирическая оценка распределения. Фактическая
токенизация закреплённым Qwen3.5-4B-Base, без chat/control tokens:
`token_positions.json` содержит все75IDs/offsets, субъективные evidence-spans,
иллюстративные веса, кандидатные формы и отдельно нормированные position-masses.
Нулевой ручной evidence-weight не является safe-токеновой меткой.

## Два разных объекта

1. `P(Y=unsafe | prefix)` — риск действия с учётом уже прочитанного. Может
   быть высоким в начале из-за истории и сохраняться высоким после опасного
   фрагмента. Не обязан суммироваться по токенам в1.
2. `q_i=P(I=i | whole message,Y=unsafe)` — условное распределение позиции
   токена-свидетельства. Суммируется в1, но использует определение того, что
   именно считать свидетельством. Нет единственного объективного опасного
   токена: передача + restricted object + recipient + permissions образуют
   совместное свидетельство, часто в нескольких несмежных участках.

Current linear0→1 мог бы изображать CDF Uniform, но не его плотность.
Плотность с пиком в середине и высоким концом — другой training target.
Не называть soft token curve автоматически вероятностью вреда.

## Ручной разбор

- Disclosure,N21: email3(u.143); confidential payroll file5–7(.238–.333);
  external recipient10–11(.476–.524); without authorization12–13(.571–.619).
  Несколько связанных сгустков, а не один естественный опасный токен.
- Transfer,N26: amount4–10(.154–.385), company account13–14(.500–.538),
  my personal wallet16–18(.615–.692), without approval19–20(.731–.769).
  Свидетельства смещены вправо. `$12,000` состоит из7токенов `$`,`1`,`2`,`,`,`0`,`0`,`0`.
- Late disclosure,N28: private API key21–23(.750–.821), external recipient
  26–27(.929–.964); опасное отправление19(.679). Контрпример универсальному
  центру: нейтральный префикс сдвигает позицию, не меняя сути утечки.

Иллюстративные ручные веса дают центры0.433/0.563/0.847; это не измеренные
вероятности или верифицированная разметка. Moment-matched Beta вJSON лишь
сжимает выбранные субъективные веса; не доказывает нужную форму, хвосты или
одномодальность. По3придуманным примерам нельзя оценить population prior.

## Кандидаты под уточнение пользователя

Ниже `g` нормирована **по пику**, не по сумме и не является измеренным риском.
Все3формы имеют максимум приu0.4, высокий левый край и ненулевой правый:

| Форма | g(0) | g(.25) | g(.4) | g(.75) | g(1) |
|---|---:|---:|---:|---:|---:|
| Усечённая Normal(mu.4,sigma.5) | .7261 | .9560 | 1 | .7827 | .4868 |
| Усечённый Student-t(nu3,mu.4,scale.6) | .7586 | .9596 | 1 | .8066 | .5625 |
| .60Uniform+.15Beta(1,2)+.25Beta(2,2), peak-normalized | .7895 | .9704 | 1 | .8388 | .5263 |

`g_normal=exp(-.5*((u-.4)/.5)^2)`.
`g_student=(1+((u-.4)/.6)^2/3)^(-2)`.
Beta-mixture density `f(u)=.9+1.2u-1.5u²`, integrates to1;
its peak is1.14, so `g=f/1.14`. Uniform provides nonzero endpoints,
Beta(1,2) raises the beginning, Beta(2,2) adds a middle hump.
Student parameters nu/scale control shoulders; decreasing nu alone at fixed
scale does not guarantee higher values at every finite normalized position.

Normal already satisfies the intended broad-tail profile; a formally
heavy-tailed real-line family is not necessary on the bounded interval.
For a genuine discrete position law: `q_i=g(i/N)/sum_j g(j/N)`.
Using q directly as BCE risk targets would force magnitudes toward1/N and
introduce length dependence. Peak-normalized g avoids that particular scaling
but must honestly be called an artificial shape prior, not a probability mass.

## Practical caveats

- The peak need not be at.4 or.5: shift mu or use a mixture when evidence is
  late/multimodal. Added neutral text can move normalized positions arbitrarily.
- High starting risk is justified by prior visible context, not by the future
  unsafe label alone. Identical causal prefixes with divergent endings cannot
  reliably have different early risk. Future-dependent targets are not an
  input leak, but may be inherently unpredictable and encourage false alarms.
- Native streaming also does not know final N until message end. These are
  offline full-length training targets, not proof of online causal calibration.
- Proposed Student-t and Beta-mixture are candidates only. No curve/model
  sweep, synthetic corpus generation, training, threshold change or GPU
  inference was launched. Existing annotation was untouched.
