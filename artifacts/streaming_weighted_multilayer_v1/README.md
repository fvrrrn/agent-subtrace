# Завершено: adaptive weighting × три слоя × наши кривые

Все шесть условий завершили **две эпохи**, везде выбрана epoch 2 по dev-select
AP (tie: BCE). Одинаковые **301 диалог / 886 действий / 147 unsafe**, один seed.
Pipeline занял **23 мин 34 с** на GPU 3 и завершился до исходного training-дедлайна
1792345185; общий дедлайн 1792348785 не переносился. Статус: `complete`.

## Главный результат

Adaptive weighting улучшил AP last-layer во всех трёх кривых, сильнее всего
для broad: **.47718 → .51855**. Это описательное dev-selected сравнение с
сохранёнными unweighted v3 контролями, не независимое подтверждение улучшения.
Трёхслойные признаки **не дали общего выигрыша**: относительно weighted last-layer
AP хуже для step и broad, немного выше для linear. Низкий recall при .5 сохраняется.

| Условие | AP | Трап. PR-AUC | ROC-AUC | Event BCE | P % | R % | F1 % | Acc % | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| last weighted step | .51209 | .51184 | .79078 | .43204 | 93.75 | 10.20 | 18.40 | 84.99 | 1 | 132 |
| last weighted linear | .49534 | .49432 | .78434 | .43908 | 78.26 | 12.24 | 21.18 | 84.88 | 5 | 129 |
| last weighted broad | **.51855** | **.51778** | **.79743** | **.40815** | 88.89 | 10.88 | 19.39 | 84.99 | 2 | 131 |
| three weighted step | .48744 | .48643 | .77545 | .47824 | 92.86 | 8.84 | 16.15 | 84.76 | 1 | 134 |
| three weighted linear | .50445 | .50350 | .78314 | .51738 | 100.00 | 8.84 | 16.25 | 84.88 | 0 | 134 |
| three weighted broad | .50672 | .50597 | .77440 | .43887 | 92.86 | 8.84 | 16.15 | 84.76 | 1 | 134 |
| unweighted v3 step | .50798 | .50711 | .79466 | .41955 | 87.50 | 9.52 | 17.18 | 84.76 | 2 | 133 |
| unweighted v3 linear | .48286 | .48166 | .77667 | .42856 | 76.19 | 10.88 | 19.05 | 84.65 | 5 | 131 |
| unweighted v3 broad | .47718 | .47602 | .75854 | .46414 | 84.21 | 10.88 | 19.28 | 84.88 | 3 | 131 |

Все бинарные метрики при заранее заданном пороге **.5**, без fitting.
AP delta weighted−unweighted: step +.00411, linear +.01247, broad +.04137.
AP delta three−last внутри weighting: step −.02465, linear +.00911, broad −.01183.
В step/linear рост AP от weighting не сопровождается улучшением BCE.

У лучшего last broad медианы scores safe/unsafe = .02033/.14034; recall при
FPR≤1% на этой же dev-выборке = 19.73%. Score не заявляется калиброванной
вероятностью вреда. Выбор лучшей из шести конфигураций тоже использует dev-select.

## Что именно реализовано

- Frozen `Qwen/Qwen3.5-4B@851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, исходные
  train-small 2009 и dev-select 301, native whole-dialog IDs, без обрезки.
- SingProbe-inspired weighted token BCE + .1 endpoint event BCE одновременно.
  Официальный repository revision `10edcb813482db08734e755f354390bf4d68a6b8`
  проверен: shipped Ling-config задаёт **T=1, tau=.35, detach=true**; это отличается
  от defaults класса. Beta — population std всех размеченных action-токенов текущего
  диалога / tau, clipped [0,1]; alpha softmax внутри action, detached целиком.
- Step, linear и broad ровно по [протоколу](PROTOCOL.md). Ни gold, ни будущие
  события/длина действия не добавляются во вход backbone или Mamba.
- Three-layer: concat `hidden_states[11,22,32]` → LN7680 → Linear7680→256 →
  прежние два residual Mamba → head. Первые два выхода до final norm, последний
  после native final norm. Runtime hooks подтвердили индексы.
- Mamba/readout initialization побитово одинаковая во всех шести условиях;
  three-layer input projection шире и инициализируется отдельно. Это ограничение
  сравнения и дополнительная ёмкость, а не parameter-matched layer-only ablation.
- Это адаптация идей к agent actions/Mamba/event loss, **не репродукция SingProbe**.

## Проверки и сохранность

- Last-layer CPU admission: все 2310 cache identities/IDs/shapes/hashes прошли;
  побитовые hashes совпали с v3, старый кеш использовался read-only.
- Новый cache: 2310 диалогов, 3 288 009 tokens, 292.01 с. Deep-компонента всех
  2310 tensors **побитово совпала** со старым last-layer cache. Frozen-weight hash
  до/после совпал. Backbone causal error = 0, hooks для hidden indices совпали.
- Config JSON/Torch round-trip; шесть CUDA gates прошли: causal/reload error=0,
  оба loss дают ненулевые градиенты обоим Mamba, readout и projection; loss mask
  и backward на реальном самом длинном входе проверены.
- Финальный CPU-аудит проверил **10 632 raw endpoints** (6 × 2 × 886), ошибка=0,
  3612 token tensors, полные IDs/spans, checkpoint hashes и неизменность source.
  `artifact_audit.json` содержит hashes всех validation logits/events/metrics.
- Сохранены checkpoint/optimizer/RNG каждой эпохи. Полные dev-select raw token
  logits и event logits сохранены для обеих эпох всех шести условий; training
  token logits не архивировались (как и в исходном trainer; см. протокол).
- Per-source и action-length срезы: `*/fit/dev_select_epoch*/fixed_threshold_slices.json`;
  class score quantiles: соответствующие `metrics.json` и общий `summary.json`.

## Пути и процессы

Серверный корень `/var/local/danefedov/experiments/agent-safety`:
- run: `runs/streaming-weighted-multilayer-v1/`;
- immutable source: `runs/streaming-weighted-multilayer-v1-launch/source/`;
- manifest SHA256: `9632419b948107c7cda90b4c072c406d4254a80b4b78ab659e8e62d7386af5ef`;
- raw tensors: `<run>/<arm>/fit/dev_select_epoch{1,2}/tokens/*.pt`;
- новый feature cache: `<run>/cache/features/*.pt`.

Launcher PID 3705095 завершён; streaming shell
`sh_14feb35c4001IOuGfAXiGD3PGS` завершён успешно, в нём реальный tqdm всех стадий.
Artifact-copy watcher `sh_14febfe0d001FGPeqAWTmQu2Yd` завершён успешно;
агрегаты, отчёты и head checkpoints скопированы локально. Feature cache,
token tensors и optimizer/RNG states остаются на сервере.
Старые запуски не изменялись, новые benchmarks/confirm/oldval не оценивались.
Новых условий, калибровки или дальнейших обучений этим результатом не назначено.
