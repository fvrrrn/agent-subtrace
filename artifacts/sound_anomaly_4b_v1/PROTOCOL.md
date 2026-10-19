# Frozen Qwen 4B: B/C, две эпохи

Источник формул: `../small_backbone_loss_v2/LOSS_RESEARCH.md`, разделы B/C.
Контроль A: `runs/sound-anomaly-4b-control-v1/candidates/control_a`.
Сначала ожидание PID 2991678 через pidfd; затем только GPU 2, последовательно
B gate → B две эпохи → C gate → C две эпохи. Повторов при ошибке нет.

B: отдельный forecast BCE с целью Y, evidence linear logits, auto-pool с
alpha=softplus(a), начальным alpha=1; event logit через разность logsumexp.
C: та же forecast-голова; causal окна 8/32/128 внутри текущего действия,
concat(mean(h_l..h_i), h_i-h_l, h_i), общий Linear768→32/ReLU,
покомпонентный max внутри каждого масштаба, concat с h_N, Linear352→1.
Используются нормализованные состояния после двух residual Mamba, как перед
исходной token-головой. Никакого добавленного attention или LoRA.

Loss: mean_action(mean_token BCE + 0.1 event BCE), один целый диалог за forward,
накопление четырёх диалогов. AdamW lr=3e-4, wd=.01, clip=1, seed=20261017.
Общая инициализация проверяется против SHA-256 gate контроля A до обучения.
Frozen cache читается с исходными строгими identity/IDs checks.
Исходные train_small=2009 и dev_select=301; StepGuard не добавляется.
Ни confirm, ни oldval не оцениваются и не используются для CUDA gates.

Обязательные gates: точная причинность и reload, раздельные градиенты обоих
loss к обоим блокам Mamba и соответствующим головам; нулевая прямая поддержка
loss вне action spans; backward реального максимального train-входа 15322.

После каждой эпохи сохраняются все 301 dev-диалог, IDs, spans, raw forecast и
event logits, метрики при 0.5, AP, трапецеидальная PR-AUC, ROC-AUC, квантили и
recall при фиксированных FPR. Legacy поле `null_logits` содержит состояния
Mamba, а не NULL logits; отдельный tensor_schema.json фиксирует это явно.
Checkpoint выбирается по AP, при равенстве по меньшему event BCE.
Пороги не подбираются. Optimizer/RNG сохраняются после каждой эпохи.

Абсолютный дедлайн обучения: 1792345185.243998; общий лимит волны B/C —
2700 секунд после завершения A. Проверка бюджета перед каждым forward,
в том числе validation; scheduler ограничивает время каждого дочернего процесса.
Исходный глобальный дедлайн 1792348785.243998 сохраняет час резерва.

A→B меняет несколько факторов и не является чистой loss-абляцией.
B/C различаются event-readout и числом параметров. Один seed и dev selection
не являются независимым подтверждением. Новый A отличается от прежнего
endpoint запуска; причина невоспроизводимости пока не установлена.

Тяжёлые token tensors остаются на сервере. Локально копируются отчёты,
протоколы, selection и логи без тяжёлых tensors/checkpoints.
