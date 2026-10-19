# Идея: attention к прошлым событиям перед/параллельно Mamba

Автор идеи — пользователь. **Записано для обсуждения, не назначено отдельным
обучением и не включено молча в сравнение readout.** Qwen остаётся замороженным;
его полный причинный token-feature cache можно переиспользовать.

## Две архитектурные гипотезы

1. Последовательная:
   `Qwen features → projection → residual causal attention → residual Mamba stack → readout`.
   Attention извлекает связи с доступными прошлыми токенами, Mamba обновляет
   последовательную память уже по этим дополненным представлениям.
2. Параллельная:
   `Qwen features → shared projection → [Mamba branch || causal-attention branch]
    → concat per token → projection/readout`.
   Сохраняет прямой путь Mamba и отдельный путь content retrieval. Конкатенация
   сама по себе не гарантирует, что модель использует обе ветви.

Один диалог по-прежнему один forward; state между событиями не сбрасывается.
Оба варианта увеличивают capacity/стоимость относительно Mamba-only baseline;
attention квадратичен по длине даже при memory-efficient SDPA. Равенство скорости
или параметров нельзя предполагать.

## Как обучать

- Основной сигнал — hard current-action event-BCE через согласованный readout,
  только на размеченных действиях. Он обучает attention и Mamba совместно;
  отдельная разметка attention weights не обязательна, но и правильность
  внимания к реальным зависимостям этим не гарантирована.
- Контрфактические пары с одинаковым current action, разной видимой историей
  и различными проверенными labels полезны для проверки/обучения использования
  истории. Такие пары нужно восстановить и проверить, не выдумывать из общих IDs.
- Не учить attention просто «смотреть на всё unsafe»: прошлое опасное событие
  может быть нерелевантно, а безопасное user permission update или источник
  restricted-данных — критичен для текущего решения.

## Если нужен явный risk bias

Возможная форма logits: `attention_logit(t,j)=q_t·k_j/sqrt(d)+beta*log(eps+r_j)`
для допустимых j≤t. `r_j` — **предсказание** риска прошлого, не gold-label.
Ненулевой floor оставляет доступ к низкоскорным, но важным разрешениям/источникам.
Beta/floor/stop-gradient не зафиксированы и не подбирались.

Риск-head может получать auxiliary BCE на известных прошлых action endpoints.
Gold используется только в loss, не как входной attention bias. Для неизвестных
token labels safe/unsafe targets не выдумываются. Можно сначала обойтись без
явного risk bias и сравнить его отдельно после проверки базового attention.

Критично избежать цикла: в схеме attention→Mamba нельзя одновременно брать
`r_j` из результата этой же Mamba и подавать обратно в её upstream attention
одного параллельного forward. Нужен отдельный upstream predictor либо явно
последовательная схема; это разные протоколы, не «бесплатный один проход».

## Аудит перед любым запуском

- causal suffix/prefix check, роли и action boundaries, отсутствие future tool-result;
- отсутствие gold teacher forcing в risk-conditioned attention;
- ненулевые градиенты attention/Mamba/fusion, frozen-Qwen hash, reload;
- known-length admission и память на самом длинном диалоге без обрезки;
- сравнение с Mamba-only на одинаковых данных/checkpoint selection;
- проверка контрфактического влияния истории, не только attention heatmaps.

Эти пункты — описание будущей проверки, не новые тесты или GPU-эксперименты.
