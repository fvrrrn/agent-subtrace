# Аналоги Jev: проверенные первичные источники

Проверено 13 октября 2026. Здесь перечислены **decision/System One модели**,
а не только обычные zero-shot/NLI классификаторы из предыдущего сравнения.
Размеры/возможности — по описаниям авторов; vendor benchmarks не считаются
результатами нашего эксперимента. Фактические версии и revisions локальных
прогонов фиксируются отдельно.

| Семейство | Что возвращает | Открытые веса / локальный запуск | Основа / размер |
|---|---|---|---|
| Kev | choice, noul, score; Jev-compatible System One | Да; Apple MLX и CUDA | Qwen + pointer head; 0.8B, 4B, 9B, 27B |
| Von | choice, noul, score; System One | Да; MPS/CPU | ModernBERT-large + option-marker head, около 395M |
| OpenDecider | choice, noul, score; System One | Да; nano на MPS, small также MLX | nano: Ettin ~400M; small: Qwen3 4B; есть более крупные |
| GLiNER2.5-Decide | schema-defined classification/typed decisions, включая multi-label | Да; gliner2 | Компактный encoder, заявлено 340M |
| Strands Decider | choice, noul, score; pointer readout | Да; локальный CPU/GPU | Qwen3.5-2B + LoRA и pointer head |
| Clef / Clef-flash | typed questions, вероятности вариантов, System One | Да; также Workers AI | Qwen-based 27B / 9B |
| Microsoft-Decision-1 | choice, binary, ordinal/rubric decisions, System One endpoint | Hosted Microsoft Foundry | Microsoft post-training; не выбран для локального прогона |

## Kev

- Официальный репозиторий: https://github.com/jaredpalmer/kev
- Модели: https://huggingface.co/jaredpalmer/kev-0.8b и
  https://huggingface.co/jaredpalmer/kev-4b
- Request: `state` + `questions` с `type`, `instructions`, `criteria`;
  response содержит полное choice distribution. Нет генерации объяснений.
- Малые текущие модели — LoRA + pointer head поверх Qwen3.5-Base. Decoder
  backbone не означает, что runtime генерирует ответ токен за токеном.
- Для Mac авторы документируют MLX; 4B измерен на 32-ГБ Mac.
- Важная деталь установки: PyPI `kev` — посторонний ORM. Decision-модель
  ставится из официального исходного репозитория, с фиксацией commit.
- Заявленные авторами temperature settings оставляем как опубликованы;
  калибровку на нашей задаче это не гарантирует.

## Von

- Официальный репозиторий: https://github.com/wfzyx/von
- Модель: https://huggingface.co/wfzyx/von
- Пакет: `von-sdk`. В публичных материалах одновременно встречаются версии
  model/runtime 1.1/1.2/1.3; сравнивать нужно pinned code и weights, не имя latest.
- Более новые runtime имеют дополнительные механизмы chains. В исходном
  алгоритме эксперимента дополнительное recursive reasoning не включается.

## OpenDecider

- Официальный репозиторий: https://github.com/manjunathshiva/opendecider
- Документация: https://manjunathshiva.github.io/opendecider/
- Nano: https://huggingface.co/manjunathshiva/opendecider-nano
- Nano — bidirectional Ettin encoder с option-marker scoring, native choices.
  Контекст по документации 2048; нужно проверить, что конкретные входы помещаются,
  а не молча обрезать их до этого размера.
- Small — next-token probability readout на Qwen3-4B + LoRA, опубликованы
  MLX/GGUF варианты. В текущую малую проверку выбран nano.

## GLiNER2.5-Decide

- Официальный анонс:
  https://fastino.ai/blog/gliner-2-5-decide-open-weight-decision-model
- Веса: https://huggingface.co/fastino/GLiNER2.5-Decide
- Это отдельный decision-checkpoint; его не следует путать с GLiClass из
  предыдущего эксперимента или обычной entity extraction моделью GLiNER.
- Native schema/API может отличаться от Jev; перенос instructions и
  descriptions, а также преобразование scores должны быть раскрыты в отчёте.
- Встроенные constraints по ответам не используются для кодирования gold
  нашей задачи. Исходный алгоритм остаётся LINK+SAFETY с фиксированными порогами.

## Дополнительные найденные варианты

- Strands Decider 2B:
  https://strandsagents.com/blog/introducing-strands-decider/
  https://github.com/strands-labs/strands-decider
  https://huggingface.co/StrandsAgents/strands-decider-2B-qwen3.5-v1-2610
  По официальному описанию LM head заменена pointer head; нет генерации текста.
  Авторы приводят измерения на M3 MacBook. Найден и проверен по первичным
  источникам; в текущем наборе запусков не измеряется.
- Cloudflare Clef/Clef-flash:
  https://blog.cloudflare.com/clef-decision-models/
  https://huggingface.co/Cloudflare/clef
  https://developers.cloudflare.com/changelog/post/2026-10-01-clef-workers-ai/
  Обновление hosted-лимитов:
  https://blog.cloudflare.com/clef-faster-cheaper-multimodal/
  Более тяжёлые модели, в этот локальный набор не включены. Hosted и self-hosted
  ограничения контекста различаются; старые публикации с 64k flash устарели.
- Microsoft-Decision-1:
  https://learn.microsoft.com/en-us/azure/foundry/foundry-models/how-to/use-foundry-models-microsoft-decision
  https://commandline.microsoft.com/microsoft-decision-1-model-foundry/
  Требуется Foundry deployment и аутентификация; не запускался.
- Также найден `open-alternative-jev`:
  https://github.com/ikermoel/open-alternative-jev. Это библиотека readout
  вероятностей из обычных open LLM, а не отдельный обученный Jev-checkpoint.
  Её не смешиваем с native decision-моделями в одном списке результатов.

Jev самого TypeSafe закрыт и запускается через API. Проверенные официальные
документы: https://docs.typesafe.ai/models. Ключи не искались, платные внешние
вызовы не выполнялись.
