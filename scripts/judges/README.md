# Судьи для разметки

`annotate.py` — один клиент для любого эндпоинта, который умеет
`POST {base_url}/chat/completions`. Подходит OpenRouter, локальный vLLM, Ollama
и вендорские API. Логика разметки (промпт, схема ответа, видимое состояние)
та же, что у прогона, которым размечен релиз, поэтому два файла аннотаций
сравниваются поле в поле.

## Установка

```bash
uv sync          # создаёт .venv и uv.lock из pyproject.toml в корне
```

Дальше скрипты запускаются через `uv run`, руками venv активировать не нужно.

## Ключ

Ключ и адрес берутся из окружения, `.env` подхватывается из текущей папки
вверх по дереву:

```
JUDGE_API_KEY=sk-or-v1-...
JUDGE_BASE_URL=https://openrouter.ai/api/v1
```

Порядок поиска: `JUDGE_*`, затем `OPENAI_*`, затем `OPENROUTER_*`. Если ключа
нет, отправляется заглушка `EMPTY` — так работают vLLM и Ollama.

`base_url` — это база **до** `/chat/completions`, то есть для vLLM
`http://127.0.0.1:18000/v1`, а не `http://127.0.0.1:18000`.

## Запуск

```bash
# OpenRouter
uv run python scripts/judges/annotate.py \
  --input data/candidates.jsonl \
  --output data/annotations_qwen3-235b.jsonl \
  --model qwen/qwen3-235b-a22b-2507 \
  --system-file scripts/dataset/flash_next/prompt.txt

# локальный vLLM с Qwen3.8-Flash-Next
uv run python scripts/judges/annotate.py \
  --input data/candidates.jsonl \
  --output data/annotations_flash_next.jsonl \
  --model qwen3.8-flash-next \
  --base-url http://127.0.0.1:18000/v1 \
  --extra-body '{"chat_template_kwargs": {"enable_thinking": false}}' \
  --system-file scripts/dataset/flash_next/prompt.txt
```

`--system-file` обязателен: скрипт не владеет промптом. Канонический промпт
релиза — `scripts/dataset/flash_next/prompt.txt` (sha256 `5417c7fd…afd6`), его
хеш зафиксирован в `dataset/manifest.json`.

`--dry-run` печатает, сколько строк ждёт разметки, и ничего не отправляет.

## Формат ответа

По умолчанию (`--json-mode auto`) запрос идёт с `json_schema`, при отказе
эндпоинта — с `json_object`, при повторном отказе — обычным текстом с
вытаскиванием JSON. Фактически сработавший режим пишется в поле
`response_format`. Явно: `--json-mode schema|object|none`.

Ответ проверяется на обязательные поля, допустимый `verdict` и известные
facet'ы. Невалидный ответ перезапрашивается `--parse-retries` раз, после чего
строка пишется со `status: error` и попадает в следующий прогон.

## Заметки

- `--limit 200` — быстрый пилот: сначала прогоняем двести строк, смотрим
  распределение `verdict` и долю `error`, потом запускаем остальное.
- Reasoning-модели тратят бюджет на размышление: при `--max-tokens 768`
  `content` может прийти пустым. Для них поднимать `--max-tokens` до 2000+.
- Часть провайдеров не принимает `seed` или `json_schema`; отказ виден в поле
  `error` строки, а не в тишине.

## Оценка

`eval.py` — тот же клиент, но по gold-набору: считает macro F1 по трём
классам, binary accuracy, unsafe recall/precision, coverage и стоимость.
Пишет файл прогона, который читают два следующих скрипта.

```bash
uv run python scripts/judges/eval.py \
  --gold data/gold_cases.jsonl --split dev \
  --system-file scripts/dataset/flash_next/prompt.txt \
  --model qwen/qwen3-235b-a22b-2507 \
  --output data/eval_qwen3-235b_dev.jsonl
```

`aggregate_runs.py <dir> [tag ...]` — сводка по нескольким прогонам и правило
отбора RRSI (описано в `docs/DATA_PROMPT_TUNING.md`).

`diff_runs.py A.jsonl B.jsonl [--only-changes]` — какие кейсы поменяли
предсказание между двумя прогонами.

## Выход

Одна строка JSON на кандидата. Поля, на которые опирается
`scripts/dataset/build_dataset.py`:

| Поле | Смысл |
|---|---|
| `key` | ключ кандидата |
| `status` | `ok` / `too_long` / `invalid_input` / `error` |
| `parsed` | разметка: `verdict`, `facets`, `severity`, `reversibility`, `actor`, `authorization`, `evidence_sufficiency`, `evidence`, `reason`, `confidence` |
| `model`, `served_model` | запрошенная и фактически ответившая модель (OpenRouter роутит по провайдерам) |
| `prompt_sha256` | sha256 от `system + "\n" + user` |
| `input_sha256_prompt` | sha256 от `json.dumps(state)` — попадает в датасет как `input_sha256` |
| `sources`, `group`, `label` | провенанс и метка источника |
| `usage`, `latency_ms`, `response_format` | стоимость и диагностика |

Прогон возобновляемый: строки со `status: ok` и `too_long` пропускаются,
`error` перезапрашиваются. Дописывание идёт после каждой строки, так что
падение процесса не теряет результат.

В конце печатается сводка: сколько строк обработано, по статусам, и суммарные
`prompt_tokens` / `completion_tokens` — по ним считается цена прогона.

## Про исторический прогон

Релиз размечен другим клиентом — `httpx` + `guided_json` на 4×H100, файл
`flash_next/annotate.py`. Он удалён, потому что дублировал этот; исходник
лежит в истории git на коммите `bd2e5f6f`. Условия прогона —
`docs/DATA_ANNOTATION_RUN.md`, формат записи тот же, поэтому старые аннотации
и новые сравниваются напрямую.
