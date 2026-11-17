# Судьи для разметки

`annotate.py` добавляет в датасет колонку с вердиктом судьи. На вход — parquet
(файл или папка), на выход — те же строки плюс `<model>_judge_label` со
значением `safe` / `unsafe` / `uncertain`. Остальные колонки копируются как
есть, поэтому метки разных судей лежат рядом и сравниваются по колонкам одной
таблицы.

Судья — любой эндпоинт, который умеет `POST {base_url}/chat/completions`:
OpenRouter, локальный vLLM, Ollama, вендорский API. Промпт, схема ответа и
видимое состояние те же, что у прогона, которым размечен релиз.

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

## Прокси

Если сеть пускает наружу только через прокси, задайте `HTTPS_PROXY` в `.env`:
`load_dotenv` кладёт его в окружение до создания клиента, а `httpx` внутри
`openai` читает переменную сам. `no_proxy` с `127.0.0.1,localhost` оставляет
локальный vLLM доступным напрямую.

## Запуск

```bash
# все четыре сплита из папки релиза
uv run python scripts/judges/annotate.py \
  --input dataset --output judged \
  --model qwen/qwen3-235b-a22b-2507 \
  --system-file scripts/dataset/flash_next/prompt.txt

# один файл
uv run python scripts/judges/annotate.py \
  --input dataset/dev.parquet --output judged/dev.parquet \
  --model qwen/qwen3-235b-a22b-2507 \
  --system-file scripts/dataset/flash_next/prompt.txt

# локальный vLLM
uv run python scripts/judges/annotate.py \
  --input dataset --output judged \
  --model qwen3.8-flash-next \
  --base-url http://127.0.0.1:18000/v1 \
  --extra-body '{"chat_template_kwargs": {"enable_thinking": false}}' \
  --system-file scripts/dataset/flash_next/prompt.txt
```

`--input` папка → `--output` тоже папка, имена файлов сохраняются.
`--system-file` обязателен: скрипт не владеет промптом. Канонический промпт
релиза — `scripts/dataset/flash_next/prompt.txt` (sha256 `5417c7fd…afd6`).

`--dry-run` печатает, сколько строк ждёт разметки, и ничего не отправляет.
`--limit 200` — пилот на первых двухстах строках.

## Колонка

Имя — слаг модели плюс `_judge_label`:

| `--model` | колонка |
|---|---|
| `qwen3.8-flash-next` | `qwen3_8_flash_next_judge_label` |
| `qwen/qwen3-235b-a22b-2507` | `qwen_qwen3_235b_a22b_2507_judge_label` |
| `llama/llama-3.3-70b-instruct` | `llama_llama_3_3_70b_instruct_judge_label` |

Переопределяется через `--label-column`.

Прогон возобновляемый: строки с уже заполненной колонкой пропускаются, `null`
перезапрашивается. Поэтому второй судья запускается на выходе первого и
добавляет свою колонку, не трогая чужую:

```bash
uv run python scripts/judges/annotate.py \
  --input judged --output judged \
  --model llama/llama-3.3-70b-instruct \
  --system-file scripts/dataset/flash_next/prompt.txt
```

`--input` и `--output` могут совпадать: файл читается целиком до записи.

Строка, на которой судья упал, остаётся `null` — это же и «ещё не размечали»,
и «не получилось». Обе категории попадают в следующий прогон.

## Формат ответа

По умолчанию (`--json-mode auto`) запрос идёт с `json_schema`, при отказе
эндпоинта — с `json_object`, при повторном отказе — обычным текстом с
вытаскиванием JSON.

Ответ проверяется на обязательные поля, допустимый `verdict` и известные
facet'ы. Невалидный ответ перезапрашивается `--parse-retries` раз, после чего
строка остаётся `null`.

## Сводка

Прогресс идёт в stderr; в баре после каждого ответа копится суммарный `cost`,
если провайдер его возвращает: `annotate: 45%|…| $0.1832`.

Одна строка JSON на файл:

```json
{"file": "dataset/train.parquet", "output": "judged/train.parquet",
 "column": "qwen_qwen3_235b_a22b_2507_judge_label",
 "rows": 14670, "pending": 14670, "labeled": 14670, "errors": {},
 "evidence_not_literal": <N>,
 "usage": {"prompt_tokens": <N>, "completion_tokens": <N>, "cost": <N>},
 "prompt_sha256": "5417c7fd…afd6"}
```

`errors` — счётчик по типам (`APIConnectionError`, `too_long`, `not_json`,
`invalid_input`). `evidence_not_literal` — сколько ответов цитируют фрагмент,
которого нет в отправленном тексте: диагностика качества судьи, в колонку не
попадает. `usage` — сумма по всем запросам файла, вложенные поля развёрнуты
через точку; `cost` появляется, когда провайдер его возвращает.

## Оценка

`eval.py` — отдельный сценарий, по gold-набору: macro F1 по трём классам,
binary accuracy, unsafe recall/precision, coverage и стоимость. Пишет файл
прогона, который читают два следующих скрипта.

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

## Заметки

- Reasoning-модели тратят бюджет на размышление: при `--max-tokens 768`
  `content` может прийти пустым. Для них поднимать `--max-tokens` до 2000+.
- Часть провайдеров не принимает `seed` или `json_schema`; отказ виден в
  `errors` сводки, а не в тишине.
- Судья видит `state`, собранный из `input`: три поля приходят JSON-строками и
  разбираются, поэтому ключи идут в алфавитном порядке — так они лежат в
  датасете.

## Про исторический прогон

Релиз размечен более ранним клиентом на `httpx` + `guided_json`. Он удалён,
потому что дублировал этот; исходник лежит в истории git. Условия прогона —
`docs/DATA_ANNOTATION_RUN.md`.
