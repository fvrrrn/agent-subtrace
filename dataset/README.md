# data — action-level safety dataset

Silver-разметка действий агента судьёй `qwen3.8-flash-next` по промпту
`scripts/dataset/flash_next/prompt.txt`.
Спецификация: `docs/DATA_SPEC.md`. Журнал разметки:
`docs/DATA_ANNOTATION_RUN.md`. Выбор промпта: `docs/DATA_PROMPT_TUNING.md`.

Единица — одно предложенное действие по видимому причинному префиксу:
`tool_interfaces`, `user_request`, `observed_history`, `proposed_action`
(без собственного `tool_result`).

## Состав

| Split | Строк | Unsafe | Safe | Uncertain | Групп |
|---|---:|---:|---:|---:|---:|
| train | 14 670 | 3 961 (27.0%) | 10 709 | 0 | 7 228 |
| dev | 2 500 | 590 (23.6%) | 1 825 | 85 | 1 459 |
| test | 2 500 | 590 (23.6%) | 1 825 | 85 | 1 421 |
| ood | 501 | 227 (45.3%) | 263 | 11 | 101 |

`dev` и `test` состоят из двух срезов:

| Срез | Строк | Unsafe | Uncertain |
|---|---:|---:|---:|
| `natural` | 2 000 | 340 (17.0%) | 85 / 85 |
| `balanced` | 500 | 250 (50.0%) | 0 |

`balanced` — 250 unsafe, по одному на facet по кругу, каждый по возможности в
паре с safe-действием из той же группы (контрфакториал), плюс 250 hard
negatives.

`train` — 3 961 unsafe, 3 961 hard negatives (1:1) и 6 748 ordinary safe.
Hard negative = safe-действие в группе, где есть хотя бы одно unsafe-действие.

`ood` — AgentDoG, в train не входит никогда. Это частичная нестратифицированная
подвыборка (501 из 24 332 строк AgentDoG), размеченная до решения об исключении
источника из очереди.

## Поля

| Поле | Тип | Описание |
|---|---|---|
| `key` | str | идентификатор кандидата, стабилен между версиями |
| `group` | str | группа сценария; ни одна группа не пересекает сплиты |
| `split` | str | `train` / `dev` / `test` / `ood` |
| `slice` | str | `train` / `natural` / `balanced` / `ood` |
| `input` | struct | видимое состояние: `observed_history`, `proposed_action`, `tool_interfaces`, `user_request` |
| `label` | int8 | `1` unsafe, `0` safe, `null` uncertain |
| `uncertain` | bool | `true` для `label = null` |
| `facets` | list[str] | 1–3 facet у unsafe, пусто у safe |
| `severity` | str | `low` / `medium` / `high` / `unknown` |
| `reversibility` | str | `reversible` / `irreversible` / `unknown` |
| `actor` | str | `user` / `third_party` / `injected` / `unknown` |
| `authorization` | str | `explicit` / `ambiguous` / `absent` / `unknown` |
| `evidence` | str | `sufficient` / `insufficient` / `unknown` |
| `role` | str | `unsafe` / `hard_negative` / `ordinary_safe` / `uncertain` |
| `sources` | list[str] | провенанс |
| `input_sha256` | str | sha256 канонического `json.dumps(input)` |
| `annotation` | struct | `model`, `prompt`, `prompt_sha256`, `confidence`, `reason`, `status` |

Релиз — четыре parquet-файла, по одному на сплит, под Git LFS. Поле `input` —
struct из четырёх частей: три свободные по схеме (`observed_history`,
`proposed_action`, `tool_interfaces`) лежат JSON-строками, потому что pyarrow не
сводит произвольные схемы инструментов в один struct, а `user_request` — обычный
текст. Порядок полей алфавитный, поэтому

```python
state = {k: (v if k == "user_request" else json.loads(v)) for k, v in row["input"].items()}
assert hashlib.sha256(json.dumps(state, ensure_ascii=False).encode()).hexdigest() == row["input_sha256"]
```

восстанавливает исходное состояние байт-в-байт (проверено на всех 20 171 строке).

## Facets в train

| Facet | Unsafe | Источники |
|---|---:|---|
| `policy_legal` | 2 174 | 4 |
| `authorization_scope` | 1 983 | 4 |
| `deception_social` | 1 136 | 4 |
| `data_exposure` | 573 | 4 |
| `financial` | 571 | 3 |
| `code_execution` | 279 | 4 |
| `destructive` | 259 | 3 |
| `credential_identity` | 113 | 4 |
| `injection_hierarchy` | 112 | 2 |
| `physical_cyber_harm` | 80 | 4 |

Счётчик — вхождения facet у unsafe-строк train (в среднем 1.84 facet на unsafe).

## Провенанс

| Источник | train | dev | test | ood |
|---|---:|---:|---:|---:|
| AgentAlign | 4 908 | 1 052 | 1 064 | — |
| StepShield | 4 981 | 732 | 706 | — |
| StepSafe | 2 566 | 312 | 318 | 79 |
| ToolSafety | 2 215 | 404 | 412 | — |
| AgentDoG | — | — | — | 501 |

Доля unsafe в train по источникам: StepShield 39.4%, AgentAlign 33.3%,
StepSafe 23.4%, ToolSafety 3.9%.

## Ограничения

1. **Silver, не gold.** Метки — ответы судьи `prompt.txt`, человеческой разметки на
   этом пуле нет. IAA не измерялся. Claim ограничивается weakly supervised
   ranking.
2. **`evidence` непригоден как спаны.** У 46.3% строк судья пересказывает, а не
   цитирует вход (`evidence_not_literal`). Поэтому `evidence` — только оценка
   достаточности, span-полей в датасете нет.
3. **Квота ≥400 unsafe на facet не достигнута** для 5 из 10 facets
   (`code_execution` 279, `destructive` 259, `credential_identity` 113,
   `injection_hierarchy` 112, `physical_cyber_harm` 80). Это предел пула, а не
   сэмплирования: столько unsafe с этими facet размечено всего.
4. **Отклонение от правила «источник ≤35% unsafe в train».** StepShield даёт
   39.4%. Ограничение источника урезало бы 361 unsafe-строку (−9.1%) и сильнее
   увело бы редкие facets от квоты, поэтому unsafe взяты все. Если правило
   важнее объёма — нужен отдельный прогон сборки с cap'ом.
5. **`ood` меньше целевого объёма** (501 против 1 000–1 500) и нестратифицирован;
   prevalence 45% — собственная prevalence AgentDoG, не 15–20%.
6. **312 строк пула не размечены**: вход длиннее лимита по символам на момент
   прогона. Они не входят в датасет (0.23% пула).
7. **Pretraining contamination** неизвестна.

## Загрузка

```python
from datasets import load_dataset

ds = load_dataset("parquet", data_files={
    "train": "dataset/train.parquet",
    "dev": "dataset/dev.parquet",
    "test": "dataset/test.parquet",
    "ood": "dataset/ood.parquet",
})
train, dev, test, ood = ds["train"], ds["dev"], ds["test"], ds["ood"]
dev_natural = dev.filter(lambda r: r["slice"] == "natural")
dev_balanced = dev.filter(lambda r: r["slice"] == "balanced")
```
