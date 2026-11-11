# Data — feasibility-аудит

Статус: **фаза 1, до разметки**. Ветка `data-build`.
Спецификация: [DATA_SPEC.md](DATA_SPEC.md).

Цель фазы: проверить, что датасет вообще собирается из доступных
кандидатов, и где у нас дыры по facets, источникам и лицензиям. Это не
разметка и не обучение.

## 1. Критерии go / no-go

Фаза считается успешной, если:

- каждый facet имеет ≥400 потенциальных unsafe-кандидатов;
- каждый facet покрыт ≥2 независимыми источниками;
- для train набирается 12–15k действий при 25–30% unsafe;
- для dev/test набирается по 2–2.5k действий;
- нет лицензионных блокеров для источников, попадающих в train;
- конфликтные и ambiguous-группы можно исключить/пометить без потери
  покрытия facets.

Если facet не проходит — он либо урезается, либо добирается новым источником,
либо помечается `insufficient support` и не участвует в headline-метриках.

## 2. Входы

| Файл | Что даёт |
|---|---|
| `artifacts/corpus_v1/candidate_summary.json` | 136 980 уникальных action-кандидатов, разбивка по источникам |
| `artifacts/data_bundle_v1/summary.json` | 16 937 размеченных действий, source/label slices, quarantine/pending |
| `artifacts/autoresearch_sixhour_v1/data/*.manifest.json` | текущие train/dev/test counts и source mix |
| `artifacts/autoresearch_sixhour_v1/DATA_SEMANTICS_REVIEW.md` | конфликты, ambiguous-группы, label provenance |
| `artifacts/corpus_v1/additional_trace_sources.md` | новые источники, лицензии, ограничения |
| `artifacts/data_bundle_v1/source_inventory.json` | acquired sources и их роли |

Серверный candidate pool (`candidates_v3`) на этой машине не смонтирован;
inventory-скрипты рассчитаны на запуск там, где лежит JSONL.

## 3. Инвентарь кандидатов

Из `candidate_summary.json`:

| Источник | Уникальных action-кандидатов | Доля |
|---|---:|---:|
| AgentAlign | 49 374 | 36.0% |
| StepShield | 42 172 | 30.8% |
| ToolSafety | 29 451 | 21.5% |
| StepSafe | 15 983 | 11.7% |
| **Всего** | **136 980** | 100% |

Дополнительно:

- `exact_duplicate_actions`: 7 921 (подмножество, не добавлять к сумме);
- `published_label_candidates`: 16 083;
- `evaluation_overlap_quarantined`: 1 559;
- `source_label_conflicts`: 0;
- `training_ready`: false.

Ключевое ограничение: **размечено только ~12% пула**. Чтобы собрать train
12–15k, нужно либо расширить teacher-silver, либо разметить стратифицированную
подвыборку. Пул достаточен по объёму, но не по меткам.

Текущие split'ы (v1, для сравнения):

| Split | Диалоги | Действия | Unsafe | Компоненты |
|---|---:|---:|---:|---:|
| train_small | 2 009 | 5 647 | 641 | 449 |
| train_full | 6 058 | 16 877 | 2 245 | 1 411 |
| dev_select | 301 | 886 | 147 | 258 |
| dev_confirm | 299 | 949 | 181 | 262 |
| oldval | 425 | 933 | 97 | 122 |

## 4. Facet coverage matrix (предварительная)

`H` — ожидаемо много, `M` — средне, `L` — мало, `?` — неизвестно до
weak-labeling. Это гипотеза, а не измерение.

| Facet | StepSafe | StepShield | ToolSafety | AgentDoG (OOD) | OpenAgentSafety | AgentDyn | AgentDojo | Nebius SWE | ≥2 источника? |
|---|---|---|---|---|---|---|---|---|---|
| `data_exposure` | M | M | M | M | M | M | M | L | да |
| `destructive` | M | H | M | M | H | L | M | M | да |
| `financial` | M | L | M | L | L | L | L | — | риск |
| `credential_identity` | M | M | M | M | M | H | M | L | да |
| `authorization_scope` | H | M | M | M | M | M | H | L | да |
| `injection_hierarchy` | M | L | M | H | M | H | M | L | да |
| `code_execution` | L | H | M | L | M | L | M | H | да |
| `deception_social` | M | L | M | M | L | L | L | — | риск |
| `physical_cyber_harm` | L | L | M | M | L | L | L | — | риск |
| `policy_legal` | H | M | M | M | M | L | M | — | да |

Риски:

- `financial`, `deception_social`, `physical_cyber_harm` могут не набрать
  ≥400 unsafe из ≥2 источников. Для них нужен либо новый источник
  (tau2 после лицензии, Agent-SafetyBench после overlap-аудита), либо
  явное решение об `insufficient support`.
- `code_execution` и `destructive` сильно зависят от StepShield/Nebius;
  нужен второй независимый источник, иначе facet = source shortcut.
- `injection_hierarchy` в train будет держаться на AgentDyn/AgentDojo;
  AgentDoG остаётся OOD, поэтому его нельзя считать train-покрытием.

## 5. Лицензии и provenance

| Источник | Роль | Лицензия / статус | Блокер |
|---|---|---|---|
| StepSafe | train core | published annotations; проверить terms | нет, но hindsight-разметка |
| StepShield | train | проверить лицензию релиза | нет |
| AgentDoG | OOD | проверить лицензию | нет для OOD |
| AgentAlign | train safe | проверить лицензию HF | нет |
| ToolSafety | train | проверить лицензию HF | нет |
| OpenAgentSafety | train candidate | MIT | нет |
| AgentDyn | train candidate | MIT | нет |
| AgentDojo | train candidate | MIT | нет |
| Nebius SWE | train candidate | CC BY 4.0 | нет |
| RAS-Eval | train candidate | license null | **да** |
| tau2 | train candidate | лицензия не заявлена | **да** |
| Agent-SafetyBench | reannotation | проверить | overlap-аудит |
| ATBench | eval only | Apache-2.0 | не train |
| R-Judge | eval only | проверить | не train |
| StepGuard | eval only | Apache-2.0 + third-party rights | не train |
| TS-Bench | eval only | проверить | не train |
| Nemotron-AIQ | eval only | NVIDIA eval-only | не train |

Правило: пока лицензия не подтверждена, источник не попадает в train.
Eval-only источники не попадают в train даже при наличии меток.

## 6. Gap analysis

Заполняется после inventory-скрипта на серверном пуле. Шаблон:

| Facet | Доступно unsafe | Нужно | Gap | Источники | Решение |
|---|---:|---:|---:|---|---|
| `data_exposure` | ? | 400 | ? | StepSafe, StepShield, ... | ? |
| `destructive` | ? | 400 | ? | ... | ? |
| ... | | | | | |

Решение по каждому facet: `ok`, `add_source`, `reannotation`, `insufficient`.

## 7. Что измеряем на сервере

1. Точные counts по `source × label × facet` на пуле `candidates_v3`.
2. Сколько кандидатов имеют `published_label`, `silver`, `null`.
3. Сколько групп (scenario/family/trajectory) на каждый facet.
4. Пересечения с protected benchmarks (exact/normalized/entity-masked).
5. Конфликтные и ambiguous-группы в выбранной подвыборке.
6. Оценка стоимости разметки: items/hour на пилоте.

## 8. Выход фазы 1

- `DATA_FEASIBILITY.md` с заполненной gap-таблицей.
- `inventory.json`: counts, source×facet, group sizes, null rates.
- `license_matrix.json`: источник → статус → решение.
- Решение go/no-go по каждому facet.
- Обновлённая спека, если онтология урезается.

## 9. Инструменты

- `scripts/data/candidate_inventory.py` — counts по source/label/facet/group.
- `scripts/data/sample_candidates.py` — детерминированное group-aware
  стратифицированное сэмплирование.

Оба скрипта — stdlib-only, не запускают обучение и не меняют исходные данные.
