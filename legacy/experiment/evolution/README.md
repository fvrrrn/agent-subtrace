# Локальная RRSI-эволюция

Единый контекст исследования: [../../AGENTS.md](../../AGENTS.md).
Метод: [RRSI, Google Cloud AI Research et al.](https://arxiv.org/abs/2609.24972).
Правила, роли и критерии принятия зафиксированы в [PROTOCOL.md](PROTOCOL.md).
Аудит исходной траектории и новых примеров: [DATA_AUDIT.md](DATA_AUDIT.md).

## Что здесь эволюционирует

Frozen runtime — Laya, без дообучения. Кандидатов предлагает сабагент
**Sol 6.1** в диалоге с родительским агентом. Родитель **GPT-6 Astra** читает
изменения, проверяет leakage и утверждает кандидата до оценки. Следующий
раунд Sol получает ledger и результаты development-прогонов.

`candidates/*.json` задают hypothesis, parent, atomic edits, proposer session,
вердикт критика и активную конфигурацию. `harness.py` реализует общие операции.
Вся логика task-conditioned, без имен/ответов примеров в коде.

Это агент-управляемый локальный цикл, а не автономный Python proposer:
Python запускает frozen classifier и детерминированный selector;
генерация гипотез происходит через реальные Sol-сабагенты OpenCode.

## Запуск

Из папки `experiment/`:

```bash
# Оценить утверждённого кандидата на original + development.
uv run python evolution/evaluate.py evolution/candidates/h0.json

# Повторить оценку выбранного harness.
uv run python evolution/evaluate.py evolution/best.json

# Применить заранее заданные gates к уже сохранённым измерениям.
uv run python evolution/select.py --round 0 \
  --incumbent evolution/runs/<h0-run> \
  evolution/runs/<candidate-run>
```

`select.py` дописывает `ledger.jsonl` и обновляет `best.json`; повторно
применять его к старому раунду без намерения дописать журнал не следует.
`evaluate.py` каждый раз создаёт новую папку и не изменяет прошлые результаты.

## Артефакты

- `dev.json`: четыре парных diagnostic-задачи, восемь оцениваемых действий.
- `heldout.json` + manifest: отдельный набор, созданный независимым Sol до
  оценки кандидатов. Его результаты не доступны proposer во время эволюции.
- `runs/*/trace.json`: все точные входы/ответы, scores, gold только после
  предсказания, software versions, model revision, data/code hashes, стоимость.
- В каждой папке прогона также копии evaluated `harness.py`, `evaluate.py`,
  `run.py` для восстановления исходного кода именно этого измерения.
- `runs/*/summary.json`: метрики отдельно original/dev/heldout и graph/full.
- `ledger.jsonl`: принятые и отвергнутые кандидаты, score/cost deltas, причины.
- `best.json`: текущий принятый harness; итог зафиксирован в финальном отчёте.

Максимум скоров нескольких rule-checks — operational OR, а не калиброванная
вероятность объединённого события. Одинаковая процедура применяется к нашему
методу и baseline. LINK остаётся одной ranking-choice задачей; пороги неизменны.

RRSI-перенос использует файлы и snapshots вместо git worktrees и commits.
В отличие от полного открытого пространства source-edits авторов, этот pilot
проверяет небольшой набор общих компонентов classifier harness. Новые варианты
добавляются после предложения Sol и approval критика; это не полная репликация
всех экспериментов статьи.

Сравнение других моделей выполняется отдельно в `../competitors/`: смена
backbone не засчитывается как улучшение RRSI при frozen Laya.
