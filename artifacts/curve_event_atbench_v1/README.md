# ATBench: шкала скоров curve + event heads

**FAILED до первого case:** отсутствовал импортируемый модуль
`evaluate_whole_dialog_benchmarks`; моделей/скоров в этом запуске нет.
Исправление удаляет лишнюю зависимость (требовался только SHA helper).
Продолжение — отдельный `runs/curve-event-atbench-v2`, старые source/logs
не изменены. Старый watcher завершён.

User-authorized evaluation, не обучение и не подбор порогов.
Server: `runs/curve-event-atbench-v1`; immutable `-launch/source`.
Launcher PID1636028, GPU2/3; watcher/copy
`sh_14ec6fd84001PErZ2oqlR0KEBa`, ждать уведомление, не опрашивать completion.

Зафиксированы validation-selected checkpoints четырёхэпохового запуска:
NULL epoch2, MIL epoch4. Head hashes/selection/history сохранены в `locked/`.
Сравнение — прежний frozen-linear epoch2 endpoint без event-loss/mono,
его исходные outputs читаются без повторного inference.

Тот же causal replay ATBench981 (478unsafe/503safe),4365actions,
19 прежних исключений по длине сохранены. Native Qwen template и прежние
точные IDs/спаны,656whole-dialog /325causal-prefix fallback из-за template
rewrite, максимум64773токена, без обрезки. Готовый BF16 Qwen cache read-only,
Qwen не загружается. Перед scoring проверяются input/checkpoint/template/
weights/revision/environment hashes. Новые ошибки сохраняются и останавливают
run без автоматического retry/ослабления допуска.

Primary: maximum native trained event score по всем действиям, включая Complete.
Sensitivity: tool-only maximum на одних и тех же tool-containing cases.
Gold только траекторный, не action/token. Score не объявляется калиброванной
вероятностью. Квантили и доля>=0.5 считаются для траекторного score отдельно
на safe/unsafe; AP, trapezoidal PR-AUC, ROC-AUC и confusion/P/R/F1/Acc при0.5.
Порог не подбирается. Это ранее просмотренный benchmark и не официальный
full1000 completed-trajectory evaluator; разные curve/readout/duration
не позволяют приписать различие одному event-loss.

Точные all-token raw/NULL logits, IDs, spans остаются на сервере в `*/tokens/`;
predictions, protocol, hashes, checkpoints, costs/complete и comparison
копируются локально по завершении. `status.json` — источник статуса завершения.
