# Linear/step × NULL/MIL

**Завершены** четыре новых условия по4эпохи; fresh same seed20261017.
`runs/curve-event-shapes-v1`, launcher PID1700053, immutable `-launch/source`.
GPU2/3: сначала parallel NULL/MIL linear, затем parallel NULL/MIL step.
Whole-dialog-v2/cache неизменны, frozen Qwen не загружается. Без LoRA/mono/attention.
`L=token_BCE+0.1event_BCE` одновременно на каждом step.

- linear: для unsafe `target_i=i/N`, i=1..N текущего действия; safe0.
- step: для unsafe все токены1; safe0.
- Контекст/unlabelled без loss; Mamba state сохраняется между действиями.

**Linear здесь local0→1, а не прежняя previous-hard-label interpolation**
whole-dialog-v3. Это специально меняет только текущий position-profile
`y*g(i/N)` относительно broad curve. Safe после unsafe остаётся0, а не спуск1→0.
Не называть это точным воспроизведением старого linear training.

В каждом run исходные CUDA gates: отдельные градиенты обоих loss, loss masks,
оба Mamba, suffix causality/reload/longest15723 backward; audit gradients
discarded, train fresh. Все4epoch checkpoints/optimizer/RNG states,
validation all-token logits и компонентные losses сохраняются.

Выбор лучшего среди4условий по minimum hard event validation BCE,
затем tie curve/arm lexicographic. Используется тот же validation933,
не ATBench. Отдельный attention-before-Mamba эксперимент стартует после
этого выбора в `runs/curve-event-attention-before-v1`, собственным процессом
и с собственными immutable source/checkpoints/protocol.

Watcher/wait/copy `sh_14ecfe4a8001XfaZoSEdLolgpl` завершён, оба этапа successful.
Внешние оценки этих новых условий не назначены. Один seed и validation
selection не доказывают held-out выигрыш/калибровку; цели искусственные,
не token gold. Для step короткая вредная составляющая тоже размечает всё
unsafe-действие; правильная локализация не гарантирована.

## Итоги validation933(97unsafe), threshold.5

| Условие | Epoch | BCE | AP | PR-AUC trapezoid | ROC-AUC | P,% | R,% | F1,% | Acc,% | FP/FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Linear NULL | 2 | .175321 | .7182 | .7165 | .9440 | 77.42 | 49.48 | 60.38 | 93.25 | 14/49 |
| Linear MIL | 2 | .195400 | .6819 | .6808 | .9209 | 84.62 | 34.02 | 48.53 | 92.50 | 6/64 |
| Step NULL | 2 | .184259 | .6774 | .6760 | .9266 | 75.41 | 47.42 | 58.23 | 92.93 | 15/51 |
| Step MIL | 3 | .200896 | .6171 | .6153 | .9122 | 67.80 | 41.24 | 51.28 | 91.85 | 19/57 |

FPR:1.67%,.72%,1.79%,2.27%. Лучший по заранее заявленному BCE —
linear NULL. Разница с broad NULL из4epoch-v2 мала: BCE.175321vs.175558,
AP.7182vs.7106, но recall49.48vs50.52%; не заявлять значимый выигрыш.
Linear MIL лучше broad MIL по BCE/AP, но это один seed и ранее наблюдался
training rerun drift. Step не дал универсального улучшения.
Для выбранного linear NULL завершён отдельный attention-before run;
сравнение в соседнем `curve_event_attention_before_v1/`.
