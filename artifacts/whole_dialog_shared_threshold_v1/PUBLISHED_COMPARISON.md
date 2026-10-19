# LoRA без mono против опубликованных моделей

## Наше условие

Qwen3.5-4B-Base + LoRA + два residual Mamba2 + бинарная голова, linear token
training, locked epoch1. Readout: среднее top ceil10% вероятностей текущего
действия; для trajectory наборов — максимум event scores. Один общий порог
**0.02398177981376648** (точное значение см. `shared_threshold.json`).
Порог выбран постфактум на5наборах с равным весом, max min(macro P,macro R).
Это НЕ held-out результат и не matched leaderboard с публикациями.

Основной отчёт: [SHARED_THRESHOLD.md](SHARED_THRESHOLD.md). AP/PR-AUC/ROC-AUC
и полные матрицы ошибок сохранены в JSON, не зависят от этого порога.
Ни одна внешняя модель в рамках сравнения не запускалась.

## Источник A: StepGuard, arXiv2608.24777v1, Table1

Проверены Table1, Appendix E.4, G.2. Все значения ниже — **опубликованные
Accuracy / unsafe-class F1, %**; результаты нашего checkpoint явно отдельные.
Harm у авторов содержит только416 definite safe/unsafe, **не731 strict**.

| Модель | Dojo Acc/F1 | Harm416 Acc/F1 | ATBench Acc/F1 | R-Judge Acc/F1 |
|---|---:|---:|---:|---:|
| GPT-5.4, published | 93.9 / 90.3 | 68.8 / 76.4 | 65.9 / 69.8 | 91.8 / 92.4 |
| Qwen3.5-397B-A17B, published | 94.6 / 91.3 | 71.3 / 77.5 | 63.5 / 64.0 | 90.2 / 90.9 |
| Qwen2.5-7B-Instruct, published | 83.0 / 67.3 | 73.1 / 71.6 | 56.4 / 28.8 | 62.2 / 66.8 |
| Qwen3-4B-Instruct-2507, published | 86.6 / 72.6 | 72.1 / 65.9 | 57.0 / 32.3 | 70.0 / 72.9 |
| Qwen3-Guard, published | 71.1 / 0.0 | 80.0 / 77.3 | 53.2 / 7.3 | 50.4 / 15.7 |
| LlamaGuard3-8B, published | 73.0 / 26.1 | 82.2 / 82.3 | 47.9 / 16.4 | 48.4 / 10.5 |
| ProGuard, published | 70.7 / 9.6 | 80.0 / 82.4 | 54.2 / 21.2 | 63.5 / 51.6 |
| AgentDoG-Qwen3-4B, published | 53.7 / 55.6 | 51.2 / 66.6 | 61.9 / 68.8 | 88.7 / 90.0 |
| ShieldAgent-THU, published | 65.2 / 61.2 | 56.4 / 69.8 | 64.6 / 69.8 | 81.9 / 85.2 |
| Safiron, published | 76.8 / 56.5 | 53.8 / 34.2 | 63.2 / 65.0 | 53.5 / 45.6 |
| TS-Guard, published | 91.7 / 83.5 | 76.2 / 76.4 | 56.6 / 21.8 | 78.6 / 76.5 |
| StepGuard4B, published | 95.7 / 93.0 | 73.8 / 75.2 | 71.2 / 71.2 | 89.2 / 89.6 |
| **Наш LoRA4B, общий post-hoc порог** | **75.57 / 43.77** | **69.71 / 62.05** | **49.13 / 65.66** | **66.20 / 73.95** |

Источник: https://arxiv.org/html/2608.24777v1#S6.T1

Почему не прямая гонка:
- Авторы судят завершённые ATBench/R-Judge trajectories, мы агрегируем
  pre-execution readouts без результата текущего действия; доступная информация
  отличается. Их Table15 прямо задаёт trajectory prompt.
- Наш ATBench981/1000 с19 длиновыми исключениями; в исходном benchmark1000.
- R-Judge у нас571; точные input hashes/IDs у внешнего evaluator не сверены.
- У авторов свои official/common prompts,32K context, greedy output parsing;
  у нас native template,65K admission, score threshold. Специализированные
  guards используют собственные промпты. Даже при одинаковых исходных примерах
  это не одинаковый inference protocol.
- Наш threshold оптимизирован на benchmark labels; external Table1 имеет
  другую процедуру model/decision selection. Нельзя заявлять superiority.

## Источник B: ToolSafe, arXiv2601.10156v1, Table3, STRICT

Здесь Harm именно731, controversial включены вunsafe. Это ближайший
source/mapping к нашему PRIMARY Harm. Опубликованы Acc/F1/Recall (%):

| Модель | Harm731 Acc/F1/Recall | Dojo1220 Acc/F1/Recall |
|---|---:|---:|
| GPT-4o | 75.23 / 84.80 / 96.19 | 55.74 / 56.59 / 100.00 |
| Qwen3-8B | 54.46 / 58.94 / 45.52 | 79.65 / 72.32 / 92.31 |
| Qwen2.5-7B-IT | 67.99 / 80.17 / 90.09 | 43.97 / 50.47 / 99.14 |
| Llama-Guard-3-8B | 81.53 / 86.35 / 81.33 | 74.75 / 33.33 / 21.87 |
| Qwen3Guard-8B-Gen | 80.57 / 86.27 / 84.95 | 70.57 / 3.23 / 1.70 |
| ShieldAgent-THU | 71.13 / 82.63 / 95.62 | 60.90 / 58.91 / 97.16 |
| Safiron | 46.10 / 45.88 / 31.81 | 70.98 / 52.80 / 56.25 |
| TS-Guard | 84.81 / 90.16 / 96.95 | 91.72 / 86.18 / 89.49 |
| **Наш LoRA4B, общий post-hoc порог** | **48.43 / 46.98 / 31.81** | **75.57 / 43.77 / 32.95** |

Источник: https://arxiv.org/html/2601.10156v1#S4.T3
Различия промптов/шкалы/decision parsing сохраняются. Числа TS-Guard из
источников A и B нельзя выдавать за одну и ту же оценку: другой Harm-subset
и даже Dojo F1 различается из-за evaluation protocol.

## Источник C: текущая официальная страница ATBench

https://yuli-cs.net/papers/atbench/ и https://github.com/LiYu0524/ATbench
публикуют более новые trajectory-level результаты, отдельно от StepGuardTable1.
Примеры (ATBench Acc/P/R/F1):
- GPT-5.4:73.7/68.5/87.1/76.7;
- AgentDoG1.5-4B-U:78.4/79.8/75.7/77.7;
- AgentDoG1.5-4B:72.4/69.2/80.3/74.3;
- Qwen3.5-4B:45.9/41.2/20.7/27.6.
R-Judge Acc/P/R/F1: GPT-5.4 93.3/93.1/94.3/93.7;
Gemini3.1Pro97.3/99.1/95.7/97.4;
AgentDoG1.5-4B92.2/91.7/93.7/92.7;Qwen3.5-4B81.0/82.1/81.9/82.0.
Это не наш causal replay, и только full1000 ATBench ранжируются официально.
Наш981-subset не входит в официальный matched leaderboard. Не смешивать эти
цифры с Table1StepGuard как один эксперимент или усреднять их.

## StepGuard-Data

https://huggingface.co/datasets/ninty-seven/stepguard_data — **training dataset**
авторов. Наши3017test/161dev — собственный нормализованный split. Для точных
3017 IDs опубликованной таблицы других моделей не найдено. Нельзя подменять
этот результат метриками StepGuard на TS-Bench или на100 Nemotron traces.

Во внешних основных таблицах выше **нет AP/trapezoidal PR-AUC/ROC-AUC**.
Эти значения не восстанавливаются из одной Acc/F1-точки и не выдумываются.
Наши три ranking-метрики приведены полностью в SHARED_THRESHOLD.md/JSON.
Все внешние числа прочитаны из публикаций; никаких API/GPU запусков
других моделей для этого сравнения не выполнялось.
