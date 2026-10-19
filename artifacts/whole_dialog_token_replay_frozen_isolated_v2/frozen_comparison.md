# Token aggregation: frozen

Exploratory fixed aggregation grid on previously viewed benchmarks; no token gold. Post-hoc thresholds are not held-out calibration. No method is selected for deployment.

Precision/recall/F1 ниже — оптимистический post-hoc максимум min(P,R); ROC/AP/PR-AUC не зависят от порога. R@FPR1% также оценивается по кривой на этих же данных.

## dojo/main: 1220 cases

| Model | Aggregation | AP | PR-AUC | ROC-AUC | P*, % | R*, % | F1*, % | R@FPR1%, % |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| frozen_linear | endpoint | 0.5495 | 0.5476 | 0.8155 | 57.92 | 60.23 | 59.05 | 2.27 |
| frozen_linear | token_max | 0.5183 | 0.5173 | 0.7813 | 55.68 | 57.10 | 56.38 | 2.56 |
| frozen_linear | top10_mean | 0.5506 | 0.5485 | 0.8054 | 58.69 | 58.52 | 58.61 | 3.41 |
| frozen_linear | tail8_mean | 0.5673 | 0.5646 | 0.8202 | 60.34 | 61.36 | 60.85 | 3.12 |
| frozen_linear_mono | endpoint | 0.5999 | 0.5989 | 0.8403 | 61.27 | 60.23 | 60.74 | 6.82 |
| frozen_linear_mono | token_max | 0.5707 | 0.5690 | 0.8096 | 57.18 | 57.67 | 57.43 | 3.41 |
| frozen_linear_mono | top10_mean | 0.5942 | 0.5923 | 0.8286 | 58.10 | 59.09 | 58.59 | 5.11 |
| frozen_linear_mono | tail8_mean | 0.6019 | 0.5993 | 0.8189 | 58.36 | 58.52 | 58.44 | 7.39 |

## harm/main: 731 cases

| Model | Aggregation | AP | PR-AUC | ROC-AUC | P*, % | R*, % | F1*, % | R@FPR1%, % |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| frozen_linear | endpoint | 0.7595 | 0.7591 | 0.5548 | 73.67 | 74.10 | 73.88 | 3.62 |
| frozen_linear | token_max | 0.7681 | 0.7678 | 0.5715 | 75.77 | 79.24 | 77.47 | 3.81 |
| frozen_linear | top10_mean | 0.7708 | 0.7703 | 0.5710 | 75.29 | 75.43 | 75.36 | 4.19 |
| frozen_linear | tail8_mean | 0.7598 | 0.7590 | 0.5526 | 72.44 | 95.62 | 82.43 | 1.52 |
| frozen_linear_mono | endpoint | 0.7585 | 0.7580 | 0.5549 | 74.50 | 77.90 | 76.16 | 2.86 |
| frozen_linear_mono | token_max | 0.7715 | 0.7711 | 0.5619 | 73.96 | 74.67 | 74.31 | 3.05 |
| frozen_linear_mono | top10_mean | 0.7691 | 0.7685 | 0.5580 | 73.69 | 75.24 | 74.46 | 3.81 |
| frozen_linear_mono | tail8_mean | 0.7642 | 0.7634 | 0.5627 | 74.03 | 76.00 | 75.00 | 2.10 |

## harm/loose: 731 cases

| Model | Aggregation | AP | PR-AUC | ROC-AUC | P*, % | R*, % | F1*, % | R@FPR1%, % |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| frozen_linear | endpoint | 0.4707 | 0.4675 | 0.7281 | 49.11 | 52.38 | 50.69 | 2.38 |
| frozen_linear | token_max | 0.4498 | 0.4467 | 0.7207 | 45.97 | 46.19 | 46.08 | 1.43 |
| frozen_linear | top10_mean | 0.4645 | 0.4615 | 0.7248 | 47.00 | 48.57 | 47.78 | 4.29 |
| frozen_linear | tail8_mean | 0.4879 | 0.4849 | 0.7375 | 45.71 | 60.95 | 52.24 | 7.14 |
| frozen_linear_mono | endpoint | 0.4681 | 0.4666 | 0.7042 | 45.45 | 45.24 | 45.35 | 5.71 |
| frozen_linear_mono | token_max | 0.4175 | 0.4148 | 0.6721 | 42.19 | 51.43 | 46.35 | 0.00 |
| frozen_linear_mono | top10_mean | 0.4291 | 0.4260 | 0.6798 | 43.81 | 43.81 | 43.81 | 1.90 |
| frozen_linear_mono | tail8_mean | 0.4679 | 0.4659 | 0.6794 | 43.81 | 43.81 | 43.81 | 6.19 |

## stepguard/external_test: 3017 cases

| Model | Aggregation | AP | PR-AUC | ROC-AUC | P*, % | R*, % | F1*, % | R@FPR1%, % |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| frozen_linear | endpoint | 0.8368 | 0.8375 | 0.8040 | 71.91 | 71.77 | 71.84 | 37.53 |
| frozen_linear | token_max | 0.7829 | 0.7833 | 0.7538 | 66.82 | 67.05 | 66.94 | 20.55 |
| frozen_linear | top10_mean | 0.7966 | 0.7965 | 0.7667 | 68.76 | 68.67 | 68.71 | 24.53 |
| frozen_linear | tail8_mean | 0.8279 | 0.8279 | 0.8015 | 71.56 | 71.56 | 71.56 | 28.57 |
| frozen_linear_mono | endpoint | 0.7712 | 0.7721 | 0.7689 | 69.53 | 69.81 | 69.67 | 14.08 |
| frozen_linear_mono | token_max | 0.7258 | 0.7263 | 0.7050 | 62.98 | 62.94 | 62.96 | 15.77 |
| frozen_linear_mono | top10_mean | 0.7184 | 0.7183 | 0.7077 | 63.41 | 63.41 | 63.41 | 14.29 |
| frozen_linear_mono | tail8_mean | 0.7847 | 0.7847 | 0.7733 | 70.20 | 70.15 | 70.17 | 17.52 |

## atbench/main: 981 cases

| Model | Aggregation | AP | PR-AUC | ROC-AUC | P*, % | R*, % | F1*, % | R@FPR1%, % |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| frozen_linear | endpoint | 0.6016 | 0.6005 | 0.6323 | 59.21 | 59.83 | 59.52 | 2.09 |
| frozen_linear | token_max | 0.6590 | 0.6579 | 0.7182 | 66.18 | 66.32 | 66.25 | 2.09 |
| frozen_linear | top10_mean | 0.6823 | 0.6809 | 0.7331 | 66.87 | 67.99 | 67.43 | 2.72 |
| frozen_linear | tail8_mean | 0.6393 | 0.6379 | 0.6690 | 61.97 | 61.72 | 61.84 | 2.72 |
| frozen_linear_mono | endpoint | 0.5272 | 0.5253 | 0.5478 | 53.67 | 53.56 | 53.61 | 1.88 |
| frozen_linear_mono | token_max | 0.6387 | 0.6376 | 0.6795 | 61.64 | 61.51 | 61.57 | 1.67 |
| frozen_linear_mono | top10_mean | 0.6596 | 0.6577 | 0.7018 | 64.03 | 64.44 | 64.23 | 2.51 |
| frozen_linear_mono | tail8_mean | 0.6016 | 0.6004 | 0.6277 | 59.75 | 59.62 | 59.69 | 2.30 |

## rjudge/main: 571 cases

| Model | Aggregation | AP | PR-AUC | ROC-AUC | P*, % | R*, % | F1*, % | R@FPR1%, % |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| frozen_linear | endpoint | 0.7436 | 0.7422 | 0.7944 | 74.75 | 74.75 | 74.75 | 2.99 |
| frozen_linear | token_max | 0.7218 | 0.7207 | 0.7243 | 67.53 | 69.10 | 68.31 | 4.98 |
| frozen_linear | top10_mean | 0.7444 | 0.7428 | 0.7481 | 69.37 | 76.74 | 72.87 | 7.64 |
| frozen_linear | tail8_mean | 0.7317 | 0.7301 | 0.7686 | 71.01 | 72.43 | 71.71 | 3.99 |
| frozen_linear_mono | endpoint | 0.7133 | 0.7106 | 0.7772 | 72.96 | 74.42 | 73.68 | 1.33 |
| frozen_linear_mono | token_max | 0.6685 | 0.6661 | 0.6725 | 63.79 | 73.75 | 68.41 | 2.33 |
| frozen_linear_mono | top10_mean | 0.7196 | 0.7179 | 0.7136 | 67.00 | 66.11 | 66.56 | 1.99 |
| frozen_linear_mono | tail8_mean | 0.7153 | 0.7133 | 0.7578 | 70.72 | 71.43 | 71.07 | 1.99 |

