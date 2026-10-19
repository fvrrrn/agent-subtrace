# Token aggregation: lora

Exploratory fixed aggregation grid on previously viewed benchmarks; no token gold. Post-hoc thresholds are not held-out calibration. No method is selected for deployment.

Precision/recall/F1 ниже — оптимистический post-hoc максимум min(P,R); ROC/AP/PR-AUC не зависят от порога. R@FPR1% также оценивается по кривой на этих же данных.

## dojo/main: 1220 cases

| Model | Aggregation | AP | PR-AUC | ROC-AUC | P*, % | R*, % | F1*, % | R@FPR1%, % |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| lora_linear | endpoint | 0.7113 | 0.7125 | 0.8878 | 68.18 | 68.18 | 68.18 | 9.94 |
| lora_linear | token_max | 0.5778 | 0.5762 | 0.8493 | 62.94 | 70.45 | 66.49 | 0.00 |
| lora_linear | top10_mean | 0.6349 | 0.6325 | 0.8652 | 63.16 | 71.59 | 67.11 | 6.82 |
| lora_linear | tail8_mean | 0.6753 | 0.6731 | 0.8579 | 68.75 | 68.75 | 68.75 | 7.95 |
| lora_linear_mono | endpoint | 0.6582 | 0.6594 | 0.8588 | 63.31 | 64.20 | 63.75 | 8.24 |
| lora_linear_mono | token_max | 0.5383 | 0.5365 | 0.8231 | 57.14 | 63.64 | 60.22 | 0.85 |
| lora_linear_mono | top10_mean | 0.5983 | 0.5959 | 0.8427 | 58.84 | 60.51 | 59.66 | 5.97 |
| lora_linear_mono | tail8_mean | 0.6439 | 0.6416 | 0.8436 | 65.06 | 65.06 | 65.06 | 7.39 |

## harm/main: 731 cases

| Model | Aggregation | AP | PR-AUC | ROC-AUC | P*, % | R*, % | F1*, % | R@FPR1%, % |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| lora_linear | endpoint | 0.8173 | 0.8182 | 0.6263 | 74.48 | 75.05 | 74.76 | 7.62 |
| lora_linear | token_max | 0.8272 | 0.8272 | 0.6420 | 76.43 | 79.05 | 77.72 | 10.86 |
| lora_linear | top10_mean | 0.8349 | 0.8346 | 0.6501 | 76.00 | 79.62 | 77.77 | 8.57 |
| lora_linear | tail8_mean | 0.8117 | 0.8114 | 0.6047 | 75.74 | 77.90 | 76.81 | 7.43 |
| lora_linear_mono | endpoint | 0.8321 | 0.8332 | 0.6478 | 75.92 | 78.67 | 77.27 | 12.00 |
| lora_linear_mono | token_max | 0.8333 | 0.8335 | 0.6456 | 76.41 | 80.19 | 78.25 | 16.00 |
| lora_linear_mono | top10_mean | 0.8388 | 0.8386 | 0.6501 | 75.69 | 83.05 | 79.20 | 14.48 |
| lora_linear_mono | tail8_mean | 0.8204 | 0.8201 | 0.6164 | 75.52 | 75.81 | 75.67 | 11.43 |

## harm/loose: 731 cases

| Model | Aggregation | AP | PR-AUC | ROC-AUC | P*, % | R*, % | F1*, % | R@FPR1%, % |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| lora_linear | endpoint | 0.6131 | 0.6137 | 0.7641 | 54.55 | 54.29 | 54.42 | 14.29 |
| lora_linear | token_max | 0.5829 | 0.5807 | 0.7738 | 53.37 | 52.86 | 53.11 | 5.71 |
| lora_linear | top10_mean | 0.6050 | 0.6016 | 0.7722 | 53.55 | 53.81 | 53.68 | 15.71 |
| lora_linear | tail8_mean | 0.6058 | 0.6044 | 0.7504 | 53.85 | 53.33 | 53.59 | 14.29 |
| lora_linear_mono | endpoint | 0.6456 | 0.6472 | 0.7749 | 56.25 | 55.71 | 55.98 | 19.52 |
| lora_linear_mono | token_max | 0.5953 | 0.5920 | 0.7726 | 52.36 | 52.86 | 52.61 | 5.71 |
| lora_linear_mono | top10_mean | 0.6262 | 0.6247 | 0.7739 | 53.81 | 53.81 | 53.81 | 17.14 |
| lora_linear_mono | tail8_mean | 0.6290 | 0.6279 | 0.7605 | 54.29 | 54.29 | 54.29 | 26.19 |

## stepguard/external_test: 3017 cases

| Model | Aggregation | AP | PR-AUC | ROC-AUC | P*, % | R*, % | F1*, % | R@FPR1%, % |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| lora_linear | endpoint | 0.8859 | 0.8866 | 0.8863 | 81.57 | 81.40 | 81.48 | 24.66 |
| lora_linear | token_max | 0.8698 | 0.8702 | 0.8609 | 78.48 | 78.37 | 78.42 | 28.91 |
| lora_linear | top10_mean | 0.8711 | 0.8711 | 0.8639 | 78.47 | 78.57 | 78.52 | 27.49 |
| lora_linear | tail8_mean | 0.8893 | 0.8892 | 0.8908 | 81.81 | 81.81 | 81.81 | 25.81 |
| lora_linear_mono | endpoint | 0.8912 | 0.8921 | 0.8845 | 80.62 | 80.73 | 80.67 | 30.39 |
| lora_linear_mono | token_max | 0.8706 | 0.8711 | 0.8608 | 78.14 | 78.30 | 78.22 | 28.64 |
| lora_linear_mono | top10_mean | 0.8741 | 0.8741 | 0.8645 | 78.25 | 78.30 | 78.28 | 29.04 |
| lora_linear_mono | tail8_mean | 0.8921 | 0.8921 | 0.8894 | 81.19 | 81.13 | 81.16 | 29.65 |

## atbench/main: 981 cases

| Model | Aggregation | AP | PR-AUC | ROC-AUC | P*, % | R*, % | F1*, % | R@FPR1%, % |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| lora_linear | endpoint | 0.5358 | 0.5351 | 0.4672 | 48.73 | 100.00 | 65.52 | 4.60 |
| lora_linear | token_max | 0.6539 | 0.6532 | 0.6831 | 64.54 | 65.48 | 65.01 | 2.93 |
| lora_linear | top10_mean | 0.6790 | 0.6778 | 0.6956 | 64.86 | 65.27 | 65.07 | 4.39 |
| lora_linear | tail8_mean | 0.5459 | 0.5448 | 0.4759 | 48.82 | 99.16 | 65.42 | 5.44 |
| lora_linear_mono | endpoint | 0.4972 | 0.4966 | 0.4268 | 48.73 | 100.00 | 65.52 | 3.56 |
| lora_linear_mono | token_max | 0.6481 | 0.6478 | 0.6691 | 63.51 | 64.44 | 63.97 | 2.30 |
| lora_linear_mono | top10_mean | 0.6692 | 0.6682 | 0.6787 | 63.60 | 63.60 | 63.60 | 4.39 |
| lora_linear_mono | tail8_mean | 0.5253 | 0.5242 | 0.4451 | 48.73 | 100.00 | 65.52 | 6.28 |

## rjudge/main: 571 cases

| Model | Aggregation | AP | PR-AUC | ROC-AUC | P*, % | R*, % | F1*, % | R@FPR1%, % |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| lora_linear | endpoint | 0.8942 | 0.8953 | 0.8341 | 79.60 | 79.07 | 79.33 | 61.13 |
| lora_linear | token_max | 0.9087 | 0.9091 | 0.8584 | 79.73 | 79.73 | 79.73 | 53.49 |
| lora_linear | top10_mean | 0.9189 | 0.9187 | 0.8746 | 81.06 | 81.06 | 81.06 | 54.49 |
| lora_linear | tail8_mean | 0.8895 | 0.8894 | 0.8481 | 78.93 | 78.41 | 78.67 | 44.52 |
| lora_linear_mono | endpoint | 0.8853 | 0.8864 | 0.8198 | 77.48 | 77.74 | 77.61 | 49.17 |
| lora_linear_mono | token_max | 0.8935 | 0.8938 | 0.8391 | 78.11 | 77.08 | 77.59 | 42.86 |
| lora_linear_mono | top10_mean | 0.9072 | 0.9071 | 0.8567 | 79.73 | 79.73 | 79.73 | 46.51 |
| lora_linear_mono | tail8_mean | 0.8670 | 0.8668 | 0.8065 | 74.83 | 75.08 | 74.96 | 45.85 |

