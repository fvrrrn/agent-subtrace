# Frozen Instruct v3

Same explicitly authorized five-arm, two-epoch experiment; original deadline
`1792345185`. v1 and v2 artifacts and snapshots remain unchanged. v3 uses its own
immutable `runs/frozen-instruct-4b-v3-launch/source` and outputs under
`runs/frozen-instruct-4b-v3`. The v1 cache/data are explicit read-only paths.

Both serialization boundaries are now repaired: runtime feature identity
restores `environment.capability` from JSON list to Torch tuple; persisted
training config copies that nested field back to list without mutating runtime
identity. Full identity and gate-config comparisons remain exact.

Before CUDA, direct CPU diagnostics check each of the five configurations
through JSON and Torch round trips and the actual base trainer's gate-config
equality predicate. All 2310 caches are then verified again, including exact
identity, input hashes, dimensions, dtype, finiteness and SHA-256. The complete
v1 cache/metadata hash manifest must equal the saved v2 admission manifest.

All mandatory CUDA gates run anew. Training keeps simultaneous token/event
losses, original initialization/data/selection, no added attention or LoRA.
GPU 3 occupancy is checked before every GPU stage. Real child tqdm output is
streamed to logs and the background SSH shell `sh_14fd1499b001WTSNJ7Z0KCv4P0`.
Launch budget was approximately 4726 seconds (78.8 minutes).

## Confirmed execution milestone

CPU serialization checks passed for all five configurations, and full cache
admission passed 2310/2310 with unchanged v1 hash manifest
`39d4055346ac4b9486699b444248869bbd49b7db7a4230be8412ff5aca97c1b1`.
A_broad CUDA gate passed. Actual epoch 1 progress reached eight completed
dialogs, establishing at least two optimizer steps under accumulation=4.
This milestone was observed through filesystem notifications, without polling.
Launcher PID: `3522241`; training child: `3528707`; GPU 3.
Remaining original budget at milestone: 4625.52 seconds (77.1 minutes).
The sequential launcher schedules all five arms for two epochs each.
The launch milestone preceded successful completion. Pidfd completion/copy watcher:
`sh_14fd2c10f001jF2ECFJNw07Exm`.

## Completed results

All five arms completed two epochs; all CUDA gates passed; launcher status is
`complete`. Both background shells exited, and metadata/logs were copied here.
Total launcher elapsed time was approximately 641 seconds. Every arm selected
epoch 2 by dev-select AP. Evaluation covers 301 dialogs / 886 actions, including
147 unsafe and 739 safe. This is validation-selected, one seed, not external
confirmation. No further training or evaluation was started.

| Arm | AP | Trapezoidal PR-AUC | ROC-AUC | Event BCE | P@.5 % | R@.5 % | F1 % | Accuracy % | TP | FP | FN | TN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A_broad | .47718 | .47602 | .75854 | .46414 | 84.21 | 10.88 | 19.28 | 84.88 | 16 | 3 | 131 | 736 |
| A_linear | .48286 | .48166 | .77667 | .42856 | 76.19 | 10.88 | 19.05 | 84.65 | 16 | 5 | 131 | 734 |
| A_step | .50798 | .50711 | .79466 | .41955 | 87.50 | 9.52 | 17.18 | 84.76 | 14 | 2 | 133 | 737 |
| B | .38554 | .38149 | .77914 | .42494 | 100.00 | .68 | 1.35 | 83.52 | 1 | 0 | 146 | 739 |
| C | .47760 | .47559 | .80905 | .36986 | 68.57 | 16.33 | 26.37 | 84.88 | 24 | 11 | 123 | 728 |

A_step has the best AP in this Instruct comparison; C has the lowest event BCE
and highest recall at 0.5. Recall remains low in every condition. The result
does not establish an overall improvement from switching Base to Instruct.
Per-arm selection, gates, histories, metrics and complete files are preserved.
