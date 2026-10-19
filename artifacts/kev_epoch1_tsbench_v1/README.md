# Locked epoch-one Kev vs Mamba + LoRA on TS-Bench

User-authorized evaluation only, no training changes. Official pinned
TS-Bench-Dojo1220 and TS-Bench-Harm731 causal action records, previously
admitted without truncation and protected from the current training bundle.
Inference scores only current action before its result. Not intervention.

New model: current Kev-0.8B NOUL + unsafe-ranking epoch1, not NOUL-only.
Checkpoint immutable in `runs/kev-noul-epoch1-snapshot-v1`, complete per-file
hashes verified before evaluation. Native temperature preserved, threshold0.5.
References reuse exact completed Mamba+LoRA exponential epoch1, validation-BCE
selected epoch2 and original Kev predictions. No checkpoint/threshold selection
on these suites. Scores/states/labels checked on common successful keys;
native admission failures stay explicit. No Mamba inference repeated.

AP, trapezoidal PR-AUC, ROC-AUC, Accuracy/F1 and confusion matrices included.
Harm primary strict maps0.5→unsafe; loose maps0.5→safe and is reported
separately; definite-only is diagnostic. Dojo has binary labels so mappings agree.
Per-domain metrics preserved. Previously viewed benchmarks; exploratory, one seed.

Root `runs/kev-epoch1-tsbench-v1`, immutable source sibling `-launch/source`.
Scheduler PID3134147 waits by pidfd for two-layer training2666657 to release
GPU3, then runs Dojo followed by Harm. It does not stop any existing process.
Live/wait/copy `sh_14b8765210017Ql0bOfWKZV2mV`; do not poll completion.
