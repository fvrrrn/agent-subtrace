# Authorized action-model run

Server: `llm01:/var/local/danefedov/experiments/agent-safety/runs/action-models-v1`.
Immutable source snapshot: sibling `action-models-v1-launch/source`.
Scheduler PID at launch: **1659888**.

GPU2 queue: frozen feature cache → original Kev/random-head baselines → frozen
Mamba exponential/cosine_rank → native Kev noul_rank (2 epochs).
GPU3 queue: one-layer LoRA exponential/cosine_rank → two-layer LoRA
exponential/cosine_rank. All Mamba arms use 8 epochs. Judges GPU0/1 and foreign
processes are untouched. The queues stop on failure, without automatic retries.

Input is the fixed benchmark-excluded data-bundle-v1 95/5 split. Initial input
16937 actions; after Mamba length admission: 15218 train / 848 validation,
871 train context-overflow rejections, no truncation. Verified unsafe pairs:
809 train / 56 validation. Native Kev performs its own additional admission.

The first preflight rejected a short/full BF16 readout difference of 0.03125
(encoder relative L2 0.010789). Original failed audit and source variants are
preserved. Separate fixed-shape suffix perturbation produced zero prefix change
in both encoder and readout. The final gate separates short/full numerical
diagnostics from an exact fixed-shape causality check; the encoder relative-L2
limit remains 0.03. All four architecture paths passed finite, nonzero
Mamba/head/LoRA gradients, unchanged frozen hashes and the applicable in-memory
state reload and causal checks. No optimizer step from preflight is reused.

Metadata/checkpoints/predictions are copied here on completion. The large token
feature cache and prepared records remain on permanent server storage. Final
comparison is on common successful validation keys used for checkpoint
selection, **not an independent test**. No protected benchmark is scheduled.
