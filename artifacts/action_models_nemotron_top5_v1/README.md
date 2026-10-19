# Nemotron final-report task: corrected five-model shortlist

Same adapter/data/limitations as `../action_models_nemotron_v1/README.md`.
No security binary labels invented; no claim of reproducing StepGuard100.

Only LoRA validation-best epoch2, Mamba quadratic, Mamba cosine_rank,
Mamba linear and original Kev. The old15-model Nemotron queue never started.
Prepared records and fixed checkpoints reused, no new data/tokenization or
training. Feature cache is built once when this task starts after ATBench.

Root `runs/action-models-nemotron-top5-v1`, source/launch
`runs/external-replay-top5-v1-launch/source`, queue PID2792288 onGPU2.
Initial PID2784790 CLI argument error preceded all model work and is archived.
Shared single live/wait/copy `sh_14b55119f0019EI7fOslM9r07p`; do not poll.
