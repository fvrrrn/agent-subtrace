# Fresh two-layer Mamba + Qwen LoRA, four epochs

User stopped current one-layer LoRA exponential training and requested the next
two-layer condition. Only worker1660108 was interrupted with SIGINT during
epoch5; four completed epochs and the best selected epoch2 remain preserved.
Original GPU3 queue correctly stopped on returncode-2, so its intervening
one-layer cosine and old two-layer commands will not start automatically.
Original native Kev training worker1896176 onGPU2 remains untouched.

New worker2666657 onGPU3, root:
`/var/local/danefedov/experiments/agent-safety/runs/action-models-lora2-four-epochs-v1`.
Source: sibling `action-models-lora2-four-epochs-v1-launch/source`, a byte-identical
copy of original immutable training code. No changes to other experiments.

Fresh original frozen-start Qwen3.5-0.8B-Base with trainable rank16/alpha32 LoRA,
two Mamba layers + projection/head. Exponential token target, no ranking.
Four epochs; all other original data, admission, seed, optimizer and batching
settings retained. Not initialized from the interrupted one-layer checkpoint.
The original successful mamba2_lora CUDA gate is saved as inherited provenance;
normal training writes a fresh first-window gradient audit. No new dependencies.

15218train/848validation records reused read-only; no cache reconstruction.
Checkpoint selection minimizes original validation hard endpoint BCE. No
automatic additional arm or benchmark evaluation is attached to this training.
Existing15-model ATBench/Nemotron snapshots stay fixed and are not replaced.

Live/wait/copy `sh_14b42bcd9001MNuBKe80xl5hkO`, do not poll completion.
