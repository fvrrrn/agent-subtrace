# Weighted broad LoRA and continuation ablation

**Current: v3 is training on GPU 3.** Strict all-token replay passed on 301/301
dev dialogs, both-loss gradient gates passed, causal/reload errors are zero,
longest backward passed at 15322 tokens. Training tqdm reached 8/2009 dialogs
in epoch 1 (two optimizer steps). Launcher PID 4134159, child PID 4134236.
Eight locked cells remain in the finite sequential queue, two epochs each.
Completion and quality metrics are not yet available.

Root cause and controlled evidence: [REPLAY_DIAGNOSIS.md](REPLAY_DIAGNOSIS.md).
Native Triton autotuner config selection was process-dependent; controlled A/B/A
reproduced/restored the drift. v3 fixes five native forward configs from an
observed exact replay. No tolerance, precision or original reference changes.

Active live shell: `sh_1502814c5001eJlCibL984wYIw`.
Training-start watcher `sh_15028572b001TqcDGuQ14yo95i` completed successfully.
Completion pidfd/report/copy: `sh_150289d29001c36an4IRUwXA57`.
Server run: `runs/weighted-broad-lora-ablation-v3`; immutable source in sibling
`weighted-broad-lora-ablation-v3-launch/source`. Current gate/protocol JSON is
copied locally to `run_v3/`. The completion watcher will add the final report.

The eight-cell protocol is in [PROTOCOL.md](PROTOCOL.md). All cells are locked
before adapted metrics, use the same selected head, and run two epochs each.
GPU 3 is checked for existing compute processes before every cell. The queue
starts with LoRA/broad/adaptive; no other user's process is stopped.

## Launch history

- `runs/weighted-broad-lora-ablation-v1`: failed before training at exact online
  warm-start replay, third dev dialog
  `54b1e81b27a2cb2fb0082149eac0852cfd490533fceda85f13a6b941e7eff92f`.
  First two matched. Source, logs, failure and frozen hash audit are preserved;
  local copy is `failed_v1/`. Isolated and five-case diagnostic replays matched
  both original hidden features and saved logits exactly. The original failure's
  numeric size was not saved; its cause is not established by those diagnostics.
- `runs/weighted-broad-lora-ablation-v2`: separate immutable sibling
  `weighted-broad-lora-ablation-v2-launch/source`. Explicitly zeros backbone
  padding before Mamba to match frozen cache representation. This removes an
  input mismatch; it is not yet proof of the v1 failure's cause. Exact replay
  assertion remains unchanged, and any failure now saves actual/reference logits.

CPU syntax/import checks passed in the installed server environment; all 2310
IDs and JSON/Torch config round-trip are checked before model loading per cell.
No dependencies, tests or formatters were run. No learned parameters were reset.

## Processes and notifications

**Historical v2 status: blocked before training.** v2 also failed exact warm-start
replay, now on the first dev dialog `df0779adc80555c72c39d0a5267fd28cd0ab6a014f0bfda0553162dc43484b85`.
Four of 5816 raw token logits differ, maximum absolute difference 0.015625.
Actual/reference tensors are preserved in
`run_v2/lora_broad_adaptive/warmstart_mismatch.pt`. Zeroing padding did not
resolve the issue. Both launchers exited; no optimizer step occurred in either
version. No gate tolerance was weakened. GPU 3 has been released by our runs.
The five-dialog isolated diagnostic had zero hidden/logit error, so the source
of process-dependent numerical divergence remains unresolved. All three v2
watchers below have completed; none is still awaiting training.

Live v2 shell: `sh_150122c5b001ivDLhi57uMKAbm`.
Training-start inotify/pidfd notification: `sh_1501267ab001SXJcjuWdYafcZD`.
Launcher pidfd/report/copy notification: `sh_1501296bd001JcDF71WwU4O3XP`.
These handles describe launched watchers, not completed training.

The completion watcher builds `comparison.json` / `COMPARISON.md` only when all
eight cells complete, then copies artifacts to `run_v2/`. Heavy token tensors,
optimizer/RNG state and gate reload snapshots remain on the server. The report
checks all 886 action identities and gives all three matched factorial contrasts.

Main command (from the immutable source with the existing Kev environment):
`python -u launch_weighted_broad_lora_v1.py`.
Do not rerun against an existing output root; launcher intentionally refuses it.
