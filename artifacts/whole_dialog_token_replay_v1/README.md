# Token readout replay: frozen and LoRA Qwen4B + Mamba

Context: [../../../AGENTS.md](../../../AGENTS.md), §7.19.
Entry point: `../../scripts/evaluate_whole_dialog_tokens.py`.

User-authorized inference only, no training or new checkpoint/temperature choices.
The parent `whole-dialog-benchmarks-v1` locked checkpoints are reused:
frozen linear±ranking epoch2, LoRA linear±ranking epoch1. Kev is not rerun.

## Fixed inputs and action spans

Exactly the parent's Dojo1220/Harm731/StepGuard3178/ATBench981/R-Judge571
successful cases,10955 actions. The19 original long ATBench exclusions remain;
there is no new truncation or window. Actual native-template token IDs and
whole-dialog/causal-prefix fallback modes are unchanged.

CPU preparation reconstructs each current assistant action body from the
saved native template, excludes role prefix/empty-think prefix and final im_end,
and verifies every endpoint and prefix token hash against the parent sequence.
Native tool-call serialization remains part of the action body, as in training.
No history or padding tokens enter action aggregations. No action/token gold
is fabricated from trajectory labels.

Action-token counts: Dojo110986, Harm72385, StepGuard329766, ATBench416363,
R-Judge156190. Every complete input's token IDs and raw token logits are saved
in a per-case `.pt` file, with action spans and sequence identities. Sigmoid
scores can be reconstructed from the float32 logits without reencoding.

## Four prespecified aggregations

1. `endpoint`: last token, parent control.
2. `token_max`: maximum probability within the action.
3. `top10_mean`: mean of the largest ceil(10%×action_length) probabilities.
4. `tail8_mean`: mean of the last min(8,action_length) probabilities.

For trajectory datasets, the case score is maximum over event aggregates.
All-events inclComplete remains primary; tool-only and matched all-events
are sensitivity checks. StepGuard dev/test and Harm strict/loose stay separate.

Each replayed endpoint must match its saved parent: absolute probability error
≤1e-6 and raw-logit error≤1e-5, otherwise stop rather than silently change the
baseline. Actual maximum errors are recorded. OOM or any failure stops its queue;
no automatic retries, replacements or additional aggregation grids.

Metrics for every model×aggregation: AP, trapezoidal PR-AUC, ROC-AUC,0.5
operating point and explicitly post-hoc max-min(P,R) operating point, recall at
FPR≤1% and≤5%, length slices. The latter curves/thresholds are diagnostics on
the already viewed data, not held-out calibrated operating points. Full-token
curves are not token-safety accuracy because token gold is unavailable.

## Two independent queues

- **GPU2, PID4032281:** frozen linear then frozen linear+ranking; only Mamba/head
  passes over the complete parent BF16 feature cache, no Qwen load or encoding.
  Analysis and preliminary notification after both finish.
- **GPU3, PID4032282:** LoRA linear then LoRA linear+ranking; adapted Qwen
  reencodes online, followed by its own group analysis.

Server root `runs/whole-dialog-token-replay-v1`; immutable source sibling
`whole-dialog-token-replay-v1-launch/source`. Parent inputs/cache/weights are
read-only, teachers/annotation are unchanged. Raw token tensors stay on the
server; summaries, prediction aggregates, masks, audits and source copy locally.

Frozen live/wait/copy: `sh_14dc774fb001jDiPCgfsdTo7fk`.
LoRA live/wait/copy: `sh_14dc7752a001MvCABnJcgfQDDv`.
Do not poll completion; preliminary conclusions must not await the LoRA queue.
