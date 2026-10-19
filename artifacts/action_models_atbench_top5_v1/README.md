# User-corrected best-five evaluation scope

The previous15-model queue was stopped by explicit user correction; preserved
as `runs/external-replay-v1-launch/STOPPED_FOR_TOP5.json`. No training changed.
ATBench primary/sensitivity, inputs, checkpoint selections and threshold0.5
are unchanged; see `../action_models_atbench_v1/README.md` for protocol details.

Only five fixed models: LoRA validation-best epoch2, Mamba quadratic,
Mamba cosine_rank, Mamba linear, original Kev. This shortlist uses previous
validation/StepGuard results, not the new ATBench or Nemotron metrics.
Three completed Mamba action outputs and the finished frozen-Qwen cache are
reused with selection/head/input-hash verification. The old interrupted native
Kev partial file remains preserved, but is not treated as a completed model.

New root `runs/action-models-atbench-top5-v1`, source/launch
`runs/external-replay-top5-v1-launch/source`, queue PID2792288 onGPU2.
LoRA2 and original Kev remain to be fully evaluated. Then the same five models
run on the already prepared restricted Nemotron final-report task.
Initial PID2784790 exited before model work because argparse rejected the
separate `--suffix -top5` argument. Its launch/log are preserved; corrected
invocation is `--suffix=-top5`, no inference was repeated by that retry.
Shared single live/wait/copy `sh_14b55119f0019EI7fOslM9r07p`; do not poll.
