# Direct linear readout control for the validation-best curve

Waiting/control launcher PID2083427, GPU2. Permanent root, created after the
curve pipeline completes:
`/var/local/danefedov/experiments/agent-safety/runs/action-models-linear-probe-v1`.
Source/launch log: sibling `action-models-linear-probe-v1-launch`.
Code: `graph-safety/scripts/launch_curve_linear_probe.py`.

Frozen Qwen token features -> **Linear(1024,1)**, 1025 trainable parameters.
No Mamba, intermediate projection, LayerNorm or LoRA. Qwen itself retains its
own contextual/hybrid architecture. This removes the additional Mamba readout,
not all sequence processing from the system.

Choose the curve by the lowest **original validation endpoint BCE** among
exponential, cosine_rank, linear, quadratic, sqrt and cosine_plain. No StepGuard
scores are consulted for this choice; hard-token/endpoint controls are not curves
and are excluded from the selector. Lock the reference Mamba checkpoint and all
candidate validation selections before control training.

Use the same prepared training/validation records and Qwen cache, four epochs,
seed, head optimizer/lr, batching, target function and loss normalization. If
cosine_rank wins, preserve its unsafe-pair ranking term; otherwise no ranking.
Select the control checkpoint by original validation BCE, then evaluate on the
same cached StepGuard3017test/161dev and compare to the chosen Mamba, original Kev
and random-Mamba baseline. Save AP, trapezoidal PR-AUC, ROC-AUC and threshold0.5
metrics. No retraining of Qwen or external selection/calibration.

This is a strict **linear probe**, not a parameter-matched replacement for the
696985-parameter Mamba head; differences can reflect capacity as well as sequence
mixing. The control is exploratory, one seed, on a previously viewed benchmark.
The process blocks via pidfd until the existing curve/evaluation pipeline exits;
it does not poll or interfere with any running training or teacher service.
