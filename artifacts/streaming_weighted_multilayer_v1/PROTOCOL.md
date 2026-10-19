# Prospective protocol: weighted token BCE × feature depth × curve

Registered before GPU execution on 2026-10-18. Six new conditions, each two
epochs: last/three features × step/linear/broad. Same original 2009 train-small
dialogs and 301 dev-select dialogs (886 actions); no other evaluation inputs.
Frozen Qwen/Qwen3.5-4B revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a.
Old native token IDs, masks, action boundaries and whole-dialog admission retained;
no truncation. Mamba state persists through each dialog; one complete forward.

## Weighting registered from inspected official implementation

SingProbe repository revision `10edcb813482db08734e755f354390bf4d68a6b8`:
`trainers/loss.py`, BalancedMultiTaskLoss._compute_confidence_weights lines 741–762;
`configs/all_models/ling-3.0-flash.yaml` lines 110–117.
The class defaults differ from the shipped configuration. We explicitly adopt
the shipped configuration: T=1, tau=.35, confidence_weight_detach=true, beta
aggregator, no minimum-weight floor. No smoothing, attention, class weighting,
dynamic task weights, RL or distillation is added.

For each forward (one dialog), p=sigmoid(z) on all labeled current-action tokens,
beta=clip(population_std(p)/.35,0,1). For each action separately,
alpha=stop_gradient(softmax(beta*p/T)); L_token=sum(alpha*BCEWithLogits(z,target)).
Thus neither beta nor alpha carries gradient. Context, padding and unlabeled
tokens do not contribute to dispersion or normalization. Gradient accumulation
does not concatenate dialogs for beta: four independent dialog losses are averaged.
This adapts the official valid-batch dispersion to our batch size of one dialog.
Singleton dispersion is zero. No weights are used at inference.

Targets at position i in an action of N tokens, u=i/N:
step=Y; linear=Y*u; broad=Y*(.9+1.2u-1.5u²)/1.14.
All conditions simultaneously add .1*BCEWithLogits(z_last,Y).
Average both losses over actions, then four dialogs per optimizer step.
Targets and action length are supervision only, never future model inputs.
These scores describe causal-prefix prediction, not independently labeled token harm.

## Three-layer semantics and controlled initialization

The local pinned text config was read: 32 decoder layers, width 2560.
Choose hidden_states indices [11,22,32] prospectively (approximately thirds).
Indices 11/22 are outputs after decoder blocks 11/22 before the final model norm;
32 is the final normalized layer output, identical to last_hidden_state.
Cache admission checks these semantics against forward hooks and checks that
all 2310 deep components exactly equal the old cached last-layer tensors.
Concatenate in that order: LN7680→Linear7680→256→existing two residual Mamba→head.
Last-layer uses the unchanged LN2560→Linear2560→256 initialization.
Initialize the baseline head first, then replace only the three-layer input
norm/projection, preserving shared Mamba/readout initial values across conditions.
The wider projection has more parameters and a different initialization: disclose
this limitation, not a feature-only ablation with equal parameter count.

## Execution and checks

Seed 20261017; existing AdamW lr3e-4, wd.01, clip1, accumulation4; BF16 features
and autocast, FP32 head parameters. Two epochs each. Selection: dev-select AP,
ties lower hard endpoint BCE, then earliest epoch. Fixed threshold .5 only.
Report AP, trapezoidal PR-AUC, ROC-AUC, P/R/F1/Accuracy, FP/FN, score quantiles,
per-source and action-length slices. Retain all validation raw token/event logits,
IDs/spans, both checkpoints, optimizer/RNG states, source/cache/weight hashes,
actual lengths, environment and runtime. Training token logits are transient as
in the existing trainer; every epoch's complete dev-select token logits are saved.

CPU config JSON/Torch round-trip before head GPU execution. Strict full cache
identity/ID/shape/hash admission. Head CUDA gates: exact causality/reload,
separate token/event gradients through both Mamba blocks, readout and projection,
loss mask, actual longest cached-dialog backward. Backbone causal/hook gate and
post-cache frozen-weight hash. All long stages stream real tqdm through shell.

Read-only controls: unweighted Instruct A_step/A_linear/A_broad v3, same seed and
two epochs, descriptive comparisons. Last weighted versus last unweighted isolates
weighting within curve; three versus last within weighted conditions tests the
multilevel adaptation including its wider projection. No unweighted three-layer
arm: weighting×depth interaction is not separately estimable. Single-seed,
dev-selected evidence, not a held-out improvement or exact SingProbe reproduction.

Initial budget check unix1792341865: training deadline1792345185, overall
deadline1792348785 (unchanged one-hour reserve). Prior same-model cache217s and
two-epoch head90s make six conditions plausible. Launcher requires at least1500s
remaining and bounds every own child by the original deadline. GPU3 only and
only while unoccupied; no foreign processes changed. Failed artifacts preserved.
No new downloads, benchmarks, confirm/oldval, threshold fitting, or retry sweeps.
