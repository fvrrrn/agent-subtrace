# Conditional bounded follow-up v1

Authorization: user requested continuing experiments, with the former budget removed.
Exactly three new two-epoch cells; no automatic retries or extension beyond this round.
The existing eight-cell queue and its source are read-only.

## Admission and selection

Wait for the original launcher using Linux pidfd if still alive. Require all eight
cells and both epochs complete, frozen-weight audits passed, and matching 886
dev action identities. If the launcher is gone but the queue is incomplete, fail
closed. Never select from a partial queue. Rank configurations by selected dev AP,
then lower unweighted event BCE, then total measured cell seconds; lexical name
only breaks an otherwise exact tie. Save all input artifact hashes before launch.

Use only the winning **configuration**, never its trained weights. Every new cell
starts from the same original last_broad epoch-2 head as the eight-cell queue,
SHA-256 `7f3d0d3950313f2f7b39ab1b7a92b4933c3f072b885f0025ebfe42701a9b2137`.
Keep its adaptation, curve, adaptive/uniform weighting, seed/order, two epochs,
Instruct backbone, last-layer inputs and pinned native Mamba forward configs.

## Locked factorial

Positive weight off/on × event coefficient 0.1/1.0. Off/0.1 is the existing
selected configuration's control, leaving exactly three new conditions:
off/1.0, on/0.1, on/1.0. Both token and event losses remain simultaneous.

Using train_small only, let `u_d = unsafe_actions_d / labeled_actions_d`.
Compute `U = sum_d u_d`, `S = sum_d (1-u_d)`, and `r = S/U` without clipping.
CPU preparation obtained `U=230.0426533266006`, `S=1778.9573466733993`,
`r=7.733163050192017` from 2009 train dialogs (5006 safe / 641 unsafe actions).
This is the effective class ratio for mean-action, mean-dialog supervision,
not the raw action-count ratio. Preserve the original accumulation-of-four
algorithm, including its last partial window. Apply the same `pos_weight=r`
to token and event BCE. For soft token target y the loss is
`-r*y*log(sigmoid(z)) -(1-y)*log(1-sigmoid(z))`.
This weights the positive term, not an entire unsafe row; broad targets retain
their negative term. Adaptive token weights remain detached and normalized
within each action. Scores are not claimed to be calibrated probabilities.

## Gates and reporting

Inherit exact 301-dialog warm-start replay, both-loss gradient gates, loss support,
causal/reload error zero, longest-training-dialog backward and frozen-weight checks.
Keep all token logits, optimizer/RNG state, input IDs and source hashes.
Check GPU ownership before each new cell; never stop other processes.
Persist tqdm output and stream it through the shell. Preserve failures.

Report AP, trapezoidal PR-AUC, ROC-AUC, unweighted event BCE, P/R/F1/Accuracy,
FP/FN at 0.5, descriptive recall at FPR≤1%/5%, source slices and class score
quantiles. Compare ranking, not only a shifted operating point. AP selection
is unchanged. This is dev-selected post-hoc research, not held-out confirmation.
No confirm/external inputs or scores, threshold fitting, new data, annotations,
StepGuard, attention or full fine-tuning.
