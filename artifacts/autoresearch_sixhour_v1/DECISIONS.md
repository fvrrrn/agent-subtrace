# Autoresearch decisions (prospective criterion retained)

## Data

Frozen grouped source/origin holdout: small2009/full6058train dialogs,
301dev-select (886events/147unsafe),299dev-confirm (949events/181unsafe),
425old-validation. AgentDoG/AgentAlign and linked StepSafe domain held out.
Components/request/trajectory overlap audit passed; repackaging and incomplete
semantic independence disclosed. No labels used to choose the split/subset.

## First wave and preserved failure

`round1` gates passed and epoch1 inputs/logits/checkpoints retained, but a
reporting exception at the initial PR plotting endpoint prevented completion.
The endpoint has no specificity. Fixed `.get('specificity')` only for the
recall@FPR diagnostic; fresh immutable source/run `round1v2`, not history rewrite.
All4candidate reruns complete4epochs, exact886common dev-select events.

| Broad curve / frozen readout | Epoch | AP | Trapezoidal PR-AUC | ROC-AUC | Event BCE | Recall@.5 | Recall@FPR5% |
|---|---:|---:|---:|---:|---:|---:|---:|
| Endpoint | 2 | .4963 | .4960 | .7792 | .5025 | 1.36% | 32.65% |
| Linear-softmax MIL | 2 | .3873 | .3817 | .7633 | .4390 | 0% | 26.53% |
| NULL | 2 | .4425 | .4402 | .7659 | .5335 | .68% | 27.89% |
| Leaky32 | 4 | .4231 | .4193 | .7669 | .4286 | 11.56% | 28.57% |

AP is the prespecified selection criterion; do not select leaky just because
its uncalibrated.5 recall is higher. Endpoint is first-wave AP winner. Diagnostic
source slices also computed with exact event/state/ID coverage; AP is not solely
driven by all-safe AgentAlign. AgentDoG AP endpoint=.5074,MIL=.5073,NULL=.5343,
leaky=.4466; source subsets overlap and are not independent replications.

## Fixed follow-ups

Round2: endpoint event-weight1 (versus.1), multiscale mean-logit scan, and one
before-Mamba attention endpoint. No dense curve/readout/weight grid. Full-data
frozen follow-up predeclared for best round1+round2 config and MIL control (or
next best if MIL itself wins). Selection still dev-select AP, ties BCE.

Detached-output gradient analysis on failed epoch1 dev-select886events:
MIL opposing signs34.09%of unsafe token coordinates but only2/147unsafe
negative action cosines; endpoint59/147negative unsafe endpoint gradients.
Weighted event/token norm ratios endpoint1.351,MIL.205,leaky.156;
NULL token-only.151 / including NULL branch1.325. These are not shared-parameter
gradient conflicts or proof weight1will help; no confirm/test data consumed.

## One prompt experiment — close branch

One fixed safety instruction, native re-rendering;3034dialogs/8415actions,
zero exclusions. Full separate BF16 cache completed in378.08seconds, frozen
weight hash unchanged, never reused original features for modified inputs.
Four-epoch matched-small endpoint run selectedepoch1:
AP=.4732,PR-AUC=.4720,AUC=.7686,BCE=.4273; control selectedepoch2
AP=.4963,AUC=.7792,BCE=.5025. All886dev events matched. Safe/unsafe medians
.0489/.1259 versus original.0174/.0489: scale changed, ranking not improved.
**Do not search prompt variants or advance it based on BCE alone.**

## Representation adaptation

LoRA endpoint and MIL, two matched small-data runs,2epochs each, same broad
token curve and event-weight.1;GPU3/2. All-linear r16/alpha32/dropout0,
Qwen lr1e-5/head3e-4; non-reentrant checkpointing, native whole-dialog.
Runtime gates include causal IDs perturbation/reload, separate component
gradients, both Mamba/adapter gradients, masks, longest-training backward and
frozen-weights audit. No confirm labels in gates. Deadline reserves90minutes
for confirmation/external reporting and180seconds for persistence/audit.

No new external scoring or dev-confirm evaluation has occurred at this point.
All final claims must use locked checkpoints/thresholds and disclose the search.

## Second frozen wave — close unsuccessful branches

All3follow-ups completed4epochs. No OOD AP improvement over endpoint.4963:

| Candidate | Epoch | AP | Trapezoidal PR-AUC | ROC-AUC | Event BCE | P/R at.5 |
|---|---:|---:|---:|---:|---:|---|
| Event weight1, endpoint | 1 | .4895 | .4888 | .7883 | .7638 | 26.95% /82.31% |
| Normalized multiscale | 2 | .3785 | .3728 | .7642 | .4662 | undefined /0% |
| Attention before Mamba, endpoint | 2 | .4784 | .4776 | .7651 | .4656 | 100% /2.04% |

Weight1largely shifted both class scales and generated328FP/739safe at.5;
do not count high recall alone as success. No additional weight/scale/attention
grid. Full-data frozen endpoint plus MIL control remains the planned comparison.

## Conditional larger-data LoRA registration

After both small LoRA screens complete, full-data LoRA is predeclared only if
best dev-select AP exceeds first-wave frozen reference by at least.02.
One best readout, same broad curve/event-weight.1, fresh full-data initialization.
Use measured screen wall time×6058/2009×1.25 for conservative2epoch estimate;
run2or1epochs only if it fits before global deadline minus1hour. Otherwise close
branch or report budget skip, never silently overrun. Selection still devAP.
Source-space/case shifts are not assessed with confirm or external labels.

## Backbone screen outcomes / bounded follow-up

LoRA endpoint small2epochs completed2397.39seconds,selectedepoch1:
AP/PR-AUC/AUC=.52707/.52532/.80093, BCE=.41070;
P/R/F1/Acc@.5=51.37/51.02/51.19/83.86%,FP71/FN72.
LoRA MIL small2epochs completed2366.37seconds,selectedepoch2:
.50627/.50373/.79867,BCE=.38090,P/R/F1/Acc=75.00/16.33/26.82/85.21%,
FP8/FN123. Both frozen-weight audits passed; peakCUDA17.36GB.
Endpoint qualifies for the predeclared larger-data follow-up; MIL is not
advanced just for better BCE or Accuracy. Confirmation remains unopened.

One final8-layer-only LoRA endpoint small2epoch control registered onGPU2
after MIL completion: lower adapters remain zero-contribution/frozen, same
initialization,curve/readout/weight/data as all-layer control. This is a single
stability/plasticity regularization hypothesis, not an8/16/24layer grid or a
biological equivalence claim. Own immutable source, no existing run modified.

FullFT small2epochs onGPU0/1 is a distinct representation-capacity control:
FSDP FP32masters/Adam/buffers with BF16compute. Fifteen-minute preflight gate
budget and a90minute final reserve apply. No memory-fit claim from estimates.

## Rejected late feature-backbone suggestion

CPU audit established pinned Kev4B is our exact QwenBase plus a decision-task
LoRA, not a verified dedicated safety encoder. Standard direct HF/PEFT features
would be compatible with original tokenizer IDs, but copied tokenizer files
have inconsistencies/no chat template; parent training ancestry and dataset
licenses/benchmark overlap are not fully available locally. Do not claim
safety pretraining or clean held-out provenance. Given reencoding cost and weak
late-budget justification, no Kev-feature experiment/GPU job was launched.

## Final evaluation engineering audit (before any confirmation scores)

Static review found lock/source/deadline/runtime-settings and fullFT precision
issues, fixed before launching evaluation. Frozen evaluator runs exact ORIGINAL
snapshot via deadline wrapper, preserving its config/source hash check. All
head/adapter/backbone/protocol files are verified against final lock; threshold
reporting consumes hashed locked records rather than refitting unchecked files.
FullFT masters loadFP32 first; inference uses BF16parameter copies plus FP32
buffers and BF16autocast to match FSDP. Exact all-token dev replay must pass
before confirmation; precision incompatibility is a failure, not a tolerance
relaxation. Corrected a runner indentation error found in review. CPU imports/
syntax for13final-stage modules passed; no GPU/runtime-eval claim from this.
