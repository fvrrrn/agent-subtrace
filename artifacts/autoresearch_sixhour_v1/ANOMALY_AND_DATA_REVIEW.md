# Steering: subsequences, data quality and evaluation

User requested analysis during autoresearch: view token-risk sequences as
time-series anomaly detection; examine subsequences with classification; question
data quality, metric choice and ATBench interpretation. This is a diagnostic
direction, not permission to restart annotation, change locked selection after
seeing results, or start an unbudgeted training grid.

## Conditional subsequence detection

Anomaly is not equivalent to unsafe: rare authorized actions can be safe and
routine-looking unauthorized actions unsafe. Model a context-conditional unsafe
subsequence, not unconditional rarity or reconstruction error alone.

Retain causal full-dialog encoding/state and simultaneous token/event losses.
A candidate interval readout can inspect an action's token-risk subsequences
conditioned on visible history, permissions and provenance. Interval extraction
does not justify cropping away authorizations or remote dependencies from input.
At each decision only already visible tokens are eligible; no future-centered
windows, gold-selected evidence intervals or future action-length features.

Existing leaky/multiscale experiments already tested simple scalar smoothing
and scanning without OOD AP improvement. They do NOT rule out learned conditional
subsequence classification, but another max/mean pooling grid is poorly justified.
Event labels alone do not identify true witness intervals. Broad positive token
targets may conflict with sparse-witness semantics; explicitly separate the
prefix-risk curve from any latent evidence channel, without dropping token loss.

## Observed data facts and alternative explanations

From fixed data/audit.json: full train has6058dialogs,16877events,2245unsafe;
source-positive attributions: StepSafe1980,StepShield246,LLM_contrastive_v2 19.
Source dialog attributions: StepShield3593,StepSafe2406,LLM_contrastive_v2 59.
These are provenance counts, not verified independent semantic mechanisms;
source attributions may overlap. More rows mostly expand existing sources.
Original grouped components:449small versus1411full, not2009/6058independent
dialogs. Dev-select also has119all-safe AgentAlign events; aggregate AP can
benefit from source recognition. Within-source reports are therefore essential.

Full frozen endpoint AP=.45125 versus small=.49634; this single-seed,
fixed-epoch comparison is not a controlled data scaling law. Full has more
optimizer steps and different source/family mix. Noise, synthetic artifacts,
inconsistent action/gold semantics, optimization and domain shift remain
alternative explanations; no finding yet establishes corrupt data.

Prioritize audits of actual current-action serialization and original labels:
unsafe request versus unsafe execution; refusal/quotation versus execution;
trajectory-level harm versus action-level safety; permissions and missing
context; source-specific artifacts; component redundancy; admission exclusions.
Audit samples must include correct predictions as well as errors and multiple
sources, not only cherry-picked model disagreements. No labels changed silently.

## Metrics / ATBench

Keep prospective dev-select AP criterion for this cycle. Report trapezoidal
PR-AUC and ROC-AUC separately, source slices/prevalence, operating-point
precision/recall/F1/Accuracy/confusions and recall at fixed FPR. Source-macro AP
cannot silently assign zero to all-safe slices; report their FPR separately.
For future source-macro protocols disclose undefined slices and group weighting.

ATBench is trajectory-gold, while our evaluator causally classifies proposed
actions and takes maximum INCLUDING Complete. Completed-trace retrospective
judgment, evidence of attack, unsafe current action and realized harm are not
interchangeable tasks. Max score depends on action count/case length and can
trigger on safe quotation/refusal/final summaries. Previously observed Complete
sensitivity warrants a task/label audit, not removal of Complete to improve AP.
Our admitted subset981/1000 excludes19unsafe long cases; preserve exclusions
and disclose selection effects. Existing metrics cannot prove labels invalid.
ATBench has already been viewed repeatedly: do not use it for adaptive model,
threshold, case-filter or task-definition selection and claim held-out evidence.

Desired follow-up report: separate model error, label ambiguity, serialization
error and genuine protocol mismatch; keep original predictions and gold, and
present any alternative interpretation strictly as post-hoc sensitivity.

## Additional backbone hypothesis (explicit user steering)

User explicitly requested adding a different backbone and continuing autoresearch.
Compare the current4BBase with one actually available alternative, prioritizing
locally cached Qwen3.5-9B-Base if its pinned snapshot is complete and a bounded
small-data probe fits. This distinguishes representation capacity from head
pooling; it does not establish a different-family or instruction-tuning effect.
An instruction-tuned/different-family backbone is a separate future hypothesis,
not interchangeable with this size control. Kev adapters are not established
safety-pretrained backbones (see previous provenance audit).

Subsequent user steering explicitly adds a smaller backbone, not only9B:
locally cached Qwen3.5-0.8B-Base@dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68
is the lightweight candidate, with priority over9B if only one new encoding
fits the remaining budget. The old0.8B prefix-training experiments are NOT this
new matched whole-dialog/token+event comparison; their caches are incompatible.
This is a size/representation control within the same family, not a claim of
cross-family robustness. All native-input/cache/causal/admission rules below
apply equally to0.8B. Neither candidate has been launched by this note.

Mandatory: native tokenizer/template re-rendering, exact current-action causal
spans, newly encoded cache with actual hidden dimension, independent identity
and hashes; no original4B feature reuse. Retain fixed split/group membership,
same token+event objectives and matched fully admitted cases. Any new overflows
or serialization failures must be disclosed before comparisons, never silently
truncated/dropped. Preserve the original six-hour deadline and final-evaluation
reserve; feasibility/preparation is not proof that GPU inference fits or launch
permission for an uncontrolled model grid. No download/install required unless
separately justified within the authorized scope and budget.
