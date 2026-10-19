# GLiClass on R-Judge

Source revision: `83ce301da3ad50dd8b397e772863f5411c3d3dc2`.
Selected 571 / 571 records; completed 569.
LINK scoring: `normalized`; threshold 0.2.

| Method | TP | FP | TN | FN | F1 | Recall | Specificity | Balanced accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| graph_pipeline | 300 | 269 | 0 | 0 | 0.690 | 1.000 | 0.000 | 0.500 |
| full_prefix_pipeline | 300 | 269 | 0 | 0 | 0.690 | 1.000 | 0.000 | 0.500 |
| full_trajectory | 300 | 269 | 0 | 0 | 0.690 | 1.000 | 0.000 | 0.500 |

- graph_pipeline: classify each proposed action using predicted ancestors; maximum score over agent actions.
- full_prefix_pipeline: identical action-level questions and maximum, but complete causal prefixes.
- full_trajectory: single standard post-hoc safety classification of the entire recorded interaction.

Only trajectory labels are available. No per-step risk, edge accuracy, early-detection or prevented-harm claims.
Max-over-steps is a decision rule, not a calibrated probability. No thresholds/prompts fitted on these labels.
Raw exact calls and token audits: calls.jsonl. Record-level graph and scores: records.jsonl.
Errors/overlength inputs are listed in summary.json; they are not silently truncated or scored as successes.

## Outcome and coverage audit (no additional inference)

All three methods classify **every completed record as unsafe**. Thus recall
100% and F1 69.0% are not useful discrimination: specificity is 0%, balanced
accuracy 50%, and all 269 safe records are false positives. The graph provides
no benefit in this run. This is a degenerate frozen-model/prompt result, not
evidence that every graph approach fails.

The pinned repository contains **571 raw records**, despite the paper's 569.
We attempted all 571, with no label filtering or deduplication. Two adapter
errors occurred because environment observations precede the first agent event:
`data/IoT/household.json:13:id=46` (label 1) and
`data/IoT/household.json:14:id=47` (label 0). These are format-adaptation failures,
not model-context overflows. They are excluded from the common 569-record metric
denominator and preserved as failures. The matching number 569 is coincidental;
we do not claim exactly the paper's curated record set was evaluated.

4,359 model calls, 242.67 seconds of replay (~4 minutes). Maximum actually encoded
input 1,939 tokens, below 8,192. All 3,475 successful safety calls had normalized
VIOLATION score >=0.5201 (maximum ~0.999997), including the direct full-trajectory
baseline. Therefore all-positive trajectory output is not explained only by the
pipelines' max-over-actions aggregation.

Per the user's instruction, no retry of failed records, threshold changes,
prompt tuning, native-score ablation or second benchmark run was performed.
