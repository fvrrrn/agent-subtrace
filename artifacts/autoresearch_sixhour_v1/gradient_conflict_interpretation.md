# Empirical interpretation: round1 epoch1 loss-gradient conflicts

Completed CPU-only on **all 301 dev_select dialogues / 886 actions per condition**:
739 safe, 147 unsafe. Four arms matched exactly on event identity, label, spans,
input hashes and action token IDs. No confirmation inputs or external labels
were read. Saved raw tensors were not modified. No model forward was run.

Native `event_logit` and broad `profile` were AST-extracted from the immutable
`runs/autoresearch-sixhour-round1v2-launch/source/scripts` snapshot. Only these
pure functions were executed, avoiding module-level dependencies/CUDA setup.
CPU float32 autograd reproduced each per-action objective; products/norms were
summarized in float64. Event weight was 0.1 throughout this diagnostic.

## Main observations

| Condition | Safe opposed token fraction | Unsafe opposed token fraction | Unsafe mean action cosine | Negative-cosine unsafe actions | .1 event/token norm, dialog-reduced |
|---|---:|---:|---:|---:|---:|
| Endpoint | 0% | 0.152% of all / 40.14% of active | .00374 | 59/147 | 1.351 |
| Linear-softmax MIL | 45.02% | 34.09% | .36611 | 2/147 | .205 |
| NULL token branch | 0% | .602% | .59594 | 0/147 | .151 |
| Leaky | 0% | .224% | .39612 | 0/147 | .156 |

“Active” means both gradients are numerically nonzero. Endpoint event BCE only
touches the final token, so its all-token opposition rate hides 59 conflicting
unsafe endpoints. Its dense-token versus sparse-endpoint mean cosine near zero
does **not** itself mean the objectives are antagonistic everywhere.

MIL has many sign-opposed coordinates, including safe actions: native
linear-softmax can push low token activations up on a negative bag. However,
the magnitude-weighted geometry is mostly aligned: safe mean cosine .8922,
unsafe .3661; only two unsafe actions have negative cosine. Just 0.129% of all
coordinates have their token-gradient direction reversed by the **combined**
token + .1 event gradient. Counting sign opposition alone exaggerates conflict
when many opposed coordinates carry little gradient magnitude.

NULL and leaky show little sign opposition and no negative action cosines on
this snapshot. That does not establish correct localization or OOD safety:
artificial token objectives can agree with event gradients while both learn
nontransferable shortcuts. It also does not contradict future-length target
incompatibility, which compares identical prefixes across alternative futures,
not merely the two losses on one realized example.

## NULL branch must not disappear from the accounting

The .151 NULL ratio above covers only its gradient into token logits. Its
additional endpoint NULL-logit branch has weighted dialog-reduced norm **.39850**.
Concatenating token-logit and NULL-logit event gradients gives a joint weighted
event/token norm ratio **1.3255** (safe 1.3958, unsafe 1.1872), versus .1511 for
the token branch alone. The token objective has zero direct NULL-logit gradient.
This is output-space accounting, not an assertion that the NULL head dominates
Mamba parameter updates. Signed per-event NULL gradients and their weighted
values are preserved in the JSONL.

## Meaning for the already prespecified event-weight-1 control

The nominal coefficient .1 does **not** imply that event gradients have one tenth
the output-space norm. Endpoint already has a concatenated dialog-reduced ratio
1.351; MIL/leaky token-branch ratios are .205/.156. Changing .1 to 1 multiplies
the fixed-logit event contribution by ten: approximately 2.05/1.56 for MIL/leaky,
and 13.51 for endpoint. NULL's corresponding joint ratio becomes 13.25.

This makes the existing control a meaningful objective-balance probe, **not**
evidence that weight 1 will improve generalization. No control was selected,
added, changed or launched using these results. Its eventual learned logits,
shared-parameter gradients and optimization trajectory need not follow these
fixed-snapshot ratios. Both losses remain simultaneous.

## Scope and artifacts

- `graph-safety/scripts/analyze_autoresearch_conflicts.py`: analysis program.
- `gradient_conflict_epoch1.md`: safe/unsafe/all summary table.
- `gradient_conflict_epoch1.json`: full summaries, native/analyzer hashes,
  environment, exact matching audit and all 1,204 input tensor file hashes.
- `gradient_conflict_epoch1.jsonl`: 3,544 per-action records with exact event IDs,
  source paths/hashes, spans/input hashes, losses, cosine, norms, sign counts,
  combined-gradient reversals and separate NULL gradients. No raw context.

Dialog-reduced aggregates divide each action's gradients by the number of labeled
actions in its dialogue, as native loss averaging does, then concatenate output
coordinates. Per-action ratios and cosine are unaffected by this common factor.
This does not reconstruct accumulation across shared parameters or training
steps. Each arm uses its **own learned epoch1 logits**; differences therefore
combine readout algebra and different training trajectories. The original run's
report failure does not make this a completed four-epoch result or an independent
test. This is development-set objective diagnostics, not threshold fitting,
causal witness validation, or a calibrated-risk assessment.
