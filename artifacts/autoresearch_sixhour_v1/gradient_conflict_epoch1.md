# Round1 epoch1 dev_select loss-gradient conflicts

CPU-only; 886 exact-matched events per arm. Event weight 0.1. No confirmation data read.

| Arm | Label | Events | Opposed/all tokens | Opposed/active | Mean cosine | Median .1 event/token norm | Dialog-reduced .1 event/token norm |
|---|---|---:|---:|---:|---:|---:|---:|
| broad_endpoint | all | 886 | 0.00035128 | 0.066591 | 0.097816 | 1.3621 | 1.3512 |
| broad_endpoint | safe | 739 | 0 | 0 | 0.11653 | 1.4274 | 1.5653 |
| broad_endpoint | unsafe | 147 | 0.0015184 | 0.40136 | 0.0037447 | 1.0776 | 1.0716 |
| broad_linear_softmax | all | 886 | 0.4249 | 0.4249 | 0.80491 | 0.17897 | 0.20452 |
| broad_linear_softmax | safe | 739 | 0.45019 | 0.45019 | 0.8922 | 0.19194 | 0.23164 |
| broad_linear_softmax | unsafe | 147 | 0.34087 | 0.34087 | 0.36611 | 0.12479 | 0.16938 |
| broad_null | all | 886 | 0.0013932 | 0.0013932 | 0.91401 | 0.14906 | 0.15105 |
| broad_null | safe | 739 | 0 | 0 | 0.97728 | 0.15326 | 0.16419 |
| broad_null | unsafe | 147 | 0.0060221 | 0.0060221 | 0.59594 | 0.11073 | 0.12366 |
| broad_leaky | all | 886 | 0.00051798 | 0.00051798 | 0.51181 | 0.10785 | 0.15609 |
| broad_leaky | safe | 739 | 0 | 0 | 0.53482 | 0.098705 | 0.10699 |
| broad_leaky | unsafe | 147 | 0.002239 | 0.002239 | 0.39612 | 0.20398 | 0.19576 |

## Interpretation safeguards

- Detached-output gradients, not parameter/shared-Mamba gradients or training-step gradients.
- Each arm evaluated on its own epoch1 logits; between-arm differences include learned trajectories.
- Per-event norms use native within-action token mean; dialog-reduced aggregates divide by action count.
- Zero gradients excluded from active-coordinate opposition; all-coordinate rates dilute endpoint conflicts.
- Autograd is float32 native readout/BCE on CPU; summary norms/products float64.
- NULL branch is separate; its larger dimension-level contribution does not identify shared-parameter conflict.
- Validation forward was saved before report failure; no confirmation data, thresholds or external labels used.
- Labels specify the two loss objectives only; targets are artificial, not token gold.
- Not causal evidence localization, calibrated risk, or proof event-weight 1 improves generalization.

NULL endpoint-branch gradients (signed, weighted and unweighted) are recorded separately per event.
Full summaries include NULL joint norms, negative-cosine counts, combined-gradient reversals, exact event identities and source/tensor hashes.
A weight-1 control multiplies the event term by ten relative to this report; effects on learned parameters cannot be inferred by rescaling detached logits.
Existing prespecified control is not selected or altered by this diagnostic. Raw source tensors are read-only.
