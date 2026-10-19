# Causal supervision research and measured diagnosis — 2026-10-19

## Previous 8+3 are complete; class weighting did not improve primary AP

All rows use 886 dev actions, 147 unsafe; selected epoch 1. They are one-seed,
dev-selected continuation results from an already trained/selected broad head.

| LoRA step uniform | AP | trap PR-AUC | ROC-AUC | P/R/F1/Acc at .5 (%) | FP/FN | R@FPR≤1% |
|---|---:|---:|---:|---|---|---:|
| control, event .1 | .53307 | .53158 | .81041 | 50.68/50.34/50.51/83.63 | 72/73 | 21.09% |
| no pos-weight, event 1 | .51993 | .51841 | .81636 | 47.10/49.66/48.34/82.39 | 82/74 | 15.65% |
| pos-weight, event .1 | .51217 | .51242 | .81308 | 32.54/74.83/45.36/70.09 | 228/37 | 18.37% |
| pos-weight, event 1 | .52302 | .52089 | .82778 | 34.55/77.55/47.80/71.90 | 216/33 | 19.05% |

Higher recall from positive weighting comes with substantially more false alarms,
lower AP and lower low-FPR recall than the control. Unweighted event BCE respectively
.40461/.40008/.68883/.74746. No held-out generalization or calibrated probability claim.

## Primary sources and code, pinned

1. Wang et al., **A Comparison of Five MIL Pooling Functions for Sound Event
   Detection with Weak Labeling**, [1810.09050v3](https://arxiv.org/html/1810.09050v3),
   sections 1, 2.1–2.2 and 4 read. Bag presence and frame localization are different
   objectives. Under SMI, a positive bag needs at least one positive instance;
   average pooling can raise irrelevant frames, attention can hide frame false
   positives behind small weights. Their linear-softmax derivative changes sign
   at half the pooled score. This analysis does **not** make our causal prefix
   scores independent token-local evidence. Our prior MIL runs already failed to
   solve transfer; no new pooling sweep is implied.
2. Yu et al., **Gradient Surgery for Multi-Task Learning**,
   [2001.06782v4](https://arxiv.org/abs/2001.06782v4). Official code inspected:
   [`PCGrad_tf.py@c5fbd7c856526373828074f06875230f7f3ee79e`](https://github.com/tianheyu927/PCGrad/blob/c5fbd7c856526373828074f06875230f7f3ee79e/PCGrad_tf.py).
   It computes separate flattened **parameter** gradients and projects a negative
   inner product component. Our endpoint-logit sign diagnostic below is cheaper
   and narrower; it cannot establish a harmful parameter-gradient conflict or
   justify adding PCGrad. The inspected implementation even calls `tf.random.shuffle`
   without assigning its result; do not blindly copy implementation assumptions.
3. Lightman et al., **Let's Verify Step by Step**,
   [2305.20050v1](https://arxiv.org/abs/2305.20050v1), and official PRM800K release
   [`7ecc794703b2877f63226f2477a49b34f9b25163`](https://github.com/openai/prm800k/tree/7ecc794703b2877f63226f2477a49b34f9b25163).
   Explicit step correctness differs from final outcome supervision. It supports
   respecting our action-level labels rather than broadcasting trajectory harm
   backward. It supplies neither safety token spans nor proof that fabricated
   broad/step token targets are human process labels. No PRM data imported.
4. Arjona-Medina et al., **RUDDER: Return Decomposition for Delayed Rewards**,
   [1806.07857v3](https://arxiv.org/abs/1806.07857v3), abstract/scope checked.
   History-sensitive contributions to a delayed return are not current-action
   safety probabilities. Return-equivalent reward redistribution in an MDP does
   not justify risk monotonicity or assigning an eventual harm label to every
   earlier safe action. No RL/return-decomposition experiment is launched.

## Measured CPU diagnosis

Exact inputs, IDs and summaries: `diagnostics/`; server run
`runs/fresh-factorial-research-v1`. Only train_small and dev_select were loaded.
No new inference, optimizer operation, threshold fitting or external labels.

- Train: 2009 dialogs, 5006 safe / 641 unsafe actions. Under mean-action then
  mean-dialog loss, unsafe mass is **11.4506%**, vs raw action prevalence 11.35%.
  Effective ratio 7.733163 matches the previous prospectively computed weight.
  Dev: 739 safe / 147 unsafe; effective unsafe mass 13.6331% vs action prevalence
  16.5914%. This is reduction/sample weighting, not evidence of label corruption.
  Per-action training mass range is 9.7600e-6–4.9776e-4 in both classes;
  safe median 7.1109e-5, unsafe median 9.9552e-5. Step-control false negatives
  account for 7.0519% of total dev dialog/action mass (73 actions), versus
  3.4385% for positive/event1 (33 actions); the latter still incurs 216 FP.
- Exact full causal **endpoint** conflicts: **3 groups / 6 train actions**, none
  in dev. These reproduce the already documented small-train subset of 19 groups/
  39 full-train actions in `autoresearch_sixhour_v1/DATA_SEMANTICS_REVIEW.md`.
  Their total effective train mass is about **0.0747%**. No new pipeline defect
  was found and no labels/data changed; retain the limitation in comparisons.
- Identical causal input at the **first action token** has both terminal labels
  in **220 train groups**. This is unresolved future/branch ambiguity, not 220
  mislabeled endpoints. Hard step and broad priors cannot simultaneously satisfy
  both targets at that same prefix. The classifier must learn conditional
  uncertainty; these targets must not be called literal token safety truth.
- **12** train groups of identical action-only tokens have differing labels.
  History can distinguish these, so action-only disagreement is not automatically
  a contradiction. No such groups in dev does not prove history is unnecessary.
- Best step control misses 73 unsafe actions. Length 33–128: 23/32 missed,
  recall 28.13%, AP .36720. Length >128: 50/115 missed, recall 56.52%, AP .57518.
  History ≤1024: 14/16 missed; >1024: 59/131 missed. Length/source confounding and
  tiny strata prevent causal attribution to memory quality or loss strength.
  Source unions overlap (e.g. AgentDoG/StepSafe) and are not disjoint provenance.
- Analytic endpoint-logit gradients oppose for **75/147 unsafe broad-control
  actions**, versus **0** for step control and positive-weight step. Broad target
  at the endpoint is .5263, so at larger scores token BCE pushes down while hard
  event BCE pushes up. Token contribution scales by 1/N, event by its coefficient;
  this does not prove a large net harmful effect, and step still has low recall.
  These are output-space derivatives, not shared-parameter gradients or localization.

## Next decision, constrained by evidence

Do not chase recall-only gains or repeat class-weight/threshold searches. Existing
eight-cell loss contrasts all inherit the same dev-selected weighted-broad head;
they measure continuation effects, not learning from matched random initialization.
Run the requested finite **fresh shared-init 2×2** with unchanged data/architecture.
See `PROTOCOL.md` for coefficient selection (1.0 chosen only among prior three,
despite the control being better), scope, exact gates and interpretation. This
removes the within-round warm-start confound, but does not promise to repair the
weak token targets or match the previous larger cumulative training exposure.
