# Stopped FullFT: matched dev-select diagnosis

## Conclusion

**The poor FullFT AP is reproduced, not a metric/readout mismatch.** FullFT
combines a much more aggressive 0.5 operating point with weaker top-of-ranking
discrimination. Its failure is not explained by uniformly depressed endpoints.
The cause in training is not identified by these stored-output diagnostics.

Completed CPU-only on llm01, with CUDA disabled and one Torch/OMP/OpenBLAS thread.
Authoritative output: `run_20261018_v3/`. No model forward, training, annotation,
confirm/external evaluation, threshold fitting, tests, or lint was performed.

## Exact matched population and numerical checks

- All four conditions contain exactly **886 identical action keys**, **147 unsafe**,
  from **301 dialogs**; no exclusions, duplicates, or case intersection filtering.
- Labels, dialog IDs, state hashes, input-ID hashes, and action spans match the
  dev-select data and manifest. Original action attribution comes from
  `action_metadata_audit.json`, rather than potentially misleading dialog sources.
- **All 3544 endpoints** (886 × 4) equal the corresponding saved raw token logit
  exactly. FullFT/LoRA store `raw_logits`; frozen files store `raw_token_logits`.
- CPU sigmoid versus saved score differs by at most **1.1920929e-7** for
  FullFT/LoRA and **2.9802322e-8** for frozen conditions. These tiny differences
  do not explain the quality gap. CPU arithmetic need not be bit-identical to
  saved accelerator sigmoid.
- FullFT/LoRA token artifacts retain ID hashes, not literal IDs. Frozen token
  files additionally match the literal IDs in dev-select exactly. This audit
  establishes stored identity consistency, not an independent inference replay.
- Independently implemented tie-preserving AP, trapezoidal PR-AUC, ROC-AUC and
  confusion counts reproduce the recorded metrics to floating-point precision.
  BCE recomputed from probabilities differs from saved logit BCE by <1e-8.

## Aggregate results

FullFT and LoRA use epoch 1. Both frozen conditions select epoch 2. Threshold
is fixed at **0.5** throughout; no alternative locked threshold was supplied.

| Model | AP | Trapezoidal PR-AUC | ROC-AUC | BCE | P | R | F1 | Accuracy | FP / FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FullFT | .42209 | .42145 | .79076 | .59280 | 33.23% | 72.79% | 45.63% | 71.22% | 215 / 40 |
| LoRA | .52707 | .52532 | .80093 | .41070 | 51.37% | 51.02% | 51.19% | 83.86% | 71 / 72 |
| Frozen A | .50919 | .50879 | .79414 | .48854 | 100% | 2.04% | 4.00% | 83.75% | 0 / 144 |
| Original frozen | .49634 | .49600 | .77923 | .50250 | 100% | 1.36% | 2.68% | 83.63% | 0 / 145 |

FullFT FPR is **29.09%**, versus **9.61%** for LoRA. Relative to LoRA,
194 decisions change: FullFT is uniquely correct on 41, LoRA on 153.
FullFT gains 32 true positives but adds 144 false positives, net 112 more errors.
Against frozen A, FullFT gains 104 true positives but adds 215 false positives.
The frozen models' near-zero recall at 0.5 must not be hidden by their high precision.

## Ranking versus operating-point shift

| Model | Recall at FPR ≤1% | Recall at FPR ≤5% | Safe median / q75 / q90 | Unsafe median |
|---|---:|---:|---|---:|
| FullFT | 4.76% | 26.53% | .03677 / .59620 / .82033 | .80807 |
| LoRA | 19.05% | 34.01% | .01826 / .09535 / .47678 | .50867 |
| Frozen A | 21.77% | 32.65% | .01640 / .03162 / .08059 | .05582 |
| Original frozen | 17.01% | 32.65% | .01744 / .03410 / .06026 | .04886 |

Relative to LoRA, FullFT mean scores increase by .15322 on safe actions and
.19964 on unsafe actions. This is a substantial scale/operating-point shift.
But AP and low-FPR recall also deteriorate: **a global monotone recalibration
cannot repair ranking**, although it could change decisions at 0.5. No calibration
or threshold optimization was performed. Similar overall ROC-AUC does not imply
similar quality in the high-score tail or at low false-positive rates.

## Where the deficit occurs

Original source/domain slices overlap and must not be summed. Complete per-slice
P/R/FPR/AP/ROC-AUC, confusion matrices, class quantiles, and exact keys are in JSON.

- **AgentDoG:** 394 actions / 72 unsafe. FullFT versus LoRA AP .43261 vs .61099;
  ROC-AUC .74666 vs .80564; FP 90 vs 24; FPR 27.95% vs 7.45%.
- **AgentDoG FineGrainedTaxonomy:** 238 / 56. AP .40981 vs .63834;
  ROC-AUC .65959 vs .78679; FP 81 vs 20; FPR 44.51% vs 10.99%.
- **AgentDoG BinarySafety:** 156 / 16. FullFT actually has higher AP
  (.60884 vs .53165), despite more FP (9 vs 4). Degradation is not uniform.
- **StepSafe:** 547 / 99. AP .46858 vs .54991, but ROC-AUC .81486 vs .80206;
  FP 157 vs 51. This illustrates why ROC-AUC alone obscures the problem.
- **StepSafe:agentdog_safety:** 183 / 25. AP .58539 vs .81568;
  FP 40 vs 5. Another 364 actions have no original domain attribution; none
  are silently assigned the dialog-level domain.
- **AgentAlign:** 119 all-safe actions; FP 5 vs 0. AP/ROC-AUC are undefined.
- **Dialog length 2049–4096:** 370 / 73; AP .48310 vs .58779; FP 104 vs 35.
  **4097–8192:** 283 / 57; AP .32552 vs .44496; FP 92 vs 28.
  For 1025–2048 tokens FullFT AP is higher (.67600 vs .62262).
- **Second labeled action:** 206 / 44; AP .38693 vs .63736;
  FPR 42.59% vs 12.96%. First-action AP is also lower (.35740 vs .42753).
- **Dialogs with >4 labeled actions:** 387 / 64; AP .38571 vs .44035;
  FP 127 vs 48. Length, source and action count are confounded: this does not
  establish context length or state accumulation as the causal mechanism.

## Token curves: seven deterministic cases

One lexicographically first action key was selected per label × original-source
signature stratum, **without selection by prediction, correctness or score**.
The same seven cases are inspected across all four models. Full action logits,
decile means, endpoint/mean/max and exact keys are in `token_curves.json`.
No text, teacher rationale, or new model input is constructed.

In FullFT, all three selected unsafe actions have endpoint logits **above** their
action means by .8123, 1.6551, and 2.0267. All four selected safe actions have
endpoints below their means (−.1546 to −4.2758). Thus this small, deterministic
sample does not support uniformly depressed unsafe endpoints as the explanation
for FullFT's poor AP. It is descriptive, not a representative localization study.

## Train/dev history and limits of inference

FullFT epoch-1 train event/token BCE = **.18999 / .26031**, versus dev
**.59280 / .47016**. LoRA epoch-1 train event/token BCE = **.20593 / .25995**,
versus dev **.41070 / .36366**. FullFT fits training event labels better while
performing worse on this source-held-out dev set. This is consistent with poorer
generalization, but **not proof of overfitting or its cause**: training losses are
online averages, not matched end-of-epoch evaluation, and weighting/optimization
must be examined separately. FullFT has only one completed validation checkpoint;
there is no FullFT learning curve establishing when deterioration occurred.

LoRA epoch 2 lowers training event BCE to .08950 but worsens dev BCE to .46768
and AP to .49789. Frozen A epoch 2 raises AP while lowering its score scale
drastically and worsening event BCE relative to epoch 1. These histories show
that ranking, hard-event loss and the 0.5 operating point are distinct objectives.

Frozen A versus original frozen differs despite the nominal same-seed protocol
(.50919 vs .49634 AP). This audit confirms exact same cases and correct metrics;
it does not explain the inter-run reproducibility difference. No causal attribution
to an architectural or loss change is warranted.

## Artifacts and failed diagnostic attempts

- `run_20261018_v3/diagnostics.json`: full metrics, histories, exact keys, slices,
  paired per-action changes, and consistency checks.
- `run_20261018_v3/input_hashes.json`: SHA-256 of every consumed artifact,
  including all 1204 token files and the executed script.
- `run_20261018_v3/token_curves.json`: seven deterministic action curves × four models.
- `diagnose_fullft_validation*.py`: immutable server execution snapshots.
  Working script: `graph-safety/scripts/diagnose_fullft_validation.py`.
- v1 stopped on an overly strict 1e-7 CPU-sigmoid assertion (observed 1.1921e-7).
  v2 stopped because NumPy removed `trapz`; v3 uses `trapezoid`. Failed source
  snapshots and empty output directories remain. v3 reports numerical differences
  directly rather than concealing them; raw endpoint equality is exact.

These are isolated, post-hoc dev-select diagnostics, with no uncertainty estimates
or independent causal intervention. They rule out the inspected metric/identity/
endpoint inconsistencies, not every possible training or inference implementation bug.
