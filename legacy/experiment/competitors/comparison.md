# Frozen h0 classifier comparison

One zero-shot replay per model; no fine-tuning or threshold tuning. Each model builds its own graph. Original: 10 safety decisions; development: 8 target decisions in 4 matched pairs. No held-out access.

| Model / device | Original graph BA | Original full BA | Original edge F1 | Dev graph BA | Dev full BA | Dev edge F1 | Dev pairs correct | Inference s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ModernBERT-base-zeroshot-v2.0 / mps | 0.500 | 0.595 | 0.273 | 0.750 | 0.750 | 1.000 | 2/4 | 5.17 |
| ModernBERT-large-zeroshot-v2.0 / mps | 0.500 | 0.929 | 0.273 | 0.625 | 0.750 | 1.000 | 1/4 | 15.08 |
| gliclass-modern-base-v3.0 / mps | 0.929 | 0.762 | 0.300 | 0.750 | 0.875 | 1.000 | 2/4 | 3.69 |

BA = balanced accuracy; edge F1 compares all predicted direct edges to gold.

## Confusion matrices and context audit

Safety counts below are **TP / FP / TN / FN**, with VIOLATION positive. Edge counts are **TP / FP / FN**.

### MoritzLaurer/ModernBERT-base-zeroshot-v2.0

Artifacts: [`results/20261012T224417Z_base_mps`](results/20261012T224417Z_base_mps); revision `d421c4545a438fd006fb43f8b981c5d908faa1e1`.
MPS calls: 53; CPU fallbacks: 0. Longest actual encoded sequence: **605/8192 tokens**. All successful usage records say `truncated=false`; GLiClass additionally checks exact native token-ID equality before each forward pass.
* **original**: graph 0 / 0 / 7 / 3; full 1 / 1 / 6 / 2; edges 3 / 11 / 5 (precision 0.214, recall 0.375).
* **dev**: graph 2 / 0 / 4 / 2; full 2 / 0 / 4 / 2; edges 8 / 0 / 0 (precision 1.000, recall 1.000).
Load/download 26.97s; inference 5.17s; total 32.14s.

### MoritzLaurer/ModernBERT-large-zeroshot-v2.0

Artifacts: [`results/20261012T224508Z_large_mps`](results/20261012T224508Z_large_mps); revision `a51e07b524299e309dd2b88d48b0cfa2bd9ec598`.
MPS calls: 53; CPU fallbacks: 0. Longest actual encoded sequence: **605/8192 tokens**. All successful usage records say `truncated=false`; GLiClass additionally checks exact native token-ID equality before each forward pass.
* **original**: graph 0 / 0 / 7 / 3; full 3 / 1 / 6 / 0; edges 3 / 11 / 5 (precision 0.214, recall 0.375).
* **dev**: graph 1 / 0 / 4 / 3; full 2 / 0 / 4 / 2; edges 8 / 0 / 0 (precision 1.000, recall 1.000).
Load/download 48.67s; inference 15.08s; total 63.75s.

### knowledgator/gliclass-modern-base-v3.0

Artifacts: [`results/20261012T224621Z_gliclass_mps`](results/20261012T224621Z_gliclass_mps); revision `ac369222ca4375ca66ebaf7fb5220f223514c035`.
MPS calls: 53; CPU fallbacks: 0. Longest actual encoded sequence: **791/8192 tokens**. All successful usage records say `truncated=false`; GLiClass additionally checks exact native token-ID equality before each forward pass.
* **original**: graph 3 / 1 / 6 / 0; full 2 / 1 / 6 / 1; edges 3 / 9 / 5 (precision 0.250, recall 0.375).
* **dev**: graph 3 / 1 / 3 / 1; full 3 / 0 / 4 / 1; edges 8 / 0 / 0 (precision 1.000, recall 1.000).
Load/download 32.61s; inference 3.69s; total 36.31s.

## Interpretation

These small domain-shifted development results are diagnostic, not a general ranking. Graph safety uses each competitor's own errors in dependency selection; the full-prefix baseline uses the identical safety question without graph filtering. Fixed offline shadow replay does not demonstrate closed-loop harm prevention.

Adapters are not native Laya-equivalent: NLI uses softmax across entailment logits; GLiClass uses native independent sigmoid scores normalized by their sum. LINK >=0.2 and VIOLATION >=0.5 are unchanged, uncalibrated thresholds for these competitors. Multi-parent edges are thresholded from a choice distribution. See [README.md](README.md) for exact inputs and scoring formulas.

All timings are single runs without warmup; parent Laya evaluations may concurrently share the MPS device. Float32 SDPA is used, compilation disabled. Tokenizer metadata may warn about a stale 512-token limit; no truncation is requested, and the pinned encoder configs specify 8192. Counts above audit actual sequences, not an 8k stress test.

**Jev: not run.** Official vendor-linked docs were inspected at https://docs.typesafe.ai/api.md (`POST https://api.typesafe.ai/v1/systemone`, typed state/questions and choice probabilities). No credentials were searched or paid external inference called. See README for provenance and interface differences.
