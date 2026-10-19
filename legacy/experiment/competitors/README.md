# Local classifier comparison

This isolated experiment replays only `../trajectory.json` and the frozen
`../evolution/dev.json`, using read-only `h0.json`, `evaluate.run_fixture`, and its
rendering/metrics code. It does not read held-out data, change thresholds, tune
prompts, fine-tune models, or execute any fixture tools. Each classifier builds
its **own** predicted LINK graph, then answers SAFETY both on its ancestors and
on the full observed prefix. Gold labels are attached by the existing evaluator
only after the event's predictions. Current results/status and future events do
not enter inputs; earlier outcomes are legitimate observations.

## Reproduce

```bash
cd experiment/competitors
uv sync --locked --python 3.12
uv run python compare.py base
uv run python compare.py large
uv run python compare.py gliclass
uv run python summarize.py
```

Run one model at a time. MPS is the default; `--device cpu` is an explicit fallback
run, not an automatic silent fallback. Weights are pinned to the revisions in
`compare.py`. Float32, SDPA, and disabled ModernBERT `reference_compile` avoid
CUDA-only FlashAttention/compilation requirements on Apple hardware. GLiClass
gets a `torch.device('mps')`, because the native pipeline silently maps a string
`'mps'` to CPU in this package version.

Package resolution is isolated in this directory's `pyproject.toml`, `uv.lock`,
and `.venv`. GLiClass 0.1.17/Transformers 4.57.3 use the Transformers-4 API; newer
GLiClass releases require Transformers 5. Laya is installed solely to import
the existing evaluator and runner, and is never loaded for inference here.

## Exact adapters and scoring

Both adapters expose `.predict(state, questions, max_len, head_max_len)`,
`.device`, `.revision`, and `.cpu_fallback_count`.

For every question, the premise/text is exactly:

```text
{h0-rendered state}

Question: {question.instructions}
```

Each hypothesis/label is exactly `{identity}: {criteria[identity]}` in original
criteria order. This preserves event IDs (and h0's observed-result descriptions)
and the fixed `ALLOWED`/`VIOLATION` definitions; no expected answer is supplied.
The added identity prefix is an adapter difference from native Laya.

* **ModernBERT base/large:** tokenize each `(premise, hypothesis)` pair without
  truncation; run one batched sequence-classification forward for all candidates.
  Both checkpoints map entailment to logit 0 and not-entailment to logit 1.
  Choice probability for candidate *i* is `softmax(entailment_logits)[i]` across
  candidates, **not** its independent `softmax(two_NLI_logits)[entailment]`.
  Both raw logits and independent probabilities are retained. This is the
  mutually exclusive zero-shot choice scoring convention, not calibrated
  probability that an edge exists.
* **GLiClass:** native `ZeroShotClassificationPipeline`, `multi-label`,
  `threshold=0.0`, with all candidate descriptions in **one forward pass**.
  The uni-encoder's exact input is its native
  `<<LABEL>>{label1}...<<LABEL>>{labelN}<<SEP>>{premise}` format
  (`prompt_first=true`, no examples or extra prompt). Native scores are
  independent `sigmoid(logit)` values. For Laya-compatible choice output,
  normalize these scores by their sum. Native scores and logits are preserved.
  This is deliberately not softmax over GLiClass logits. The model card notes
  NLI works best with a single hypothesis; our all-labels choice adapter differs.

The NLI hypotheses intentionally retain the original h0 descriptions rather
than introducing tuned causal assertions: a past result's entailment is not
itself proof that it supplies the proposed action's input. This semantic mismatch
is a limitation of the requested minimal choice adapter, especially for LINK.

Thresholds remain parent probability **>=0.2** and violation probability
**>=0.5**. Even when several parents pass, the normalized LINK choice is not an
independent multilabel edge classifier. No threshold calibration is performed.

## Context and artifacts

All three pinned checkpoint configs advertise **8192** encoder positions.
NLI budget includes premise, hypothesis, and special tokens for each pair;
GLiClass budget includes all label descriptions, separators, premise, and special
tokens in its single sequence. `max_len=8192`; `head_max_len` is accepted for
compatibility but there is no separate Laya question head in these encoders.
Inputs over budget raise an error, never truncate. For GLiClass, untruncated IDs
are compared exactly against the native pipeline's prepared IDs before inference
(the library itself defaults to truncation and a 1024-token budget). Every usage
record has actual sequence lengths and `truncated=false` on successful calls.
This verifies these replay inputs, not performance at the full 8k boundary.

Each timestamped `results/*` directory contains:

* `trace.json`, complete fixture inputs, outputs, decisions and evaluation labels;
* `calls.jsonl`, append-only per-call inputs, raw native outputs/logits, token IDs,
  timings and exceptions (also preserves calls from a failed replay);
* `summary.json`, safety confusion matrices, balanced accuracy, edge TP/FP/FN,
  precision/recall/F1, paired correctness, versions, revision and times;
* pinned model card/config/tokenizer config and source/data snapshots, with
  original SHA-256 hashes in the trace. Source/data snapshots stay under this
  directory; no parent files are modified.

Token cost is actual total encoder tokens, so NLI includes repeated premises for
each candidate and is not directly comparable to Laya's shared-state token cost.
Timings include synchronization but no warmup/repeats; downloads/load are recorded
separately. Parent Laya runs may share the MPS device, so these are not isolated
latency benchmarks.

## Primary sources and limitations

Model cards inspected before adapter implementation:

* https://huggingface.co/MoritzLaurer/ModernBERT-base-zeroshot-v2.0
* https://huggingface.co/MoritzLaurer/ModernBERT-large-zeroshot-v2.0
* https://huggingface.co/knowledgator/gliclass-modern-base-v3.0

The ModernBERT cards describe general zero-shot/NLI training and an 8k context,
but warn their training mix does not fully exploit long context. The large card's
heading incorrectly says base; its description and config identify large.
GLiClass v3's card describes logical-task LoRA training and native one-pass
classification, not an agent safety checkpoint. These are domain-shifted zero-shot
adapters, not interface-equivalent substitutes. The small original/development
fixtures are diagnostic shadow replay, not held-out generalization or closed-loop
harm prevention. No tools are actually called.

### Jev: official documentation inspected, not run

The vendor homepage https://typesafe.ai links to https://docs.typesafe.ai.
Official API reference: https://docs.typesafe.ai/api.md; overview:
https://docs.typesafe.ai/introduction. The official endpoint is
`POST https://api.typesafe.ai/v1/systemone` with Bearer authentication and
`{model, state, questions}`. Choice returns typed choice/probabilities/confidence;
Noul and Score are also supported. Choice supports up to 255 options and returns
probabilities summing to one. The vendor describes calibrated decisions without
text generation, not a local downloadable checkpoint.

**Jev was not run.** No credentials were searched, no key was accessed, no paid
API was called, and no fixture was sent externally. Search results also included
independent reseller sites; those were not treated as the official API contract.
