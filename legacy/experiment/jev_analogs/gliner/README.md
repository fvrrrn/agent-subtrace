# GLiNER2.5-Decide original-protocol replay

Model: official `fastino/GLiNER2.5-Decide` (340M), revision recorded in
`model_revision.txt`. Local package `gliner2[local]`, isolated `.venv`; exact
dependencies in `requirements.lock.txt`. No hosted API, credentials, training,
calibration, or holdout access.

`laya` is installed only because the shared runner's imported `experiment/run.py`
imports it; no Laya model or predictions are used in this replay.

## Mapping fixed before replay

`run_gliner.py` imports the shared, unchanged `run_original` evaluator. Original
XML state is native classification text, unchanged. Each original choice
question maps to `ClassificationSchema.single` under its original question name:

- Original instructions become native `instruction`, compiled to `prompt`.
- Original ID-only option names and order are native labels. Original criterion
  descriptions are native `label_descriptions`. Nothing is added to the content.
- The native processor serializes descriptions into the schema head with
  `[DESCRIPTION]` markers. These are standard model/package tokens, not custom
  task-specific prompts.
- The native processor appends `.` to text not already ending in sentence
  punctuation; original XML is preserved in full, followed by that standard
  package normalization. Both original and actual native text are saved.
- Exclusive `auto` activation uses **native softmax of all raw label logits**,
  temperature 1. This is implemented by the package's
  `ClassificationScores.probability` and matches the default `classify_text`
  exclusive decoder. No external normalization adapter is needed. Native logits
  and their sigmoid values are retained for audit; sigmoid is not the probability
  used by the runner.
- Shared LINK 0.2 and VIOLATION 0.5 thresholds are unchanged.

Primary evidence: `docs/model_card.md`, `docs/classification.md`,
`docs/package_README.md`, and `docs/long_context.md`; URLs are in `docs/urls.json`.
Installed package source snapshot is in `source/gliner2/`. Relevant code is
`classification/scoring.py`, `classification/compiler.py`,
`inference/runtime.py` (`_compute_classification_probabilities`),
`processor.py` (`_process_classifications`, `_transform_schema`,
`_format_input_with_mapping`), and `models/base.py` (`encode_tokens`).

## Input coverage and hardware

Float32, Apple MPS, standard checkpoint attention; MPS fallback explicitly
disabled. The checkpoint declares `max_position_embeddings=512` but sets
`position_biased_input=false`, with bucketed relative attention. This is not
itself proof of a 512-token hard limit: the native encoder receives the full
sequence. Native `max_len=None` disables word-token truncation. Every request
records the actual processor token IDs, schema and state subword counts, full
word coverage, and the encoder-forward shape. A mismatch or budget overflow
raises an error; no dropped state or chunk aggregation is permitted. Shared
4096 state / 512 head budgets are checked against actual native subwords.

`results/raw/*.json` preserves native requests and responses, including raw
logits, native softmax probabilities and token checks. `results/trace.json` and
`results/summary.json` use the common evaluator format. `run.log` preserves
loading/inference messages and concrete failures if any.

## Actual result

Completed 29 calls / 10 events in 15.75 seconds of replay on MPS float32.

| Mode | TP | FP | TN | FN |
| --- | ---: | ---: | ---: | ---: |
| Graph safety | 1 | 3 | 4 | 2 |
| Full-history safety | 0 | 2 | 5 | 3 |
| Edges (45 possible earlier-event pairs) | 4 | 6 | 31 | 4 |

10 predicted edges, 8 gold edges, 3/10 exact parent sets. Every call used one
full native encoder pass. Maximum actual sequence was **1073 subwords**; maximum
state was **862**, maximum head **210**. Final full-state safety input was
**862 state + 88 head + 1 separator = 951**. Thus the checkpoint's nominal 512
position setting did not force truncation or native chunk aggregation in this
package/encoder configuration. Audit: `results/input_audit.json`.

Loading emitted generic model-type compatibility warnings (`gliner2` versus
`extractor` / empty model type), preserved in `run.log`; the official
`AutoExtractor` resolved the span checkpoint and inference completed. A preliminary
coverage assertion initially compared the unnormalized word count to the
package's period-appended count; `preflight_failure.log` preserves that failure.
The corrected check includes native punctuation normalization and verifies no
original state words are omitted.

## Reproduce

```sh
uv venv experiment/jev_analogs/gliner/.venv --python 3.12
uv pip install --python experiment/jev_analogs/gliner/.venv/bin/python -r experiment/jev_analogs/gliner/requirements.lock.txt
experiment/jev_analogs/gliner/.venv/bin/python experiment/jev_analogs/gliner/run_gliner.py
```

The pinned Hugging Face snapshot is saved in `model/`; inference forces offline
Hub mode. Re-running overwrites only this directory's replay artifacts.
