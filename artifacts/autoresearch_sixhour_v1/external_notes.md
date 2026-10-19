# Locked final-only external evaluator

`graph-safety/scripts/evaluate_autoresearch_external.py` owns no training or
threshold selection. One finalist/suite per invocation and GPU. No GPU scoring,
tests, lint, or formatting run during implementation.
CPU AST parsing succeeded; server imports and GPU execution remain unverified.

```bash
python graph-safety/scripts/evaluate_autoresearch_external.py \
  --plan /absolute/locked_finalist.json --root /absolute/finalist_external \
  --suite atbench
```

Repeat with `rjudge` and `dojo`; each creates a fresh `ROOT/SUITE`, refusing
overwrite. There is no automatic resume or retry. Use the existing project
`PYTHONPATH` including `graph-safety/src` and snapshot the imported scripts.

## Exact plan schema

All paths are absolute. SHA values are actual lowercase SHA-256 file hashes.
One plan describes one already selected finalist and its dev-select threshold.
The hashed `threshold_source` JSON must itself contain `"split": "dev_select"`
and `"threshold"` exactly equal to the plan threshold. It may contain additional
selection/fit provenance. Optional top-level `seed` defaults to original seed;
the existing trainer seed/environment setup is reused before model creation.

```json
{
  "final_only": true,
  "mode": "frozen",
  "head": {"path": "/absolute/selected_head.pt", "sha256": "HEAD_SHA"},
  "readout": "linear_softmax",
  "attention": false,
  "prompt": null,
  "template_sha256": "ORIGINAL_NATIVE_TEMPLATE_SHA",
  "frozen_weights_sha256": "ORIGINAL_QWEN_FROZEN_WEIGHTS_SHA",
  "threshold": 0.5,
  "threshold_source": {
    "split": "dev_select",
    "path": "/absolute/locked_dev_threshold_record.json",
    "sha256": "DEV_RECORD_SHA"
  },
  "deadline_unix": 1792348785.243998,
  "deadline_reserve_seconds": 120,
  "sequence_budget_seconds": 180,
  "suites": {
    "atbench": {
      "cases": {"path": "/var/local/danefedov/experiments/agent-safety/runs/whole-dialog-benchmarks-v1/atbench/cases.json", "sha256": "CASES_SHA"},
      "spans": {"path": "/var/local/danefedov/experiments/agent-safety/runs/whole-dialog-token-replay-v1/atbench/spans.json", "sha256": "SPANS_SHA"},
      "audit": {"path": "/var/local/danefedov/experiments/agent-safety/runs/whole-dialog-benchmarks-v1/atbench/audit.json", "sha256": "AUDIT_SHA"},
      "shared_rejections": {"path": "/var/local/danefedov/experiments/agent-safety/runs/whole-dialog-benchmarks-v1/atbench/shared_rejections.json", "sha256": "REJECTIONS_SHA"},
      "cache": "/var/local/danefedov/experiments/agent-safety/runs/whole-dialog-benchmarks-v1/atbench/features"
    }
  }
}
```

Add analogous suite entries for `rjudge` and `dojo`. Optional suite
`common_ids: {"path": "/absolute/ids.json", "sha256": "..."}` references a
nonempty unique JSON string list, independently chosen as the common fully
evaluated case IDs across finalists. It adds a matched report; it never skips
scoring remaining admitted cases. Main must intersect completed IDs across
finalists and only compare matching inputs; the evaluator does not discover
other runs automatically. Without `common_ids`, primary reports still list
every fully evaluated ID explicitly for downstream matching.

`mode: "lora"` additionally requires:

```json
"adapter": {
  "path": "/absolute/adapter",
  "files_sha256": {
    "adapter_config.json": "ACTUAL_SHA",
    "adapter_model.safetensors": "ACTUAL_SHA"
  }
}
```

The adapter manifest must include **every** regular file recursively, using
relative POSIX filenames (including README if present). Online encoding uses
the pinned original Qwen text model, validates its frozen weight hash before
PEFT loading, then calls `original.encode(model, sequence)` on unchanged native
IDs. Frozen mode loads original saved features read-only and checks the entire
original cache identity (weights, IDs, revision, template, Torch, bf16, SDPA,
padding), as well as exact padded tensor shape.

Readout choices are `endpoint`, `linear_softmax`, `null`, `multiscale`, `leaky`;
the evaluator imports the current trainer's `event_logit` verbatim and current
curve-head implementation. State dict loading is strict. Latest optional
before-Mamba attention is supported. No feature regeneration or input recoding.

## Explicit prompt limitation

**Main integration update:** explicit `mode:"online_frozen"` now supports ONLY
independently prepared native prompted inputs, never original cache reuse.
`prompt` must be `{"text":INSTRUCTION,"sha256":HASH,"training_audit":{"path":PATH,"sha256":HASH}}`.
Training audit, external preparation audit, vocabulary and fixed instruction hashes
must match; external audit must have `new_exclusions:[]` and `preparation_only:true`.
Prepare via `prepare_autoresearch_prompt_external.py`, then lock its four suite
files. Original base weights are verified before frozen online encoding; no
adapter is loaded. This integration is not yet runtime-validated.
The initial limitation below describes the delivered pre-integration version.

Any non-null `prompt` value produces `failure.json` with status
`NOT_IMPLEMENTED` **before GPU initialization/scoring**. There is no unprompted
fallback or manual text/ID prepending. Main must implement native
`state_messages + with_instruction + apply_chat_template`, regenerate spans,
validate prefix stability/fallback and cache provenance separately to evaluate
a prompted finalist. This evaluator intentionally does not claim that support.

## Outputs and failure semantics

Primary trajectory readout is max event score **including Complete**. Tool-only
and all-events-on-tool-cases sensitivities use exactly the same toolcase cohort.
Dojo requires one action per case and uses its action gold; trajectory gold is
never copied onto actions/tokens. Both locked dev-select threshold and .5 control
report AP, trapezoidal PR-AUC, ROC-AUC, P/R/F1/Accuracy/FPR/confusion and class
score quantiles. No external threshold fitting.

`predictions.jsonl` is flushed after each fully evaluated case. Sequence tensors
are persisted immediately under `tokens/` (including exact IDs, token/NULL
logits, spans, hashes, lengths and elapsed time); partial-case tensors survive
failure. Protocol records plan/source/adapter/head hashes and kernel environment.
Inherited context exclusions are copied verbatim from pinned original audit and
rejections. Any new overflow, OOM, mismatch, missing cache, or nonfinite value
aborts rather than excluding a case; traceback/completed/missing IDs survive.
Only fully completed suites get `summary.json` and `complete.json`.

Deadline checks precede model initialization/cases/sequences and reserve the
larger of configured first-sequence allowance or 1.5 times the longest observed
sequence runtime, plus the persistence reserve. They cannot preempt a running
CUDA kernel/model load; main should budget conservatively and schedule early.
No partial-run benchmark summary is silently emitted.
