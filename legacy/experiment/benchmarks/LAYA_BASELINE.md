# Laya: R-Judge full-trajectory baseline

Requested baseline only: no LINK predictions, graph construction, prompt search,
threshold fitting, fine-tuning, per-step alarms or max-over-actions aggregation.
One choice classification per published record, using the complete observed
interaction. This matches R-Judge's **trajectory-level** labels.

## Fixed protocol

- Model `convaiinnovations/laya`, pinned revision
  `7b928d828b7b0e022f929d9bd2e44165aa270148`; `laya==0.4.6`.
- Runner: `rjudge_laya_baseline.py`; environment: `../pyproject.toml`,
  `../uv.lock`. The runner can be copied alone with these environment files and
  the cached data; it does not import the GLiClass adapter or download new data.
- Data: same pinned R-Judge repository/checksums as the GLiClass run, revision
  `83ce301da3ad50dd8b397e772863f5411c3d3dc2`; all 571 raw records in the same
  deterministic hash order, without label filtering.
- Exact same full-trajectory state strings, safety question and two criteria as
  GLiClass. `data/gliclass_reference.json` preserves the previous record keys,
  scores and state/question hashes. These hashes are checked **before inference**
  for all 569 records where GLiClass produced a final classification.
- Laya receives state and question separately through its native `predict_batch`
  API. GLiClass appended its question to the classified premise and encoded
  criteria as labels. Therefore matched source text/question does **not** imply
  identical token-level inputs, score semantics or model architecture.
- Native choice score `VIOLATION >= .50`. Shipped temperatures and library
  clamping remain unchanged and are recorded, not fitted on benchmark labels.
- Eager CUDA backend, batch size 16, seed 0; `max_len=8192`,
  `head_max_len=512`. Exact usage and all outputs saved. Truncated/collapsed
  inputs are explicit failures, never included as successful classifications.
- Only source profile and contents enter the model. `goal`, `label`,
  `risk_description`, scenario and attack_type do not. Annotation labels are
  used only for reference checks and evaluation, never included in model input.
- Full-trajectory inputs can contain initial environment observations; no
  agent-event conversion is required. Thus the two previous GLiClass adapter
  failures are attempted rather than arbitrarily excluded.

Both all-record metrics and metrics on the common successful record set are
reported. AUROC and class-conditional score distributions are included without
any post-hoc threshold selection. No graph accuracy or harm-prevention claim
is possible in this baseline. Public-benchmark training contamination is unknown.
Dataset use remains non-commercial research under the source's CC BY-NC-SA note.

## Remote run on llm01

Host: `avi-ix-llm-devbox01` (SSH alias `llm01`); physical GPU 3, H100 PCIe.
Own isolated task directory: `/home/danefedov/laya-rjudge-baseline-20261013`.
Existing projects/environments and other GPU workloads are left untouched.
The disk is nearly full, so packages and model cache use the executable tmpfs
`/dev/shm/danefedov-laya-rjudge-20261013`. Scripts/data/results stay in the home
task directory; outputs will also be copied back to this workspace.

Remote environment:

```bash
UV_CACHE_DIR=/dev/shm/danefedov-laya-rjudge-20261013/uv \
UV_PROJECT_ENVIRONMENT=/dev/shm/danefedov-laya-rjudge-20261013/venv \
uv sync --locked --python /opt/python3.12/bin/python3.12 \
  --project /home/danefedov/laya-rjudge-baseline-20261013
```

Remote invocation (one complete evaluation):

```bash
CUDA_VISIBLE_DEVICES=3 \
HF_HOME=/dev/shm/danefedov-laya-rjudge-20261013/huggingface \
/dev/shm/danefedov-laya-rjudge-20261013/venv/bin/python -u \
  /home/danefedov/laya-rjudge-baseline-20261013/rjudge_laya_baseline.py \
  --device cuda:0 --batch-size 16
```

`cuda:0` is logical GPU 0 after restricting visibility to physical GPU 3.
The isolated locked environment includes Linux/CUDA dependencies, unlike the
Mac environment; exact installed versions and effective dtype/AMP are retained.

## Artifacts

Each run saves `summary.json`, `report.md`, `records.jsonl`, `calls.jsonl`,
the runner snapshot, `pyproject.toml`, `uv.lock`. Summary includes source/data
hashes, selected/completed/failed counts, environment versions, checkpoint
configuration, effective temperatures, device/host, load/inference timing.
Calls include exact state/question, full native output and usage/errors.

## Completed baseline

[Report](results/20261013T051421062580Z_rjudge_laya_baseline/report.md),
[summary](results/20261013T051421062580Z_rjudge_laya_baseline/summary.json).
571/571 completed, no errors/truncation, max 1,870 encoded tokens. H100 physical
GPU 3, bf16 AMP; 14.45 s download/load and 8.94 s inference.

All-record results: TP=55, FP=36, TN=234, FN=246; recall 18.3%, specificity
86.7%, F1 28.1%, BA 52.5%, AUROC .542. Common 569-record AUROC .541 versus
GLiClass .630; BA .525 versus .500. Laya avoids the all-positive collapse but
does not provide strong safety discrimination. No prompt/threshold changes or
graph experiments followed this baseline. Remote execution log is retained.
