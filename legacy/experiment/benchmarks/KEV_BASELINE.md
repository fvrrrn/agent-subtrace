# Kev-4B: R-Judge full-trajectory baseline on llm01

Explicitly requested after the completed Laya baseline. This is **not** a
restart of the cancelled Mac graph replay: it is a new, graph-free evaluation
on R-Judge. No other Kev sizes/checkpoints, prompt variants or training.

## Protocol fixed before inference

- Same 571 raw pinned R-Judge records, deterministic order, state strings and
  binary safety question as the Laya/GLiClass full-trajectory baselines.
  Shared conversion/metrics come from `rjudge_laya_baseline.py`; importing its
  helpers does not load Laya or run inference.
- Exact prior GLiClass state/question hashes checked for the 569 common cases.
  Labels, annotator explanations, evaluator goal and category metadata stay
  outside classifier inputs. Initial environment messages remain admissible.
- Frozen `jaredpalmer/kev-4b`, adapter revision
  `6cfce5c2fa4b4bd64026336ab649c5ca78857d52` (the previously selected v1.0).
- Base `Qwen/Qwen3.5-4B-Base`, revision
  `1001bb4d826a52d1f399e183466143f4da7b741b`.
- Official Kev code pinned at `85ccf56a0afb952323c621055b02876e8498d60c`.
  Native `Checkpoint.load`, `SystemOneRequest`, `to_record`, strict `admit`,
  `DecisionModel.forward`, temperature-scaled softmax and `to_answers`.
  No autoregressive generation and no server or paid endpoint.
- Official evaluation-style CUDA precision: fp32, **unmerged** frozen LoRA,
  TF32 and fused SDPA off, no serving fusions or CUDA graphs. CUDA DeltaNet
  kernels use flash-linear-attention .5.2, Triton 3.7.1 and causal-conv1d 1.7.0;
  `TRITON_F32_DEFAULT=ieee`. Actual kernel bindings are recorded through native
  `kernel_environment`; this is different from Laya's bf16 batch runtime.
- Shipped pointer-head temperature remains unchanged (previous metadata:
  2.406050072164233); no calibration on R-Judge. Artifact hashes for the adapter
  and head are verified before loading the backbone.
- One safety classification per full trajectory, threshold .50. Native
  four-decimal probabilities determine metrics; unrounded probabilities and
  scaled/reconstructed raw logits are also saved without additional inference.
- Strict admission, no truncation. Additional requested combined budget 8192
  and question-branch budget 512. Broken model/CUDA errors stop the run rather
  than retrying or producing hundreds of repeated failures.
- All-record and common-successful-record metrics, ROC-AUC and class-conditional
  score quantiles. No threshold search or label-based input selection.

R-Judge evaluates trajectories, not edges or steps. These are classification
metrics, not prevention results. Public-model training contamination is unknown.
The dataset's CC BY-NC-SA research-use note remains applicable.

## Remote environment

SSH `llm01` / `avi-ix-llm-devbox01`, physical H100 PCIe GPU 3.
Task: `/home/danefedov/kev-rjudge-baseline-20261013`; copied immutable Kev source
under `source/`, standalone runners at the task root. `data/` references the
already transferred, checksum-verified R-Judge inputs from the Laya task.
Packages/model/Triton caches use own executable tmpfs directory
`/dev/shm/danefedov-kev-rjudge-20261013`, without touching existing project envs.

Source `uv.lock` is installed with `uv sync --frozen --no-dev --python 3.12`.
CUDA additions are the official image's pinned flash-linear-attention .5.2,
Triton 3.7.1 override and the causal-conv1d cu12/torch2.8/cp312 Linux wheel.
Triton/causal-conv1d are installed `--no-deps` to avoid restoring torch's pinned
Triton 3.4; the resulting installed package inventory is retained separately
from the source lock. No tests or linter were run.

```bash
CUDA_VISIBLE_DEVICES=3 \
HF_HOME=/dev/shm/danefedov-kev-rjudge-20261013/huggingface \
HF_HUB_DISABLE_IMPLICIT_TOKEN=1 \
TRITON_CACHE_DIR=/dev/shm/danefedov-kev-rjudge-20261013/triton \
TRITON_F32_DEFAULT=ieee \
/dev/shm/danefedov-kev-rjudge-20261013/venv/bin/python -u \
  /home/danefedov/kev-rjudge-baseline-20261013/rjudge_kev_baseline.py
```

Outputs: `summary.json`, `report.md`, `records.jsonl`, `calls.jsonl`, runner and
helper snapshots, source dependency files. Calls retain the exact native
request, internal mapping, token encoding, output/usage, logits and errors.
Source revision, checkpoint/base revisions, hashes, shipped temperature, actual
kernel environment and times are saved. Remote logs/results are copied back
to the local workspace after completion.

## Completed evaluation

[Report](results/20261013T052137026041Z_rjudge_kev4b_baseline/report.md),
[summary](results/20261013T052137026041Z_rjudge_kev4b_baseline/summary.json).
571/571 completed without errors/truncation; max 1,786 combined tokens.
51.62 s model download/load, 236.65 s inference, 288.28 s total excluding setup.

All-record TP=214/FP=78/TN=192/FN=87; recall 71.1%, specificity 71.1%,
F1 72.2%, BA 71.1%, AUROC .765. Common 569-record BA=.712/AUROC=.766,
versus Laya .525/.541 and GLiClass .500/.630. Kev is a stronger baseline here,
but still incorrectly flags 28.9% of safe trajectories. Native scoring
distributions/logits and remote runtime/setup logs were copied locally.
No graph construction, additional model or tuning was performed.
