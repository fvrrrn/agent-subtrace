# Kev-0.8B: full-trajectory R-Judge baseline

User requested this **after** the first Kev-4B graph run. "Ordinary scenario"
is interpreted explicitly as the same full-trajectory R-Judge baseline without
a graph; this is not the old ten-event XML replay. The user then explicitly
requested immediate concurrent execution on a different GPU: 0.8B uses GPU 2,
while the 4B graph remains on GPU 3. No concurrent jobs share a GPU with each other.

Same 571 raw pinned records, source states, final safety question/option order,
threshold .50, no prompt/threshold fitting. Exact prior GLiClass input hashes
are verified on the 569 comparable cases; both all-record and common-record
metrics are saved. One native forward per complete trajectory, no generation,
graph, step aggregation, training or calibration.

## Frozen model

`jaredpalmer/kev-0.8b` @ `bf75a6a8848ea6960ff2ed108d9ed44c2941174f`, the
previously downloaded released checkpoint. Head/adapter hashes were inspected
from the local public cache, without loading or evaluating the backbone:

- Adapter SHA256 `9b908623acb162118575f4e7a94524f9c139c335be4bfb74d6cfceca01e1885a`.
- Head SHA256 `f400bd12802b2b105ae45d6b03774a158a3db4fccff42413734ddca2e5c920b6`.
- Base `Qwen/Qwen3.5-0.8B-Base` @
  `dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68`.
- Shipped pointer temperature **2.3510958125672174**, unchanged.

Same official pinned Kev source and isolated CUDA environment as 4B. Like 4B's
ordinary baseline, use fp32 unmerged LoRA, native `forward`, strict admission,
no TF32/fused SDPA, serving fusions or CUDA graphs. Native rounded probabilities
drive metrics; unrounded scores/logits and actual kernel environment retained.
Source/weight/head hashes are checked; no hidden truncation at 8192/512 limits.
Model training contamination against public R-Judge is unknown. Dataset research
license and trajectory-only interpretation remain unchanged.

## Remote invocation

`llm01`, physical H100 GPU 2, own task directory
`/home/danefedov/kev08-rjudge-baseline-20261013`. Existing immutable source/data
and isolated environment are reused; no package changes or other GPU workloads.

```bash
CUDA_VISIBLE_DEVICES=2 \
HF_HOME=/dev/shm/danefedov-kev-rjudge-20261013/huggingface \
HF_HUB_DISABLE_IMPLICIT_TOKEN=1 \
TRITON_CACHE_DIR=/dev/shm/danefedov-kev-rjudge-20261013/triton \
TRITON_F32_DEFAULT=ieee \
/dev/shm/danefedov-kev-rjudge-20261013/venv/bin/python -u \
  /home/danefedov/kev08-rjudge-baseline-20261013/rjudge_kev_baseline.py --size 0.8b
```

Separate timestamped results preserve all calls/token encodings, scores, logits,
usage/errors, metrics, versions, hashes and code snapshots. Outputs/logs will
be copied locally. No tests, formatter or linter are requested or run.

## Completed single run

[Report](results/20261013T055733531881Z_rjudge_kev0.8b_baseline/report.md),
[summary](results/20261013T055733531881Z_rjudge_kev0.8b_baseline/summary.json).
All571 successful without truncation/errors; max1,786 combined tokens.
TP49/FP83/TN187/FN252; recall16.3%, specificity69.3%, F122.6%, BA42.8%,
AUROC.496. Common569: BA42.8%/AUROC.497 vs Kev4B71.2%/.766 and
Kev27B90.4%/.942. The 0.8B frozen readout is not useful on this workload.

H100GPU2 concurrent with 4B graph on GPU3; download/load24.96s, inference147.16s,
total172.12s. Not isolated timing. All artifacts/log/pinned source copied locally;
no tuning, graph construction for0.8B or repeat inference followed the baseline.
