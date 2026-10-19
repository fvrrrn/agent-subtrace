# Kev-27B: the same R-Judge full-trajectory baseline

Requested after Kev-4B completed. One frozen run of all 571 raw pinned R-Judge
records, with the **same state strings, question, option order and .50 safety
threshold** as the completed baselines. No graph, generation, prompt variants,
threshold selection or fine-tuning. Common-record results use the same 569
successful GLiClass cases; private annotations do not enter inference inputs.

## Frozen checkpoint and runtime

- Public released Kev-27B **v2**, full-weight bf16 checkpoint:
  `jaredpalmer/kev-27b` @ `af0e6d551bdc2cc724f3e9d7a8bee1cd4fb8f7bf`.
  Pinned from the public model API before inference, not a moving `main`.
- Base provenance `Qwen/Qwen3.8-27B` @
  `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` (post-trained, not a `-Base` model).
- Released aggregate weights SHA-256:
  `d27af6ab2be16824166ac639907b4dba40979ff338599c2872721aa6c5072022`;
  pointer-head SHA-256:
  `7968f17b03479c1ef9d1c0f3ab8a15e31ecb441cf40691b07ee945ab554d45ad`.
  Native `Checkpoint.weights_sha256()` checks all 11 shard hashes before load.
- Shipped temperature **1.319507910772894**, unchanged. Metadata/head hash
  inspected before starting the full run. No R-Judge calibration.
- Same official Kev source `85ccf56a0afb952323c621055b02876e8498d60c`, CUDA
  environment/kernel package set and native request/admission/readout as Kev-4B.
  `rjudge_kev_baseline.py --size 27b` reuses the same conversion and metrics.
- Unlike Kev-4B's unmerged fp32 LoRA evaluation, 27B loads its **full trained
  weights as saved in bf16**, no adapter and no upcast. This fits one H100 80 GB.
  TF32 and fused SDPA remain off; no serving-model fusions or CUDA graphs;
  `TRITON_F32_DEFAULT=ieee`; actual model/kernel environment is recorded.
- One native forward per trajectory, strict admission, no truncation;
  requested combined/branch limits 8192/512. Native four-decimal probability
  drives metrics; unrounded scores and scaled/reconstructed logits retained.

This compares **released checkpoints**, not model size alone: their bases,
training recipes, shipped calibration and runtime precision differ. Training
contamination against public R-Judge remains unknown. There are no action/edge
labels here, so neither early detection nor graph accuracy is measured.

## Remote invocation

SSH `llm01` (`avi-ix-llm-devbox01`), physical H100 PCIe **GPU 3**, checked free
before launch. Own task directory:
`/home/danefedov/kev27-rjudge-baseline-20261013`. Immutable data and pinned
Kev source reference the previous task; the existing isolated Kev environment
is reused without changing packages. Large weights use the existing own
executable tmpfs cache, not the nearly-full system disk.

```bash
CUDA_VISIBLE_DEVICES=3 \
HF_HOME=/dev/shm/danefedov-kev-rjudge-20261013/huggingface \
HF_HUB_DISABLE_IMPLICIT_TOKEN=1 \
TRITON_CACHE_DIR=/dev/shm/danefedov-kev-rjudge-20261013/triton \
TRITON_F32_DEFAULT=ieee \
/dev/shm/danefedov-kev-rjudge-20261013/venv/bin/python -u \
  /home/danefedov/kev27-rjudge-baseline-20261013/rjudge_kev_baseline.py --size 27b
```

Previous Kev-4B outputs/source snapshots stay unchanged. All current inputs,
outputs, logits, errors, hashes, checkpoint settings, installed packages and
times go to a separate `results/*_rjudge_kev27b_baseline/` directory, copied
back locally after completion. No other model/size is launched.

## Completed single run

[Report](results/20261013T053204261439Z_rjudge_kev27b_baseline/report.md),
[summary](results/20261013T053204261439Z_rjudge_kev27b_baseline/summary.json).
571/571 successful, no truncation/errors, max 1,786 combined tokens. Native bf16
CUDA; download/hash/load 162.50 s, inference 112.47 s, total 274.98 s.

All-record TP248/FP4/TN266/FN53; precision98.4%, recall82.4%, specificity98.5%,
F189.7%, BA90.5%, AUROC.942. Common569: BA90.4%, AUROC.942, compared with
Kev-4B71.2%/.766, Laya52.5%/.541, GLiClass50.0%/.630. Safe/unsafe risk medians
.1193/.9589 on all571. A stronger baseline, but 53 unsafe records are still
missed; graph quality/prevention and training contamination remain unmeasured.
No fitting or repeat inference was performed. Outputs/logs/pinned source
snapshots were copied to this workspace; previous results remain unchanged.
