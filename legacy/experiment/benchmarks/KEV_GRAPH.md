# First Kev-4B graph run: independent binary LINK questions

Requested first attempt, fixed before inference. All 571 raw pinned R-Judge
records in the existing deterministic order; no pilot, repeats, prompt search,
threshold calibration, gold-edge injection or candidate prefilter.

## Graph and causal adaptation

- Node: one recorded agent message (thought/action/text) with subsequent
  environment observations attached for **later** checks. A proposed action is
  inspected before its following observations are read. No invented success or
  block status, and decisions do not alter the replay.
- Each initial environment message before any agent action is a separate
  `initial_observation` node. It can supply inputs to later actions; no adapter
  failure or silent omission of the two affected records.
- Source agent profile and every user message available so far are always
  visible to both SAFETY arms. They are not filtered by predicted dependencies.
- At each proposed action, score **all** earlier nodes, one native `noul`
  question per candidate, including instruction/resource/data dependencies.
  Shared topic/time alone is not an edge; indirect-only sources are excluded.
  An error or failed attempt can still supply useful observations. Dependency
  describes what the action uses, not whether following that source is allowed.
- Add `Ei -> Ej` when native `noul >= .50`. No `NONE`, joint normalization
  over node IDs, forced top-k or maximum-parent cap. Edges point only forward
  in source time. Questions are separately normalized binary readouts, not a
  statistical-independence/calibration guarantee.
- SAFETY graph arm receives all predicted ancestors, in source order. The
  full-prefix control receives all earlier nodes. Both use the **same renderer
  and SAFETY question/criteria**, without graph annotation in the model input,
  to isolate history selection rather than extra hints.
- Exact action-level SAFETY wording follows the previous GLiClass R-Judge
  pipeline's broad-risk question (not the compact/H0 authorization question).
  Safety threshold .50 and maximum score over agent actions form the trajectory
  alarm for both arms. Max aggregation can amplify false alarms with length.
- Private label, risk_description, evaluator goal, scenario and attack_type do
  not enter the model. The source's recorded thoughts are observations, not
  annotator explanations or generated reasoning from our classifier.

R-Judge has **trajectory labels only**. Report trajectory confusion/F1/recall/
specificity/BA/AUROC and descriptive graph retention; no gold edge accuracy,
step recall, early-detection or prevented-harm claims. Labelled edge validation
would require a separate annotation task. Dataset research-use/license and
unknown training contamination caveats remain unchanged.

## Model, scoring and comparison

- Same frozen Kev-4B revision/base/head as the completed full-trajectory run:
  adapter `6cfce5c2fa4b4bd64026336ab649c5ca78857d52`, source
  `85ccf56a0afb952323c621055b02876e8498d60c`; base revision, adapter/head hashes
  and installed CUDA kernel environment are verified/retained.
- Same fp32 unmerged LoRA, no TF32/fused SDPA, serving fusions or CUDA graphs;
  shipped pointer temperature remains unchanged.
- Use native `model.probs()` to share the state prefill within a multiple-LINK
  request. This differs from the previous baseline's plain `forward()` row
  execution; the library documents agreement to fp32 rounding, not bit parity
  across kernel shapes. Both new SAFETY arms use the same `probs()` path.
- A read-only pointer-head hook preserves logits from the actual calls, without
  additional inference or changing outputs. Save native rounded and unrounded
  scores, reconstructed raw/scaled logits, mapping and full token encodings.
- Strict native admission; each state+question row <=8192, branch <=512.
  Multiple questions are admitted without incorrectly applying the row budget
  to the sum of all isolated branches. No truncation/retries.
- `graph_pipeline` versus `full_prefix_pipeline` is the main matched comparison.
  `full_trajectory_saved` reuses the existing 571 successful Kev-4B baseline
  scores and validates exact source-state hashes against those saved calls.
  It sees complete outcomes and uses a trajectory question, so it is a separate
  post-hoc reference, not a controlled measure of graph history selection.

## Remote invocation and artifacts

Own task `/home/danefedov/kev4b-noul-graph-20261013` on `llm01`, physical
H100 GPU 3. Existing isolated Kev CUDA env/cache and immutable source/data
reused without changing packages or previous artifacts.

```bash
CUDA_VISIBLE_DEVICES=3 \
HF_HOME=/dev/shm/danefedov-kev-rjudge-20261013/huggingface \
HF_HUB_DISABLE_IMPLICIT_TOKEN=1 \
TRITON_CACHE_DIR=/dev/shm/danefedov-kev-rjudge-20261013/triton \
TRITON_F32_DEFAULT=ieee \
/dev/shm/danefedov-kev-rjudge-20261013/venv/bin/python -u \
  /home/danefedov/kev4b-noul-graph-20261013/rjudge_kev_graph.py
```

Runner `rjudge_kev_graph.py`; shared helpers and pinned model spec from the
baseline runners. `data/kev4b_reference.json` preserves the previous 4B scores,
input hashes and source-result hash. Results go to a separate timestamped
directory and are copied locally: `calls.jsonl`, `records.jsonl`, `summary.json`,
`report.md`, code/lock snapshots and remote log. Counts include selected,
completed, failed and unattempted records; fatal errors stop rather than trigger
repeated calls. All ancestor selections, edges and per-action scores are saved.
No tests, formatter or linter are run.

## Completed first attempt

[Report](results/20261013T055106078924Z_rjudge_kev4b_noul_graph/report.md),
[summary](results/20261013T055106078924Z_rjudge_kev4b_noul_graph/summary.json),
[paired diagnostics](results/20261013T055106078924Z_rjudge_kev4b_noul_graph/paired_diagnostics.json).
571/571 successful, no truncation. Graph TP211/FP81/TN189/FN90 versus matched
full-prefix TP213/FP81/TN189/FN88: BA70.05% vs70.38%, AUROC.74509 vs.74493.
Four trajectory decisions change: one fix and three regressions, all unsafe.
No demonstrated graph benefit; no statistical-significance claim.

Ancestors retain77.2% of prior-event appearances; only198/1461 action checks
actually prune history. SAFETY token saving6.56%; including LINK, token total
1.82x matched full-prefix and observed call time2.10x (not isolated timing).
3814requests/4480questions; maxrow1823/state1719/branch109. Total860.50s.

Three regressions discard external instruction origins (LINK scores .1079,
.1739, .1744); one preserves the data producer but loses who initiated the
retrieval/email. A typed data/control-edge split is a future proposal, not
performed or tuned here. No edge gold exists. All outputs/logs/source snapshots
were copied locally; the separate 0.8B baseline also completed on GPU2.
