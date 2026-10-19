# Prompted protected external inputs — conditional preparation only

New script: `graph-safety/scripts/prepare_autoresearch_prompt_external.py`.
No existing evaluator was changed. No dataset preparation or GPU work was run.
Use only if the fixed-prompt model becomes a locked finalist and main authorizes
preparation; this is not another candidate-selection dataset.

## CPU preparation

```bash
python graph-safety/scripts/prepare_autoresearch_prompt_external.py \
  --parent /var/local/danefedov/experiments/agent-safety/runs/whole-dialog-benchmarks-v1 \
  --suite atbench --root NEW_PROMPT_EXTERNAL_ROOT
```

Repeat explicitly for `rjudge` or `dojo` when authorized. Output lives under
`ROOT/SUITE/`: `cases.json`, `spans.json`, `audit.json`,
`shared_rejections.json`. There is no inference/cache phase in this script.

The tokenizer is local-only, original pinned Qwen4B revision and native template.
Each parent action's exact JSON state hash, original native text, original
sequence IDs hash, and causal-prefix IDs hash are checked before modification.
`state_messages` preserves source roles, tool declarations, and opaque supplied
history behavior. The same fixed `with_instruction` function as training adds
the instruction. Only explicit input fields enter templates; labels and other
annotations remain metadata. No manual text/ID prepend is used.

The final state's native render supplies a candidate full dialog, exactly as in
the parent preparer. Every independently rendered action-state prefix must
match its characters and IDs through the action endpoint. If any fails, ALL
actions in that case receive separate native sequences rendered from their own
original causal states, not truncated slices of the final state. Future tool
results are therefore absent from the applicable causal prefix. Native closing
tokens remain after the endpoint in fallback sequences, as in the parent.

Each action body is independently identified from offsets in the selected
sequence and checked against its causal-render span. The native empty-think
prefix is skipped; role closing tokens are excluded. The schema matches
`evaluate_autoresearch_external.py`: spans are keyed by case, indexed by original
action index, with `key`, `state_sha256`, `start`, `end`, `action_tokens`, and
`prefix_ids_sha256`. Parent sequence/prefix hashes and endpoints are additional
explicit metadata. Original case/action keys, state strings, kinds, labels and
ordering are retained.

All sequences must fit 65,536 tokens without truncation. Any NEW overflow,
missing span, hash mismatch, or causal error aborts the entire suite before an
output directory is created, with the offending case key. No smaller evaluation
set is silently emitted. Inherited exclusions are copied byte-for-byte: the
parent ATBench 19 exclusions remain inherited, not newly evaluated cases.
Audit records matching IDs, input/output hashes, instruction hash, actual
lengths, parent/new whole-dialog/fallback counts, and mode-change IDs.

## Main evaluator integration (not implemented in existing files here)

1. After finalist locking and authorization, run CPU preparation. Hash-lock all
   four output files in the final plan's suite entries; use the emitted audit
   hash as the plan's `audit.sha256`. Require no new exclusions and exact parent
   admitted/action sets. Compare `prompt` and `prompt_sha256` with the locked
   training preparation audit, and require the pinned base/revision/template.
2. Replace the evaluator's blanket non-null-prompt rejection only for this
   verified prepared-input path. Never use unprompted inputs as fallback. Keep
   all existing sequence/hash/span checks; these outputs support them directly.
3. Add an explicit online-encoding mode for frozen prompted finalists. Load
   `original.backbone(False)`, verify `frozen_hash` against the locked plan,
   freeze/eval, and call `original.encode(model, seq)` exactly like the current
   online LoRA branch, but WITHOUT loading an adapter. Do not access old feature
   caches or require a cache path for this mode. If LoRA+prompt is ever separately
   authorized, attach only its locked adapter to that same checked backbone.
4. Set protocol `prompt` to the actual fixed instruction/hash and record prepared
   audit/parent hashes and online encoding. Keep BF16/SDPA/native padding and
   original token tensor retention/deadline guards. No cache identity may be
   reused from unprompted inputs.
5. Keep the locked dev_select threshold and final-only reporting. Compare on
   exact matched admitted case IDs (normally all parent cases). Preserve primary
   all-events including Complete and predeclared tool-only/matched sensitivities;
   Dojo remains action gold. Neither labels nor external scores may tune this
   instruction, readout, threshold, or finalist selection.

Preparation cannot promise successful online inference. Runtime/OOM/deadline
errors must be retained and fully evaluated common sets disclosed, not converted
into silent data exclusions.
