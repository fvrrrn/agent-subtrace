# Published injection-related action data, staged separately

Server: `/var/local/danefedov/experiments/agent-safety/datasets/injection-actions-v1/`.
Converter: `graph-safety/scripts/prepare_injection_action_data.py`.

This is a format conversion, not a new annotation round or a training run.
It does not modify `corpus-v1`, its active annotation selection or old results.

Completed conversion and exact source-prefix audit:

| Source | Definite unique actions | Safe | Unsafe | Injection-context subset |
|---|---:|---:|---:|---:|
| ToolSafe / TS-Bench | 5401 | 3770 | 1631 | 4985 |
| StepSafe RL | 15030 | 12682 | 2348 | 2235 |
| Total | 20431 | 16452 | 3979 | 7220 |

Input conversion: 7182 ToolSafe + 22000 StepSafe records. Quarantine: 8751
records; reason counts overlap: 2506 ambiguous/intervention-only labels,
1219 known evaluation-overlap-family records, 5186 duplicate inputs.
All 20431 accepted states were reconstructed against source history/action,
all frozen manifest hashes matched. Full server artifact is about 620 MB;
only metadata and four illustrative records are copied here locally.

## Files

* `toolsafe/records.jsonl`, `stepsafe/records.jsonl`: definite published action
  labels, after duplicate/conflict and known evaluation-overlap exclusion.
* Each source's `injection_context_records.jsonl`: a subset selected by published
  benchmark-suite or trajectory risk categories. It is NOT newly verified
  injection exposure, attack success, or injection-span gold.
* Each source's `quarantine.jsonl`: ambiguous labels, duplicate inputs, conflicting
  labels, or connected families overlapping known evaluation requests. Inputs
  and original annotations are preserved rather than discarded.
* `source_records.jsonl`: immutable pre-exclusion conversion of all admitted
  source records. Do not mistake this audit file for the training selection.
* `summary.json`, `manifest.json`, `overlap_audit.json`, `duplicates.json`:
  counts, hashes, lexical overlap audit and duplicate provenance.
* `agentdrift_status.json`: AgentDrift not converted because the historically
  recorded snapshot could not be found and the public HF endpoint returned 401.

## Schema

Each JSONL row has `key`, `state`, `label`, `group`, `trajectory`, `segment`,
`split`, `source`, `event_type`, `label_scope`, `input_sha256`, `provenance`.
`state` is the JSON **string** used by the existing action-training format:

```json
{
  "tool_interfaces": "original interfaces/profile",
  "user_request": "original request",
  "observed_history": "original text or chronological event list",
  "proposed_action": "original action text or structured action"
}
```

Model-visible field order is tools → user request → observed history → proposed
action. Preserve the string or render explicitly in that order. Source labels,
reasons, risk categories, future results and taxonomy stay outside `state`.

`label=0/1` means safe/unsafe **current action**. ToolSafe score 0.5 stays
ambiguous. StepSafe only maps STOP+policy_violated=true to 1 and
CONTINUE+policy_violated=false to 0; ASK is not automatically safe or unsafe.
All named reviewed actors are retained, not just actors literally called agent.

No injection-event label is fabricated (`injection_event_label=null`). The
presence of a dangerous instruction does not make every following action unsafe.
No token-danger targets, temporal monotonic pairs or completed-trajectory gold
are inferred. No context truncation or new token-admission filtering is applied.

## Split and leakage

`split=unassigned` and `training_release=false`: no source is silently admitted
to training. These include published evaluation data. Training on them prevents
claiming independent evaluation on their original benchmark families.

Normalized/entity-masked requests and source trajectory/counterfactual families
are connected before splitting. Exact input duplicates are removed, conflicting
binary labels quarantined. Known evaluation requests are excluded by exact and
masked matching plus existing MinHash/Jaccard lexical checks, expanding exclusion
to whole connected families. This is not proof of semantic independence.

StepSafe's canonical rebuilt RL task admission is preserved; annotations outside
those task IDs are not silently mixed in. Published labels can use hindsight;
format conversion does not establish semantic correctness or human-gold quality.
