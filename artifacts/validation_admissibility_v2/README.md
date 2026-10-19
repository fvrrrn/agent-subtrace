# Validation admissibility v2 — CPU audit, 2026-10-18

## Result

| Split | Before dialogs/actions (safe/unsafe) | After dialogs/actions (safe/unsafe) |
|---|---|---|
| dev_select | 301 / 886 (739/147) | **301 / 886 (739/147)** |
| original_validation | 425 / 933 (836/97) | **421 / 917 (824/93)** |

Two exact causal-token-input contradiction groups were found, both in original
validation: `cfrl-appeal_judge` and `cfrl-student_humiliation`, each pairing
`unsafe_execution:4` with `safe_correction:4`. Four conflicting actions (2 safe,
2 unsafe) cause exclusion of their four whole dialogs, including 16 supervised
actions (12 safe, 4 unsafe). No label was chosen as correct or relabelled.
All members of both conflicting groups are removed. Retained dialogs retain their
complete context, spans, IDs and labels; no within-dialog event deletion occurs.

Both pairs originate in StepSafe revision
`05ea723c3d9903c7f68152c6037198292faf890f`,
`rebuilt/rl/trajectory_annotations.jsonl`, rows 3302/3303 and 3525/3526,
`cf_relabel_v2_release_convention`. `contradictions.json` contains exact causal
IDs including the current action, complete keys and provenance. Hash equality
was followed by direct token-list equality; future tokens are excluded.

No dev-select defect was demonstrated. Source history/current-action/tool
preservation, source-state hashes, original labels, full native serialization,
and independently reconstructed source causal prefixes passed for all 1,819
audited actions. No current `proposed_action.tool_result` was present. These
checks establish preservation of supplied context, not completeness or correctness
of the underlying source's permissions, intent or annotations. Ambiguity alone
is not an exclusion criterion; there was no manual gold adjudication.

## Files and integration

Server data: `/var/local/danefedov/experiments/agent-safety/runs/small-backbone-comparison-v2/data`.
Authoritative manifest: that directory's `manifest.json` (also copied here).
Script: `graph-safety/scripts/prepare_validation_admissibility_v2.py`.
Server source: sibling `admission-source/`; no existing script was changed.

Five-list `splits.json` retains original key strings and ordering, with counts:
**train_small 2009, train_full 6058, dev_select 301, dev_confirm 299,
original_validation 421**. Train and confirm JSON files are byte-identical copies.
Confirm data was copied/hashed as opaque bytes, not parsed. The original bundle
was loaded to select audited action keys; nonselected labels were not inspected.
Original whole-dialog file was hashed only. No scores/predictions were read.

**Downstream explicit admission extension is required:**
`prepare_autoresearch_backbone.py` currently asserts original_validation=425.
Do not weaken this assertion generically: authorize this manifest's 421-key set
and hash for the new comparison. Other legacy fixed-count consumers must likewise
use this explicit version. This audit does not modify trainer code.

The new protocol preserves dev-select AP selection, ties by event BCE then compute,
and deadline 1792348785. Existing inspected dev-select is not an independent test.
All new backbones must use the same admission; further tokenizer-specific
rejections require explicit matched-case handling. Oldval numbers on 421 dialogs
are not directly matched to previous 425-dialog results. Dev-select is unchanged.
Known train noise (39 conflicting full-train actions/19 groups; small 6/3) is
retained, not silently cleaned.

## Hashes and retained evidence

`manifest.json` records exact old/new file hashes, script/helper hashes, original
records and dialogs hashes, template and tokenizer revision. `audited_actions.json`
contains per-action provenance, source-record hashes and causal-ID hashes.
Server `original_validation.excluded.json` retains all four exact removed dialogs;
`excluded_source_records.json` retains all 16 excluded source records locally and
on server. `issues.json` gives exact exclusion reasons/keys. Heavy retained full
data remain on the server. No GPU, annotation API, tests, lint or training ran.

An initial offline-cache-path failure left an empty directory, retained as
`admission-empty-offline-cache-failure`. A preliminary audit incorrectly rebuilt
native tool argument key ordering before reproducing the original trie's sorted
JSON serialization. Its **invalid** output is preserved under
`admission-rejected-audit-key-order-v1`; do not use its exclusions. The corrected
audit exactly reproduces the original trie serialization and finds zero such
serialization defects. Only final `data/manifest.json` defines this release.
