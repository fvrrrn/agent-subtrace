# Fixed autoresearch data, CPU preparation

Remote root: `/var/local/danefedov/experiments/agent-safety/runs/autoresearch-sixhour-v1/data`.
Prepared with `graph-safety/scripts/prepare_autoresearch_splits.py`; immutable input
and script hashes are in `audit.json`. Full dialog JSON files stay on the server;
manifests and provenance-only action metadata are copied here.

| Explicit input file | Dialogs | Events | Unsafe events | Multi-action dialogs | Components |
|---|---:|---:|---:|---:|---:|
| `train_small.json` | 2009 | 5647 | 641 | 652 | 449 |
| `train_full.json` | 6058 | 16877 | 2245 | 1963 | 1411 |
| `dev_select.json` | 301 | 886 | 147 | 206 | 258 |
| `dev_confirm.json` | 299 | 949 | 181 | 204 | 262 |
| `original_validation.json` | 425 | 933 | 97 | 128 | 122 |

`train_small` is intentionally contained in `train_full`; all other pairs have
disjoint components, original groups, request families, explicit provenance
families and trajectories. No original train dialogs required quarantine after
reconstructing components. All 7083 original dialogs / 19645 action labels are
accounted for by train_full + dev_select + dev_confirm + original_validation.
`exact_overlap_audit.json` additionally checks dialog keys, token-ID hashes and
action keys, with zero overlap between disjoint sets.

## Actual source holdout, not invented source IDs

All components touching **AgentDoG**, **AgentAlign**, or the explicit
`StepSafe:agentdog_safety` domain in any original action provenance are excluded
from training. Component reconstruction uses all 20233 original source/action
records, including records whose actions were not admitted to whole-dialog-v2.
It preserves existing connected groups and adds normalized/entity-masked request,
explicit source-qualified family, trajectory/audit ID and exact-input links.

Dev select/confirm have 197/196 AgentDoG-attributed dialogs and 89/89
AgentAlign-attributed dialogs, respectively. They also have 118/127 dialogs
attributed to StepSafe. These counts overlap: **StepSafe repackages AgentDoG**,
so this is a held-out-origin/domain experiment, not complete disjointness of all
source names. StepSafe remains in training, but AgentDoG provenance and its
explicit StepSafe domain do not. AgentAlign has no unsafe admitted examples;
source balance cannot manufacture positive labels. Dev unsafe event totals were
reported only after label-blind salted-hash allocation; no labels, teacher
responses, or model scores were used to choose components/subsets.

Available domain fields are incomplete. `StepShield:train` is a partition tag,
not a semantic domain; missing domains are not guessed. The inherited lexical
benchmark exclusion audit does not prove semantic independence. Protected
benchmark rows were never imported; the original request-family blocklist and
protected source-name checks were rechecked. `ToolSafety` is the distinct training
source name, not the protected `ToolSafe` benchmark.

## Trainer integration

- Load the explicit JSON files or select original keys using the corresponding
  `.manifest.json`. Rows are **unchanged original dialog dictionaries**, with
  `key`, `group`, `ids`, `ids_sha256`, `tokens`, `messages`, `tools`, `spans`,
  `sources`, and the **original** `split` field. Do not filter new dev files by
  `row['split']=='validation'`: new dev rows retain `split='train'`.
- Each span retains `start`, `end`, `label`, original action `key` and
  `state_sha256`. Use labels only in loss/evaluation. Never feed manifests,
  provenance, teacher metadata, aggregate positive counts, or split names into
  the model. Metadata audit deliberately omits teacher rationales/responses.
- Every manifest lists ordered original `dialog_keys`, component IDs, exact JSON
  file SHA-256, row hashes, token-ID hashes and per-row provenance attribution.
  Top-level counts include positive events and source/domain coverage.
- Read existing features from
  `/var/local/danefedov/experiments/agent-safety/runs/whole-dialog-4b-v3/features/<original-key>.pt`.
  Parent cache markers match the original dataset hash; all 7083 files exist.
  No tensor was loaded, changed or copied during preparation. The trainer must
  retain per-row `ids_sha256` and full cache-identity verification. Do not replace
  parent completion hashes with the subset-file hash or encode new features.
- Original IDs/template/action masks/token lengths are unchanged, with no
  truncation. `train_small` totals 2455960 tokens and includes 62 dialogs with
  multiple unsafe events; `train_full` totals 7204221 tokens. Dev files total
  832049/817189 tokens, so evaluation is not as cheap as their dialog counts imply.
- Use dev_select for selection; reserve dev_confirm for locked confirmation.
  Old original_validation is previously viewed, overlapping in source domains,
  and diagnostic only; it must not be represented as new held-out confirmation.

Preparation used Python standard library only and no GPU work, tests, formatter,
linter, commits, process stops or edits to original datasets/caches.
