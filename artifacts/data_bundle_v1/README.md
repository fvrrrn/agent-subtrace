# Unified data bundle: training/validation separate from benchmarks

Server root: `/var/local/danefedov/experiments/agent-safety/datasets/data-bundle-v1/`.
CPU-only assembler: `graph-safety/scripts/assemble_data_bundle.py`.

## Layout

* `training/train.jsonl`, `training/validation.jsonl`: approximately 95/5 of
  usable labeled actions. `training/records.json` contains their union in the
  existing training-record format. No internal test or external evaluation rows
  are included in these files.
* `benchmarks/`: R-Judge, ATBench (including ATBench500), the **entire** StepGuard
  dev+test pool, and **all** published TS-Bench steps. Raw data remain unchanged;
  links point to their original permanent locations. Normalized StepGuard and
  TS-Bench action JSONL are provided separately, with `split=benchmark`.
* `quarantine/`: known benchmark-family overlaps, conflicting labels and teacher
  disagreements. These are not binary training examples.
* `unlabeled/pending_selected_records.jsonl`: selected actions without a complete
  pair of teacher judgments at the frozen cutoff. `label=null`.
* `unlabeled/master_candidate_pool`: reference to the master candidate pool,
  which also contains already labeled rows; it is NOT a train manifest.
* `raw_sources`: reference to all previously acquired source snapshots. An
  acquired corpus is not automatically suitable or licensed for training.
* `snapshots/`: byte-prefix snapshots of completed judge replies and the accepted
  machine-silver subset. Judges continue independently on two H100. Later replies
  do not silently change this bundle.
* `source_inventory.json`, `summary.json`, `manifest.json`: source availability,
  data counts, source/label slices, revisions via source manifests and file hashes.

## Selection and partition

Training candidates: definite StepSafe RL action annotations plus accepted
strong_pilot, strong_synthetic and the current frozen strong_bulk replies.
No weak/nonthinking teacher batch, ambiguous label, trajectory-only label,
unlabeled event or unavailable AgentDrift payload is converted into action gold.

TS-Bench is explicitly **not** a training source in this version, per user
choice. Requests from all protected benchmarks are excluded from other sources
using exact/entity-masked matching and MinHash retrieval followed by exact
five-word Jaccard >=0.85. Exclusion expands to entire connected scenario families.
This is a lexical audit, not a guarantee of semantic independence.

Exact full-input duplicates are merged while preserving all origins. Conflicting
binary labels are quarantined. Request families, source scenario groups,
counterfactual variants and lexically near-duplicate requests stay together.

Validation selection approaches 5% of **rows**, retaining whole groups, in a
fixed hash order without consulting labels or model scores. The achieved ratio
and class composition are reported. No internal test remains. The former 80/10/10
snapshot is historical and is not modified or used as the current split.

`state` is a string containing only tools → user request → observed history →
proposed action. Labels and judge metadata remain outside it. All labels here
concern the **current action**; injection-event/token danger labels are not
fabricated. No truncation or model/token admission is performed during assembly.

No training is launched. The latest user instruction prohibits it until further
explicit permission.
