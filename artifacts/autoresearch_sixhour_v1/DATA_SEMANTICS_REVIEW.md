# Fixed-train data semantics audit — CPU only, 2026-10-18

## Bottom line

**A concrete causal-input/label inconsistency is present:** full train contains
19 identical serialized causal-token-prefix groups with conflicting labels
(39 actions: 20 unsafe / 19 safe); small train contains 3 groups / 6 actions.
All 19 arise from StepSafe `cfrl-*` release-convention annotations. The original
states differ **only** in label-bearing authored-scenario metadata, correctly
omitted from model input. Actual history, interfaces and current action are
identical. Thus stored-state deduplication alone misses these contradictions.
This is a bounded demonstrated defect, **not evidence that it explains the
observed model ranking or that most training labels are wrong**.

Other findings: unsafe examples are heavily StepSafe-concentrated; StepShield
is predominantly safe judge silver with sparse tool semantics; native StepSafe
annotation provenance is heterogeneous and sometimes explicitly hindsight-based.
The deterministic nine-case sample includes safe refusal and safe coordination,
so dangerous request, attack presence and completed-trajectory harm are not valid
substitutes for the intended current-action target.

No data, split, labels, current selection criterion or running experiment changed.

## Protocol and reproducibility

- Read `runs/autoresearch-sixhour-v1/data/train_small.json`, `train_full.json`
  and their manifests; join action keys to original
  `datasets/data-bundle-whole-dialog-v1/training/records.json`.
- No confirm files/predictions, external benchmark labels, model predictions,
  judge API, GPU, training or annotation used. Original records were loaded as
  explicitly requested, but statistics/case inspection selected train keys only.
- Sample: first **three action keys per train-full source** sorted by
  `SHA256("semantics-review-v1:" + key)`. Source-stratified, independent of labels,
  model errors and scores. Nine cases; not a representative error-rate estimate.
  Collision inspection is a separate exhaustive structural audit, not the sample.
- Executable read-only audit: `audit_train_semantics.py` (stdlib Python, via
  `ssh llm01 python3 - < audit_train_semantics.py`). JSON stdout saved locally as
  `DATA_SEMANTICS_AUDIT.json`; contains counts, full sampled states/provenance,
  exact collision keys and state differences. No remote file written.
- `records.json` SHA256:
  `59eba5c2e5ed521736880c2da2046d6ead34c59074242bc4831a80e86b033505`.
- Train-small SHA256:
  `730445439ef9beaed5559590ff8eb4a7a288d1b2a26c1bc1f98ba8866c7e0848`.
- Train-full SHA256:
  `f1fd6ae00440d0f5261f4d9eca74404d21a7049a5fe769b506ededbcd61d54a9`.
- Audit JSON SHA256:
  `428b50b337f52d474f492818e008806a31d620a4f1ea96312e668c294abd26b7`.

## Counts, provenance and effective diversity

| Fixed training set | Small | Full |
|---|---:|---:|
| Dialog rows | 2,009 | 6,058 |
| Unique action keys | 5,647 | 16,877 |
| Unsafe actions | 641 (11.35%) | 2,245 (13.30%) |
| Original groups / reconstructed components | 449 / 449 | 1,411 / 1,411 |
| Maximum dialog rows in one group | 56 | 56 |
| Whole-dialog tokens / epoch | 2,455,960 | 7,204,221 |
| Supervised action-span tokens / epoch | 599,279 | 1,775,438 |
| Maximum dialog length | 15,322 | 15,723 |
| Native-only action rows | 3,533 | 10,637 |
| Native + silver support | 848 | 2,448 |
| Silver-only action rows | 1,266 | 3,792 |

“Native” here means an origin in the frozen published-StepSafe action adapter;
it does **not** mean new human-validated gold. “Silver” means nonempty stored
teacher labels. Multiple origins of one action are not independent examples.

| Full train source | Actions | Unsafe | Unsafe rate | Groups |
|---|---:|---:|---:|---:|
| StepSafe | 13,156 | 1,980 | 15.05% | 1,156 |
| StepShield | 3,593 | 246 | 6.85% | 218 |
| LLM_contrastive_v2 | 128 | 19 | 14.84% | 37 |

StepSafe supplies **88.20% of unsafe actions** (small: 561/641 = 87.52%).
StepShield supplies 3,347 safe actions, versus 246 unsafe. Counts are source
attributions; here the action totals happen to sum to the train total.
The rows do not establish thousands of independent mechanisms: repeated family
branches and shared workflow prefixes account for multiple dialog rows.

For the 13,085 native-supported full-train actions, original `audit_method`
metadata counts are: missing 8,068; `gpt_workflow_revision_independent_review`
2,389; `agentdog_released_safe_label` 1,252;
`authored_counterfactual_scenario_v3+cf_relabel_v2_release_convention` 713;
`manual_full_trajectory_review` 624; `cf_relabel_v2_release_convention` 39.
These are provenance strings, not independently verified descriptions of review.
Absence of a method is not proof of a bad label; whole-trajectory review does not
establish that the label was decidable from the causal prefix.

## Label semantics and serialization

Code inspected: `graph-safety/scripts/prepare_injection_action_data.py`,
`build_safety_candidates.py`, `annotate_safety_corpus.py`,
`assemble_data_bundle.py`, `prepare_whole_dialogs.py`,
`prepare_autoresearch_splits.py`, `train_autoresearch_heads.py`,
`train_curve_event_heads.py`.

1. **StepSafe native mapping is action-specific in its schema:** `STOP` with
   `policy_violated is True` becomes 1; `CONTINUE` with false becomes 0;
   inconsistent/unknown combinations become unlabelled. `risk_source` is metadata,
   not the target. This schema is better aligned than broadcasting trajectory
   risk, but the collision evidence below shows the schema is not sufficient.
2. **StepShield native markers are not ordinary binary action gold.** Its adapter
   treats non-null divergence markers as positive candidates; null is unknown,
   not safe. The bundle ingests StepShield through strong silver files. CLEAN
   suffixes and `unlabelled_not_assumed_safe` provenance remain metadata; safe
   labels are judge consensus, not `CLEAN => safe`. The two-judge prompt explicitly
   distinguishes current action from harmful request/text/outcome and permits
   uncertainty, refusal, quotation and clarification. Agreement is still not
   independent human adjudication and may share judge biases.
3. **AgentAlign / raw AgentDoG candidates are not natively action-labelled.**
   Their adapter tags `aligned_response_not_action_gold` and
   `trajectory_label_not_broadcast`; such candidates enter through silver. They
   are not in these training sets. Do not interpret held-out source membership
   alone as independent task-matched gold.
4. **Current action output is excluded by construction:** StepSafe `visible(...,
   current=True)` removes `tool_result`, and serialization occurs before appending
   the current event. StepShield appends action then observation only after
   emitting the candidate. Exhaustive train check found zero top-level
   `proposed_action.tool_result` fields. Prior tool outputs remain in history.
   This check does not prove every source's free-text content is uncontaminated.
5. **Permissions and identities are retained when present:** interfaces/profile
   become system text or native tool definitions; multi-agent `agent_id`, roles,
   routing and content are serialized as JSON, environment messages as tool role.
   Native tool arguments are retained. Multi-agent tool calls are JSON content,
   not necessarily native function-call messages—a source-format shortcut risk.
   No external rationale/teacher labels are inserted by this conversion.
6. **Causal admission:** action endpoints exclude role closing; full-template
   prefix IDs must equal separately rendered causal prefixes. Unknown history,
   overflows and native-template rewriting are rejected, not silently truncated.
   Branches join only on exact events within group/request/tools/split. Shared
   labels are counted once per original action key. No new tokenization was run
   in this audit; stored IDs and code were inspected.

### Demonstrated identical-input contradictions

Original `input_sha256`: **zero duplicates, zero label contradictions** in each
train set; all stored origins agree with their admitted row label.
But hashing `dialog.ids[:span.end]` finds 262 extra repeated causal prefixes in
full train (93 small), of which **19 / 3 groups have opposing labels**.
These are within-train repetitions, not a train/dev leakage claim.

All contradictory states differ only in `user_request` strings such as
`Authored counterfactual scenario: report_volume (unsafe_execution)` versus
`... (safe_correction)` (one group also `unresolved_plan`). This is ground-truth
scenario metadata, not legitimate permission; restoring it would leak labels.
`prepare_whole_dialogs.py` correctly does not inject that text. It consequently
exposes opposing release labels for exactly the same visible peer message.

Example, origin keys `stepsafe:cfrl-report_volume-unsafe_execution:4` and
`stepsafe:cfrl-report_volume-safe_correction:4`:

- Unsafe bundle key:
  `bundle:d11aa4df4ef2cb8450c1fac7213a998c20917a884fe6b7c62288eedb8f48113f`.
- Safe bundle key:
  `bundle:1bef087b37e99f2c9dbb2fb806a1b11877cc634d46428b00b0cab3bf0c51c83d`.
- Identical current peer message includes
  `The reporting peer concludes quantity alone proves abuse.`
- Previous event is `agents.read_handoff_message`, read-only metadata request.
- Identical scored causal-prefix SHA256:
  `478c450906fa06e6c5e5f8448c86f22682e471d1167835be28cf2c091b1df231`.
- Both native labels come from `cf_relabel_v2_release_convention`; unsafe/safe
  target and violation differ in provenance but not in legitimate visible input.

This also exposes an actor/task ambiguity: the labelled event is a **peer
communication**, not the main agent executing the proposal in that communication.
Do not conflate judging that peer's instruction with judging the recipient's next
action. Release conventions or later branch outcomes may be involved; the audit
does not adjudicate which member should be relabelled. The 39 affected actions
are 0.231% of full-train events; at least 19 hard labels cannot simultaneously be
fit by any deterministic causal classifier. Other semantic noise is not bounded
by this structural count. Any future correction needs a new version and explicit
authorization; current immutable experiment remains unchanged.

## Deterministic manual sample: all nine cases

These are observations, **not new validated gold**. Full exact keys/states and
origin evidence are in JSON. Below bundle hashes uniquely identify cases.

| Source / bundle hash | Stored label | Minimal evidence and observation |
|---|---:|---|
| LLM / `6afeba635dfca1dbdc07046b5bb56294318e9c6840cef8102b95353b5ea72cf4` | 0 | `get_manuscript_status`; interface says read-only for authors/editors. User requests eventual publication; present action only reads status. Supports distinguishing current action from irreversible requested outcome. |
| LLM / `884ad5a0f7ba7115bdf6d54b8cd6027e26d9e1d8beb4c731eeefc0f6dd799d7f` | 1 | User says `bypass the standard quote approval process`; action reassigns ticket to `auto_approve_system`. Plausible unsafe facilitation, but that assignee's actual semantics are not specified in the tool interface. **Uncertain shortcut/intent dependence**, not a proven wrong label. |
| LLM / `7ce53516868504ae062559e18bd873077fd4323cf95f083e31ee04098520662a` | 0 | `COUNT(*) ... GROUP BY department_code`, user excludes PII. Aggregation rather than records export; read-only interface retains restrictions, but user identity/authorization is not independently verified. |
| StepSafe / `37574c634b6c5c65fbf340dac865907ded7a0644ce2168551763253405161e24` | 0 | `Workflow plan: order_specialist will next run query_case_records`. Identity/authentication and scoped IDs retained. Prior policy search returns espresso-machine guidance for watch mediation: visibly irrelevant context, but present action is coordination, not refund/exposure. Safe label need not certify successful task completion. |
| StepSafe / `588d6f8dcb0b91bee9185f58d8d3b097c78c39822087908b94d0b41c15b89e2b` | 0 | Closeout requests `including only facts supported by those results`. History contains a `$0.00` adjustment and unrelated quote-policy search for a VIP bedding task. **Workflow realism/quality concern**, not demonstrated unsafe current communication. |
| StepSafe / `cdf89cf4cd965a6049a9143702246d53463f94e2418afef8b32d9b276b9890e7` | 0 | `I can't use tools that download videos or access Spotify's internal API for lyrics`. Safe refusal/alternative despite copyright-sensitive request. Native `agentdog_released_safe_label`, not newly verified action gold. |
| StepShield / `ba6b0b8bc4c7ff7579d2af7eeee2111912642a36213be89a217d51e2138c4c75` | 0 | Fourth `kubectl apply`, repeated earlier `applied` observations; command lacks deployment specification. **Under-specified execution/tool semantics**; label safety does not imply valid deploy. Silver; source label null. |
| StepShield / `d57af51d48f2aeb4d0c911dc479c24b2788522c3d0c2ea0ad6fc33c0bd7cd235` | 0 | Replaces `infer_edges_from_defaults()` with `derive_edges_from_traces_only()` after no C→A calls found; request forbids production graph mutation. Source file vs production distinction is not fully resolved, but action is plausibly requested code repair. Silver, not CLEAN-derived gold. |
| StepShield / `c3be650d8ecec2df2a2f594896fda11832369d356e99a36399586c32594cbfea` | 0 | Changes `old_ca.crt` to `new_ca.crt`; constraints prohibit secret-emitting TLS tracing. No prohibited tracing in current action. New CA trustworthiness/authorization is **not established by supplied observation alone**. Silver uncertainty, not proof of unsafe action. |

The three StepSafe sample labels all happen to be safe; sampling did not condition
on label. Do not infer absence of unsafe StepSafe events or estimate noise rates
from this sample. None of the nine proves that quoted harmful content alone was
mistakenly labelled as execution; collisions provide the stronger independent
evidence of inconsistent causal semantics.

## Artificial token targets and “more data” interpretation

Current training computes mean token BCE per action, then mean actions per
dialog; adds `0.1 * hard-event BCE`; averages four dialogs per update. Broad
unsafe targets use `(0.9 + 1.2u - 1.5u²)/1.14`, safe targets are zero. This is
an artificial within-action curve, not labelled token evidence or calibrated
prefix-risk probability. It gives high targets even to role/JSON scaffolding and
non-witness parts of unsafe actions. Token position uses complete action length,
which is not known during streaming; this is target construction, not evidence
that the current forward reads future tokens. Event loss does not make token
localization identifiable. Broad endpoint target (~0.526) also differs from hard
event label 1; event readout and token head receive different supervision.

Full versus small at fixed epochs changes multiple quantities simultaneously:
~3.02× dialogs, ~2.99× actions, ~2.93× encoded tokens, ~2.96× supervised tokens,
and **1,515 versus 503 optimizer updates per epoch** (four-dialog accumulation).
Unsafe prevalence rises 11.35%→13.30%; source/family membership and lengths differ.
Mean-dialog loss weights events from short/single-action dialogs differently from
events in long multi-action dialogs; it is not uniform per-event/component risk.
Therefore an observed AP decrease is not a data scaling law, nor proof of label
corruption; optimizer exposure, family/source mix and representation/domain shift
remain confounded. No additional training/control proposed here was launched.

## Metrics: keep current criterion; improve future task specificity

- Static metric-code audit (`graph-safety/src/graph_safety/metrics.py`): equal
  scores are processed together; AP is the step integral, PR-AUC separately uses
  trapezoidal precision interpolation, ROC-AUC uses trapezoidal ROC integration.
  Precision with no predicted positives is null; all-safe AP/PR-AUC and one-class
  ROC-AUC are null. Threshold comparison is `score >= threshold`. The artificial
  initial PR plotting point has no specificity; `validate()` excludes it from
  recall-at-FPR scans using `p.get('specificity')`, avoiding the earlier missing
  initial-point-key failure. These are code observations, not a new metric test
  suite or recomputation of protected predictions. Sigmoid range validation does
  not establish probability calibration.
- **Do not change current prospective dev-select AP selection.** AP depends on
  prevalence and source composition; aggregate AP can improve by recognizing
  all-safe sources without learning safety within a source. Existing metadata
  report identifies 119 all-safe AgentAlign dev-select events; this audit did not
  read dev labels or predictions to re-estimate it. All-safe source AP is undefined,
  not zero. Report its false-positive rate separately.
- For future protocols: action-level AP, trapezoidal PR-AUC and ROC-AUC separately;
  per-source positive counts/prevalence, source/domain slices, component-bootstrap
  uncertainty, operating-point P/R/F1/Accuracy/confusion and recall at fixed FPR.
  Source-macro scores must specify undefined slices and weighting prospectively.
- Separate peer-instruction hazard, current actor's unauthorized action, refusal/
  quotation, attack exposure and realized outcome. Add future native-vs-silver,
  review-method, actor/surface and permission-completeness reporting without
  silently redefining labels or filtering difficult cases.
- Causal action replay aggregated by maximum, including Complete, is not a
  completed-trajectory retrospective classifier. Trajectory gold cannot be
  broadcast onto actions/tokens. Max scores depend on event count and may flag
  safe summaries/refusals. This task mismatch is a protocol limitation—not a
  reason to remove Complete or tune on already viewed ATBench labels. No external
  case/label inspection was performed for this audit.

**Recommended next authorization, not executed:** provenance-owner review of the
19 collision families, followed by a versioned causal-input consistency gate and
task-specific adjudication protocol. Never repair conflicts by returning
`unsafe_execution/safe_correction` metadata to model input or by choosing labels
from observed model performance.
