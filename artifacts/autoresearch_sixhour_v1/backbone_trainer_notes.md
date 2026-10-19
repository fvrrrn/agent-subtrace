# Native small-backbone trainer handoff

Ready source: `graph-safety/scripts/train_autoresearch_backbone_heads.py`.
Only this trainer and this note were added in the follow-up. No GPU execution,
tests, lint, confirmation features/predictions, or external data were accessed.
Runtime/memory/gates remain **unverified until parent runs them**.

## Binding and identity

The trainer loads an **isolated instance** of `train_autoresearch_heads.py` via
`importlib.util`; it binds that instance's `make_head`, `load_data`, `features`,
`config`, `forward`, and `gate` only. Imported original/shared modules are never
modified. Existing loss, event readout, validate/raw-output, optimization,
devAP selection, checkpoint/RNG saving, and metrics code runs unchanged through
these explicit bindings. No code-string rewriting or source patching is used.

Only pinned `Qwen/Qwen3.5-0.8B-Base` revision
`dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68`, hidden1024, and native template
`273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80`
are accepted. It supports both current prepared schemas:

* `prepare_autoresearch_backbone.py` (parent is already preparing this version);
* the separately owned `prepare_autoresearch_small_backbone.py` schema, without
  changing that file. Do not mix one preparer's data with another's cache.

Checks require exact fixed four-list membership/counts, original parent manifest
hash, prepared data hash, empty rejections/exclusion reasons, no truncation,
native IDs/action-prefix hashes, preserved parent IDs/message-tool hashes, all
completed cache shards with correct counts, exact tensor-file population, shared
new-base/revision/template/tokenizer/model identity and matching per-tensor
identity/parent state. Each accessed tensor must be finite BF16[tokens,1024].
No4B weights/template/features can be silently accepted. The five-list input
manifest is reduced to the authorized four lists; a four-list manifest is also
accepted. `train_full` exists only as an internal alias of `train_small`; CLI
does **not** permit claiming a6058-dialog full-data experiment.

Head factory is the existing alternate-preparer's `make_head`: seed first,
instantiate the unchanged curve/readout/Mamba head on CPU, replace only input LN
and projection with1024→256, then move toCUDA. Both Mamba2 blocks remain unchanged.
The actual initialization and source hashes are recorded. The1024 feature and
synthetic-stress allocations are explicit; no residual2560-width input allocation
from the shared trainer is called.

## Phases and outputs

All phases require `--data --cache --splits --root --deadline-unix`.
Defaults are2epochs, broad curve, endpoint readout, no attention, original seed,
and **token_BCE +0.1event_BCE simultaneously**. Other existing readouts/curves are
exposed solely for parent-controlled conditions, not an authorized sweep.

```bash
python train_autoresearch_backbone_heads.py gate \
  --data "$DATA" --cache "$CACHE" --splits "$SPLITS" --root "$RUN" \
  --deadline-unix 1792345185
python train_autoresearch_backbone_heads.py train \
  --data "$DATA" --cache "$CACHE" --splits "$SPLITS" --root "$RUN" \
  --deadline-unix 1792345185 --epochs 2
python train_autoresearch_backbone_heads.py eval \
  --data "$DATA" --cache "$CACHE" --splits "$SPLITS" --root "$RUN" \
  --deadline-unix 1792348785 --checkpoint "$RUN/fit/selected_head.pt"
```

Gate checks exact causal invariance, exact reload, independent token/event
gradients through projection/readout/both Mamba blocks (attention if enabled),
direct-logit supervision masks, NULL endpoint support where applicable, optional
MIL algebra, and synthetic longest admitted native length backward. It does not
read held-out feature tensors or use their labels for stress. Outputs keep the
existing `gate.json`, `fit/run.json`, epoch checkpoints/RNG state, every
`fit/dev_select_epochN/tokens/*.pt`, events/metrics, `selection.json`, and
`fit/complete.json` conventions. Selection remains maximumdev_selectAP, ties
lower eventBCE, then earliest epoch.

Eval permits **dev_select only** and accepts only the completed run's selected
checkpoint/config/hash. It uses the same normal evaluation protocol and additionally
requires exact saved raw token/NULL logits, unchanged IDs/spans, and event-logit
error<=1e-6 against the selected epoch's retained outputs. A successful replay
writes `evaluations/dev_select/raw_replay.json`. No confirm/external scoring is
implemented or authorized here. Loading prepared metadata necessarily parses the
combined dialogs manifest, but no confirm feature tensors or prediction files are
opened, and no held-out labels influence fitting/model selection.

Gate/train deadline is capped at **1792345185** (originaldeadline minus3600s),
including per-forward checks during validation. Eval can use the original
**1792348785** deadline. Parent must still measure actual runtime before launch;
this interface is not permission to launch or to overrun the reserve.

## Final-lock integration caveat (parent-owned)

Existing final-lock discovery does not yet discover this candidate. Add it
explicitly only if completed, eligible, and devAP-selected. The distinct-config
signature must include **base/revision/native-template identity**, otherwise
endpoint4B and endpoint0.8B collapse incorrectly. Retain source/data/cache identity
and checkpoint hashes from this trainer's config/selection record.

Existing confirmation/external scripts assume4B/2560 and old native prepared
inputs/caches. They **must not** evaluate this selected0.8B checkpoint through
those inputs. Parent needs a separately bound native0.8B evaluator (same head
factory/features identity plus independently native-prepared external suites),
an explicitly locked final protocol, and the successful exactdev-select replay
before confirmation/external inference. If that cannot fit the original final
reserve, report the completed screen only; do not substitute4B features or hide
the missing final evaluations.
