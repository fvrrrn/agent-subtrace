# Frozen cached autoresearch trainer

Implementation: `graph-safety/scripts/train_autoresearch_heads.py`.
No GPU work, tests, lint, or formatting were run during implementation.
CPU `ast.parse` succeeded; imports and runtime gates require the server environment
and have not been executed here.

## Data/split contract

`--data` accepts the original whole-dialog directory or its `dialogs.json`.
The adjacent original `data_audit.json` and original cache completion metadata
must agree with the full immutable source hash. Subsetting is exclusively by
split membership; it does not rewrite source data or invalidate cache identity.
`--cache` accepts the original `features` directory or its parent.

`--splits` is exactly a JSON object with five keys, each a list of original
dialog key strings: `train_small`, `train_full`, `dev_select`, `dev_confirm`,
`original_validation`. Small must be a subset of full. Full/select/confirm/old
validation must be pairwise disjoint. Original validation must contain exactly
the original validation keys; all other keys must come from original train.
Empty confirm is accepted until explicitly evaluated; train/select must be nonempty.
This validates dialog separation, not semantic/family independence: main must
construct family-safe split manifests upstream.

## Readouts and objective

Main-session integration added fifth choice `leaky`: normalized signed raw-logit
leaky averaging with fixed e-folding32tokens, read at current action endpoint.
Other choices: `endpoint`, `linear_softmax`, `null`, `multiscale`.
All train `token_BCE + event_weight * hard_event_BCE` on every step; weight must
be finite and strictly positive. Broad/linear/step use the existing curve code,
safe target is zero. Average tokens within action, actions within dialog,
then four dialogs per optimizer step (actual size for partial final windows).
No truncation, history reset within dialog, label inputs, or Qwen encoding.

MIL uses the exact stable logit logsumexp(2 log sigmoid(z)) minus
logsumexp(log sigmoid(z)+log sigmoid(-z)). NULL uses token logmeanexp minus
the learned endpoint NULL logit. Endpoint uses the final action token logit.
Multiscale uses valid sliding means of raw action logits at distinct widths
min(N, {1,8,32,128}); logmeanexp within each scale, then logmeanexp over scales,
all at fixed unit temperature. Normalization removes the explicit number-of-
windows reward, not every statistical length effect. No clipping/probability
division, learned pooling parameters, or target q/N. Scan is an event readout
at the action boundary; it does not claim token-localized causal evidence.

The inherited head retains original state-dict names and native Mamba2 kernels.
Optional `--attention` is the existing single causal residual preLN SDPA layer
before Mamba. Fixed seed produces repeatable initialization and order; GPU
kernel bitwise reproducibility is not promised (environment is recorded).

## Exact launch pattern (main runs GPU commands)

```bash
python graph-safety/scripts/train_autoresearch_heads.py gate \
  --root /var/local/danefedov/experiments/agent-safety/runs/autoresearch-sixhour-v1/CANDIDATE \
  --data /var/local/danefedov/experiments/agent-safety/datasets/whole-dialog-4b-v2/dialogs.json \
  --cache /var/local/danefedov/experiments/agent-safety/runs/whole-dialog-4b-v3/features \
  --splits /var/local/danefedov/experiments/agent-safety/runs/autoresearch-sixhour-v1/data/splits.json \
  --train-split train_small --epochs 4 --curve broad --readout multiscale --event-weight 0.1
```

Use identical arguments with `train` instead of `gate`. Optional attention must
be present in both. Use a fresh candidate root for each configuration/full rerun.
No automatic cache creation or gate bypass. Gate checks exact future-suffix
invariance/reload, separate token/event gradients through readout and both Mamba
blocks (and attention if enabled), direct-logit loss support, NULL support,
and backward on synthetic features at the maximum original dialog length
(without reading confirmation features or labels). Failures preserve traceback and
partial artifacts; they never resume automatically.

After successful training, same arguments with phase `eval` and:

```bash
--checkpoint CANDIDATE/fit/selected_head.pt --eval-split dev_confirm
```

`original_validation` is another explicit eval-only option. Eval refuses any
checkpoint not matching the locked selected SHA and requires completed training.
Neither confirm nor original validation participates in epoch selection.

`fit/` contains per-epoch heads, optimizer/Python/Torch/CUDA RNG states, complete
dev-select token/NULL logits, IDs/spans, events and metrics; `selection.json`
chooses **maximum dev_select AP**, ties lower hard event BCE then earliest epoch,
as fixed by the prospective autoresearch protocol. No threshold
fit. AP, trapezoidal PR-AUC, ROC-AUC, confusion/P/R/F1/Acc at .5, class quantiles,
actual lengths, cache/template/source/checkpoint hashes, kernel versions and
environment are recorded. Heavy original feature files are opened read-only;
their pinned producer weights identity and per-row IDs are checked on each load.
Original feature files are not rehashed individually on each epoch.

Main added deadline persistence/stop guards, recall@FPR1%/5%, and support for
prompt-branch `parent_split` plus explicitly completed cache shards. First wave
failed after epoch1 because the initial PR plotting endpoint lacks specificity;
fixed `.get('specificity')` in a NEW immutable `round1v2` source snapshot.
Failed original sources/checkpoints/logits retained. No threshold semantics or
case admission tolerances changed. CPU threads limited to1 per frozen process
to avoid four independent PyTorch CPU pools oversubscribing feature validation.
