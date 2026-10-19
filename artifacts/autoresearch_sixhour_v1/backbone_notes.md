# Alternate-backbone feasibility (CPU inspection only, 2026-10-18)

**No GPU jobs started, no processes stopped, no downloads.** Preparation/cache
implementation is `graph-safety/scripts/prepare_autoresearch_backbone.py`.
The subsequent user preference for a **small** backbone takes priority: the CLI
defaults to **0.8B**; 9B remains an explicit optional hypothesis, not a launched run.

## Locally verified candidates

Server root: `/var/local/danefedov/experiments/agent-safety`.
Snapshots below are under `models/huggingface/hub/models--<org>--<name>/snapshots/`.

| Candidate | Pinned actual revision | Text state | Layers | Weight completeness |
|---|---|---:|---:|---|
| Qwen/Qwen3.5-0.8B-Base | `dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68` | 1024 | 24 | 1/1 indexed shards; safetensors end offset equals file size |
| Qwen/Qwen3.5-9B-Base | `68c46c4b3498877f3ef123c856ecfde50c39f404` | 4096 | 32 | 4/4 indexed shards; all safetensors end offsets equal file sizes |
| Cloudflare/clef-flash | `8b2e5fd17c09fd49bc2880b805d5323ec7ff4ff3` | 4096 | 32 | 4/4 indexed backbone shards; all end offsets match; separate joint head present |

9B config has 16 attention heads, 4 KV heads, head_dim256, intermediate12288,
full attention every fourth layer, otherwise linear attention. 0.8B config has
8 attention heads, 2 KV heads, head_dim256, intermediate3584. Both context configs
say262144, but **our fixed gate remains16384**, with no truncation.

Native `AutoTokenizer(..., local_files_only=True)` CPU inspection:

* All are fast tokenizers, vocabulary length248077.
* 9B native template SHA256:
  `a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715`.
  9B tokenizer.json SHA256:
  `fe000e3ed39ed12b8d2481d527d44f93c65d37e87645d2dcc80d1bf9d50d2927`.
  Its tokenizer_config.json SHA256 is
  `3891e840d7dc5fca0af33d3a25083a735e36fe06214e3f707024820cb6b9f89c`.
  Those two tokenizer files and template match the local4B exactly. New native
  preparation is still mandatory; neither old features nor identity is reused.
* **0.8B native template differs**, SHA256:
  `273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80`.
  Its actual message-boundary compatibility is not yet checked over the corpus.
  Never borrow the4B template to force admission.
* Clef native template is in `chat_template.jinja`, not tokenizer_config; the
  loaded template SHA matches9B/4B. Its tokenizer.json differs:
  `06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523`.

Neither local Qwen Base snapshot contains README or LICENSE. Thus **license text
and training/benchmark provenance are unverified locally**, not proved absent
upstream or permissive. Clef includes Apache2.0 LICENSE, but no README. Its local
`joint_schema_model.py` describes joint typed decisions (`noul`, `choice`, `score`),
and a separate joint-head config4096→1024. This does **not** establish that Clef is
simply an instruction-tuned version of this pinned9B Base, nor clean benchmark
ancestry. No generic Base-vs-instruction-only causal contrast is currently justified.
No unmodified Qwen3.5 instruction checkpoint was present among local Qwen snapshots.

## Preparation and cache contract

CPU preparation reconstructs every labeled original assistant message stop with
the **original4B tokenizer**, matching parent IDs/spans/prefix hashes. It then
renders original messages/tools independently with the selected native tokenizer,
reconstructs new action spans, and requires each independently rendered causal
prefix to equal the new full-dialog IDs through the action endpoint. No input
instruction, text prefix, ID prepend, gold, or future-result supervision is added.
Original metadata (including keys/groups/labels) survives unchanged; parent row,
messages/tools, complete parent-state and prepared-input hashes are recorded.
One complete dialog remains one forward/state sequence, with no inter-action reset.

The input manifest may be the parent's five-list manifest, but only fixed
train_small2009/dev_select301/dev_confirm299/original_validation425 are selected;
train_full6058 is not prepared. Any failure/overflow creates an explicit rejection
record and **blocks cache/training entirely**. A matched-subset protocol requires
parent approval, not silently accepting a smaller denominator. No confirm scores
are computed by this utility (preparing their inputs is not model selection).

Prepared audit pins revision/config/tokenizer/template/weight index/file sizes;
cache additionally hashes actual frozen loaded model parameters before and after,
and binds environment, encoder/source hash, IDs, parent state, split/prepared input
hashes, last normalized text-layer output and actual hidden width. New BF16 caches
cannot be passed into the old2560-width trainer unchanged. Cache phase explicitly
requires `--authorized-gpu-cache` and supports nonoverlapping shards; it is **not
authorized by merely existing**. Interrupted partial shards are preserved; retry
uses a fresh output root rather than overwrite.

Example commands for parent adaptation after immutable-source staging:

```bash
python prepare_autoresearch_backbone.py prepare --backbone 0.8b \
  --data /var/local/danefedov/experiments/agent-safety/datasets/whole-dialog-4b-v2 \
  --splits /var/local/danefedov/experiments/agent-safety/runs/autoresearch-sixhour-v1/data/splits.json \
  --root /var/local/danefedov/experiments/agent-safety/runs/NEW-small-backbone/data

# ONLY after parent GPU/resource/time authorization; explicit independent output:
python prepare_autoresearch_backbone.py cache --backbone 0.8b \
  --data /var/local/danefedov/experiments/agent-safety/runs/NEW-small-backbone/data \
  --splits /var/local/danefedov/experiments/agent-safety/runs/autoresearch-sixhour-v1/data/splits.json \
  --root /var/local/danefedov/experiments/agent-safety/runs/NEW-small-backbone/cache \
  --shard-count 1 --shard 0 --authorized-gpu-cache
```

Use `--backbone 9b` with separate data/cache/output roots for9B; no shared identity.
These are examples only: **neither preparation over the corpus nor GPU encoding
has been executed by this subagent**.

## Minimal head/trainer follow-up (parent ownership, not changed here)

The utility includes a deferred-import CPU-only `make_head(hidden_size, arm,
attention)` factory implementing the following replacement; parent must seed
before calling it. In a new immutable trainer source, retain existing curve/event head and replace
only `input_norm=LayerNorm(hidden_size)` and
`projection=Linear(hidden_size,256)`; keep2residual Mamba2 blocks,256state width,
state64/expand2/headdim64/chunk128/readout/loss unchanged. Instantiate on CPU,
replace input modules, then move toCUDA; seed and record the actual initialization.
This changes the input projection parameter count and cannot be an identical
initialization/hash claim across dimensions. No slicing4096→2560 or zero-padding
1024→2560 is a matched representation comparison.

Parent must also replace **every** trainer feature allocation/shape gate/stress
input hardcoded2560, extend strict data/cache identity checks to the new base,
and handle the four-list small-only split manifest explicitly. Do not patch only
the head while leaving the old identity/feature loaders. Preserve both
`token_BCE + .1 event_BCE`, endpoint broad-curve control, seed and samedevAP
selection; no new threshold/readout sweep. Require original causal/reload/gradient/
longest gates and exact dev-select replay before any confirm/external inference.

## Budget / honest feasibility

No measured encoding throughput, GPU peak memory, or training wall time is available
for these newly prepared inputs. 9B indexed total_size=19,306,216,416bytes
(~18GiB including non-text weights);0.8B shard~1.75GB. Frozen9B encoding might fit
an80GBH100 but is **not gated/proven**; parameter ratio is not a runtime estimate.
BF16 feature payload is `2*hidden_size*sum(admitted tokens)` bytes: small caches
are0.4× the4B payload,9B1.6×, before container overhead. Native0.8B token lengths
may differ and must first be measured by CPU preparation.

All four GPUs were occupied at parent check. Do not preempt existing work.
After a GPU becomes free, parent should budget actual probe/cache/head timing
against the **original deadline minus one-hour final reserve**; if it cannot fit,
retain the hypothesis/preparation without a rushed launch. No deadline extension,
no promises of improvement, and no retraining based on confirm/external scores.
