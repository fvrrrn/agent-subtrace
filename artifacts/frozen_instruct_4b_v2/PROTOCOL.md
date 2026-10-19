# Frozen Instruct v2: authorized identity repair restart

The user explicitly authorized fixing the failed v1 cache-identity check and
restarting the same five conditions, two epochs each, on GPU 3. The original
training deadline remains `1792345185`; no automatic retries are allowed.

## Established cause

A recursive CPU comparison of the first cached row's full identity against
`complete0.json` (excluding `count` and `shard`) found exactly one difference:
`environment.capability` is the tuple `(9, 0)` in the Torch cache and the list
`[9, 0]` in JSON. JSON serialization loses the tuple container type. The repair
validates this field as two integers and restores its tuple type in the expected
identity. The original exact full-identity assertion remains unchanged.

## Admission and execution

- Source: `runs/frozen-instruct-4b-v2-launch/source`, independent immutable snapshot.
- Output: `runs/frozen-instruct-4b-v2`.
- Explicit read-only inputs: `runs/frozen-instruct-4b-v1/data` and `/cache`.
- No model download or encoding; original v1 source/results are untouched.
- CPU admission checks every one of 2310 cache entries: complete identity,
  token-ID hashes, actual lengths, BF16 shapes and finite values. It saves cache
  file SHA-256 hashes and v1 metadata hashes before any GPU stage starts.
- All original CUDA gates remain: shared initialization, causality, reload,
  gradients of both losses and both Mamba blocks, masked loss support and
  longest-dialog backward. Gate checks and CPU admission display tqdm progress.
- Sequential arms: A_broad, A_linear, A_step, B, C; two epochs each. Token and
  event losses are simultaneous, frozen Qwen, no LoRA or added attention.
- Native model: `Qwen/Qwen3.5-4B`, revision
  `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- Each GPU stage refuses to start if GPU 3 has another compute process.
- Child stdout/stderr, including real training tqdm progress, is streamed both
  to per-stage logs and the SSH launcher output. Deadline handling only signals
  the launcher's own child process group.

Background execution shell: `sh_14fce763e001YMz2R3NqBxCHfm`. Completion is not
assumed from submission; authoritative results are admission, gate, fit and
status JSON files. No confirm/external inference or threshold fitting is run.

## Confirmed launch

CPU admission passed **2310/2310**, including full identity equality after only
the documented tuple restoration. Hash manifest SHA-256:
`39d4055346ac4b9486699b444248869bbd49b7db7a4230be8412ff5aca97c1b1`.
Launcher PID `3500848`; the first GPU stage `A_broad_gate` started on GPU 3,
child PID `3504700`. Training completion is not yet confirmed.
The pidfd completion/copy watcher is `sh_14fcf2e67001lNMWSeifbIOx84`; it copies
metadata/logs to this directory after launcher exit without polling.

## Final outcome: stopped before training

The first CUDA gate passed all checks. `A_broad_train` then failed before the
first training step: the persisted gate config contains JSON list capability,
but the in-memory config embeds the restored tuple from the cache identity.
The launcher stopped without retry; both background shells exited and metadata
was copied locally. No arm trained. A follow-up local-source fix makes only the
config representation of this field a list, keeping Torch cache comparison as
tuple. That follow-up is NOT part of the immutable v2 snapshot and has not been
launched. A further version/restart must be explicitly coordinated with parent.
