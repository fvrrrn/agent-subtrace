# Frozen Instruct 4B v1 — cache complete, head gate failed

The pipeline stopped at `A_broad_gate` before any optimizer step. No automatic
retry was performed. The pidfd watcher completed and copied light artifacts.

The frozen cache completed all 2310 dialogues / 3,288,009 tokens. Encoding took
217.3566 seconds (excluding weight download/loading); frozen weights matched
before/after. Backbone future-token perturbation gate passed with error 0.
The initialized A head matched the Base control hash before the failure.

Failure cause is a metadata serialization mismatch, not a model prediction:
`train_autoresearch_heads.features` compares the saved cache identity with its
JSON completion metadata. `environment.capability` is the tuple `(9, 0)` in
Torch payloads but the list `[9, 0]` in JSON. A CPU comparison found this to be
the only differing field. Both refer to the same H100 CUDA capability.
Fixing this requires an explicit new source version that restores the JSON
capability to a tuple for exact feature-identity comparisons (and preserves
all other identity checks), followed by rerunning the mandatory head gate.
The cache, failed source, gate traceback and all existing artifacts remain
unchanged. No head training has run, and no gate tolerance was loosened.

Launcher PID **3391387**; initial child PID **3391931**, stage `cache`, GPU 3.
At launch confirmation the child was downloading the pinned model weights;
CUDA encoding and head gates had not yet been confirmed. The pidfd watcher
`sh_14fbf3559001BD2S7mk95nELfF` waits for launcher termination and copies light
artifacts/checkpoints locally, excluding cache, token tensors and dialogues.

Native admission passed: **2310/2310 dialogues, 6533 actions**, zero exclusions;
maximum length 15322. Train-small/dev-select counts remain 2009/301. Instruct
and Base native template hashes are identical:
`a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715`.
The four reused core source files are bit-identical to the Base B/C snapshot;
see `core_source_audit.json`. The CUDA gates still must verify the initialized
shared head against Base and both loss gradients before training.

Four new versioned entry points are implemented in `graph-safety/scripts/`:
`prepare_frozen_instruct_4b_v1.py`, `cache_frozen_instruct_4b_v1.py`,
`train_frozen_instruct_4b_v1.py`, `launch_frozen_instruct_4b_v1.py`.
They reuse the existing whole-dialog tokenizer admission, encoder, residual
Mamba/head architectures, losses, metrics and AP-selected training loop.
Training is restricted to the five authorized conditions and two epochs.

Initial CPU preparation failed before rendering any example because the
default Hugging Face cache did not contain the Base tokenizer. Its source,
empty data directory and traceback are retained on the server as
`frozen-instruct-4b-v1-launch/preflight-failure-source`,
`frozen-instruct-4b-v1/preflight-failure-data`, and `preflight-failure.log`.
The corrected preparation resolves both tokenizers through the existing
explicit permanent cache path. This is an operational path correction,
not a model/data/protocol change or an experiment repeat.

CPU imports and syntax parsing of the four entry points succeeded. The first
reference-file check identified the control gate's actual location:
`runs/sound-anomaly-4b-control-v1/candidates/control_a/gate.json`.
No CUDA gate or training success is implied by these checks.

The scheduler refuses an occupied GPU 3, runs one child at a time, enforces
the original training stop time, and does not retry failed stages. It never
signals other processes. All CUDA stages depend on complete native-prefix
admission and a separately hashed Instruct cache.
