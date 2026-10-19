# Frozen-Mamba hard-target controls

Authorized addition on GPU2, scheduler PID **1783083**.
Server: `/var/local/danefedov/experiments/agent-safety/runs/action-models-hard-targets-v1`.
Immutable code: sibling `action-models-hard-targets-v1-launch/source`.

Two concurrent conditions, 8 epochs each, no LoRA and no ranking:

* `mamba_token_hard`: target y=0/1 on every token of the annotated proposed
  action, BCE averaged within each action then over minibatch actions.
* `mamba_endpoint_hard`: BCE with y=0/1 only on the last action token.
  Gradient still passes through Mamba over the complete observed prefix.

Context tokens are not automatically safe or directly supervised in either arm.
The state resets between independent full-prefix examples, as in the parent run.
No exponential/cosine target anchors are used in either condition. `protocol.json`
inherits the parent's descriptive curve fields for provenance; its new
`conditions`/`objective` fields and each arm's explicit run arguments define the
hard-target experiment, not those inherited curve descriptions.

The prepared records and BF16 Qwen token cache are read-only symlinks to the
existing action-models-v1 artifacts. Same 15218 train / 848 validation examples,
871 original train length rejections, same splits, spans, data hashes, seed,
one-layer Mamba/head initialization, optimizer, batching and validation-BCE
checkpoint selection. No retokenization, fresh Qwen extraction, data resampling
or changes to the parent run. Original Kev/random-head baselines are reused.

CUDA audit on one safe and one unsafe training action checked finite nonzero
Mamba/readout gradients and that direct loss gradients are restricted exactly
to the specified action tokens or endpoint. Initial-head hashes match both
new conditions and the running original frozen-Mamba exponential arm. Audit
gradients are discarded before training; no preflight optimizer step is reused.

The result is a validation-selected exploratory comparison, not independent test
evidence. Simultaneous jobs share GPU2 and storage, so times are not isolated
architecture-speed benchmarks. Judges and all original training jobs remain
untouched. Predictions/checkpoints/metadata are copied locally on completion;
large records/features remain in permanent server storage.
