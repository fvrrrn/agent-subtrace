# Kev original-algorithm replay

Official source: https://github.com/jaredpalmer/kev, pinned commit
`85ccf56a0afb952323c621055b02876e8498d60c`.
Dependencies: `source/uv.lock`, installed using
`uv sync --frozen --extra serve --no-dev --python 3.13` in the source directory.
Private environment: `source/.venv`; public model downloads: `hf-cache/`.
The shared runner imports Laya even though its model is not used here. Install
its existing experiment pin with `uv pip install --python source/.venv/bin/python
--no-deps laya==0.4.6`; `environment-freeze.txt` records the complete installed set.
The initial invocation stopped at `ModuleNotFoundError: No module named 'laya'`
before loading or evaluating Kev; the pinned import dependency fixes this.
The first model-download process exceeded the shell's 120-second limit before
inference. Its replacement reuses the public cache and has no shell time limit.
The user subsequently restricted scope to the already-started Kev-4B evaluation:
do not run Kev-0.8B. A previously initiated 0.8B download is not an evaluation.
No PyPI ORM package, paid endpoints, credentials, training, calibration, prompt tuning,
tests or lint. No server is started. One model per process, sequential runs.

Run from this directory:

```sh
source/.venv/bin/python run_kev.py --model jaredpalmer/kev-4b@v1.0
source/.venv/bin/python run_kev.py --model jaredpalmer/kev-0.8b@v1.0
```

The shared `../replay.py` supplies the unchanged original XML state, original
LINK and SAFETY questions, 10 events and .2/.5 thresholds. No heldout is read.
The adapter uses official `SystemOneRequest`, `to_record`, strict native
`admit`, `MLXDecisionModel.forward`, softmax and `to_answers`.
The mapping renders choice options as `label: description`, maintaining insertion
order and original text. Probabilities have the native API's four-decimal rounding.
The shipped pointer-head temperature remains unchanged. MLX backbone is bf16;
the official pointer head runs in torch fp32 on CPU. The native prefix path uses
1024-token prefill chunks, retaining every admitted token.

Every call saves the native request and response, internal mapping, complete token
encoding, exact state/branch/row lengths, temperature-scaled logits, unrounded
probabilities and raw pointer logits reconstructed by multiplying by the unchanged
temperature. Reconstruction can carry fp32 rounding; it does not cause another
inference or change any answer. State/branch counts must also fit the original
4096/512 limits. Truncation is refused, not silently accepted.

Artifacts per model: `trace.json`, `summary.json`, `metadata.json`,
`call-001.json` through `call-029.json`, and `edge-confusion.json`. Shared replay
copies fixture and runner source into the model's result directory. Edge TN uses
all 45 chronological previous-event/current-event candidate pairs; self/future
edges are not candidates. Latencies are not isolated: other agents may share MPS.

## Outcome

Kev-4B did not reach inference: the existing unlimited-time background process
was killed by SIGTERM, as reported by the harness. Its log has no model exception;
the sender/reason is unknown. `kev-4b-v1.0/metadata.json` exists, but there are no
prediction artifacts or completed events. Graph/full/edge confusion counts are
unavailable, not zero. See `kev-4b-v1.0/failure.json` and `kev-4b-run.log`.
No new process was started following termination. The optional 0.8B public
download completed, but no 0.8B evaluation was started.

## Later, separately requested CUDA baseline

The user subsequently authorized Kev-4B again, after the Laya R-Judge baseline.
This new task evaluates **R-Judge full trajectories without a graph on llm01**,
not the cancelled original ten-event Mac replay. Same pinned v1.0 checkpoint,
official torch/CUDA readout; no 0.8B or additional versions.
Protocol/runner: [KEV_BASELINE.md](../../benchmarks/KEV_BASELINE.md),
[rjudge_kev_baseline.py](../../benchmarks/rjudge_kev_baseline.py).
