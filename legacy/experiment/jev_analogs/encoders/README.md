# Native System One encoders on the original algorithm

Only `experiment/jev_analogs/encoders/` is modified. Shared evaluator: `../replay.py`.
Both models completed all 10 events, 29 calls each, sequentially in separate processes.

Reproduce from repository root (public downloads, no credentials):

```bash
uv venv --python 3.12 experiment/jev_analogs/encoders/.venv
uv pip install --python experiment/jev_analogs/encoders/.venv/bin/python -r experiment/jev_analogs/encoders/requirements.txt
experiment/jev_analogs/encoders/.venv/bin/python experiment/jev_analogs/encoders/prepare.py
experiment/jev_analogs/encoders/.venv/bin/python experiment/jev_analogs/encoders/run_native.py von
experiment/jev_analogs/encoders/.venv/bin/python experiment/jev_analogs/encoders/run_native.py nano
experiment/jev_analogs/encoders/.venv/bin/python experiment/jev_analogs/encoders/summarize.py
```

`pins.json` pins public HF weights and primary GitHub documentation revisions;
`requirements.txt` pins installed distributions. `docs/` preserves primary READMEs,
model cards, configuration, OpenDecider API docs and the exact installed runtime sources.
`results/{von,nano}/` preserves native responses and input token audits (`native-*.json`),
shared evaluator trace/summary and runner snapshots, graph edges/maximal paths,
Mermaid graph and `report.md`. Native weights remain unmodified in `weights/`.

Von SDK 1.3.7 labels its native model `von-1.3.0` and uses shipped calibration.
Its chains are disabled explicitly; no thresholds, weights or temperatures are fitted.
OpenDecider 0.8.1 loads nano with default fp32 inference. Original XML state, ID LINK
question and SAFETY questions are unchanged; only native interface serialization applies.
Truncation is audited before inference and reflected from native responses, never hidden.
MPS is shared with other agents; recorded timings are not isolated benchmarks.
