# ATBench #1: one-layer Mamba + LoRA, locked validation epoch2

Same fixed case `atbench:0:1` as the previous Mamba-linear token plot.
Real readouts from the selected adapter/head, using online adapted Qwen and
the evaluation's batch1/token budget8192. No training or checkpoint selection.
Recomputed endpoints match saved benchmark outputs exactly (max error0).
Endpoints E1/E2/E3:0.00337661/0.02675928/0.02228619; max0.02675928,
so this gold-unsafe trajectory is missed at0.5. Internal token max0.07807817
is not the benchmark trajectory score (which aggregates action endpoints).

Each action independently replays its full causal prefix. Plot concatenates
only proposed-action readouts, not continuous carried Mamba state. No token or
action gold is available. Full-prefix readouts also saved in token_scores.json.
Sigmoid scores are not calibrated probabilities or proof of interpretability.
Server `runs/token-plot-atbench-1-lora-v1`, exporter snapshot in
`runs/token-plot-atbench-1-v1-launch/export_atbench_lora_token_scores.py`.
