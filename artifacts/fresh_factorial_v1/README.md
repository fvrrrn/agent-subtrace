# Fresh-init factorial v1 — training confirmed

Previous recovery and all three follow-up conditions completed. No prior queue
was duplicated. [Research and measured diagnosis](RESEARCH.md), [locked scope](PROTOCOL.md).

Detached scheduler **1471838**, first trainer **1472126**, GPU3.
First cell `lora_step`: **10 dialogs, at least 2 optimizer steps**.
Both-loss gradients passed; causal/reload error zero, loss masks passed, longest
backward 15322 tokens. The fresh shared head hash is
`711d8ffcbe2187d642f304b43634df9c311fa3d2d8577528829a35722c3501d8`.
Zero-output adapters verified. Old trained-head replay is explicitly inapplicable.

Four locked cells, two epochs each: LoRA step/broad then frozen step/broad.
Uniform BCE, no positive weighting, token + 1.0 endpoint event loss simultaneously.
Coefficient selection and its restriction are disclosed in the protocol.
`lock.json` pins all prior 8+3 results, source/data hashes and the new finite scope.
No new metrics were available at lock time. Completion is not yet established.

Server run: `runs/fresh-factorial-v1`; immutable sibling `fresh-factorial-v1-launch/source`.
Both scheduler and trainer stdout/stderr are regular files (actual fd evidence
in `first_steps_evidence.json`); viewers are independent of process lifetime.

- Live tqdm: `sh_15356ce70001jUSlu6cbru1cmX`.
- First-step inotify/pidfd observer: `sh_15357633c001XuggzSk2ok37Vy` — **completed**.
- Completion pidfd/report-copy: `sh_1535796a5001oSdQ8qwF49z03t`.

CPU diagnostic source and fresh wrappers are in
`graph-safety/scripts/streaming_followup_v1/` (`diagnose_fresh_round.py`,
`train_fresh.py`, `launch_fresh.py`, `start_fresh.py`, `view_fresh.py`).
Raw token tensors, initial/checkpoint tensors and optimizer/RNG states stay on
the server. No confirm/oldval/external labels or threshold fitting were used.
