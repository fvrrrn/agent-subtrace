# Additional system instruction: frozen Mamba-linear, three epochs

User-authorized experiment. Frozen Qwen3.5-0.8B-Base, one Mamba layer and
binary head, linear token targets, no LoRA or ranking. Same seed/initial head,
optimizer and available training split as the original Mamba-linear.

`run_system_prompt_mamba.py` prepends a fixed English `[SYSTEM]` safety-monitor
block to the previous token sequence. This is a textual role block for a Base
model, not a claimed native system-role instruction hierarchy or a switch to
an instruct checkpoint. The old renderer already contains safety instructions;
this measures additional instruction wording/placement, not instruction vs none.
Exact prompt and token IDs are stored in each new protocol.

Separate frozen feature caches are required for train/validation and ATBench.
Original caches and running sources are unchanged. Start from original admitted
rows; record additional prompt overflows, rejecting whole ATBench cases with
any overflowing prefix. No truncation, label input or future outcomes.

Choose minimum hard endpoint validation BCE over exactly three epochs before
ATBench. Report AP, trapezoidal PR-AUC, ROC-AUC and Accuracy/F1 at fixed0.5.
Compare validation on common keys with original Mamba-linear selected epoch3
(its original run had four epochs). ATBench uses original trajectory gold,
all-events primary and tool-only/matched all-events sensitivity, not action gold.
Reference source-state hashes remain unchanged; input token IDs deliberately differ.
One seed and previously examined ATBench: exploratory, not independent confirmation.

GPU2 scheduler PID2940297 waits through pidfd for existing GPU2 evaluations,
native Kev training and its already registered downstream evaluators. It does
not stop them or touch GPU0/1/3. No GPU work begins while waiting.
Root `runs/action-models-system-prompt-v1`, immutable source sibling `-launch/source`.
Live/wait/copy `sh_14b6aad73001QW2byWIDvUu8WS`; await notification, do not poll.
