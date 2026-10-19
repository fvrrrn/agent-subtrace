# Recovery v1 and follow-up v2

**2026-10-19 update: complete.** All six recovered conditions and three follow-up
conditions finished; scheduler 886673 exited. Both recovery and follow-up terminal
statuses are `complete`. All 8+3 checkpoint/identity checks were repeated before
the next finite round. Follow-up comparison is copied to `followup_v2/`;
see `../fresh_factorial_v1/RESEARCH.md` for metrics and the next locked experiment.
The launch evidence and watcher IDs below are historical, not active-process claims.

Detached recovery scheduler actually launched as **PID 886673** on GPU3;
first recovered trainer is **PID 886886**, `lora_step_adaptive`.
Server source: `runs/weighted-broad-recovery-v1-launch/source`.
Server output: `runs/weighted-broad-recovery-v1`.
Conditional follow-up output: `runs/streaming-followup-v2`.

The scheduler compares every original source file hash with the immutable v3
snapshot before recovery. No original trainer, model, loss, gate or native kernel
configuration changes. Completed first two conditions are reused read-only;
the remaining six are freshly executed from their original common warm-start,
seed and two-epoch configuration. These are recovery executions, not six new
experimental conditions. All eight must complete with matching action IDs before
the locked three-condition follow-up can start. There are no automatic retries.

The original third condition has no `warmstart_gate.json`, `protocol.json`,
`gate.json`, checkpoint, optimizer state or training history. Its saved frozen
hash audit is unchanged. In the immutable trainer, warm-start replay precedes
protocol/gate, and optimizer creation follows the saved gate. Thus zero optimizer
steps is supported by control flow and artifacts, not merely `epochs=0`.
Original failure is `BrokenPipeError(32, 'Broken pipe')`. Its exact traceback
was not persisted; do not claim the initiating SSH event was directly observed.
The old launcher used a child stdout pipe, forwarding synchronously to its own
stdout without handling broken pipes or persisting a launcher failure status.

New scheduler stdin is DEVNULL, stdout/stderr a regular file, and session detached
using `start_new_session=True`. Each trainer writes directly to its own regular
log file, not a pipe to the scheduler or SSH. Viewer loss cannot terminate these
processes. An independent tail viewer streams real tqdm logs; completion waits
use pidfd, and first-step observation uses inotify plus pidfd.

Live tqdm viewer: `sh_150af4747001Xmcwp3Hlp7c33H`.
First-step observer: `sh_150afaadd001UlHQtTBE1IOLu1`.
Completion pidfd/report-copy watcher: `sh_150b08c65001v2y8he6MKoazap`.
The first-step observer completed: **11/2009 dialogs, at least two optimizer
steps**. `first_steps_evidence.json` is copied locally. Strict replay passed on
301 dialogs / 886 actions with raw error zero; token/event gradient gates passed,
causal/reload error zero, loss support passed, longest backward 15322 tokens.
Actual `/proc` fd targets confirm DEVNULL stdin and regular-file stdout/stderr
for both scheduler and trainer. The live viewer and completion watcher remain
independent; first-step observer is complete, not an active watcher.

Reports retain exact hashes and paths for reused/recovered conditions through
`completed_manifest.json`; the composite `completed_eight` directory contains
read-only references and its own status, never modifies the original status.
The original failed follow-up v1 remains preserved. Follow-up v2 uses the local
prepared objective and train-only ratio 7.733163050192017, locks all three new
cells after eight valid outcomes, and writes full comparison/source slices.
