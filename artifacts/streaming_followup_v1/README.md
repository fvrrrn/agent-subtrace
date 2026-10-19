# Follow-up v1: registered, blocked before GPU work

**Update:** user explicitly requested infrastructure recovery and continuation.
Detached recovery v1 is now running, followed conditionally by follow-up v2;
see [RECOVERY.md](RECOVERY.md). The admission failure below is preserved history.

The admission scheduler actually ran as PID **861962**, then exited with code 1.
No active scheduler/watcher remains and no new GPU training was started.
The original launcher **4134159** and child **473298** were absent on direct
inspection. Parent queue status still said `running`, but the child recorded:

```
lora_step_adaptive/status.json:
status=failed, error=BrokenPipeError(32, 'Broken pipe'), epochs=0,
frozen_weights_unchanged=true, seconds=33.67833522707224
```

Two completed LoRA/broad cells passed the follow-up's completion, checkpoint hash,
selection and 886-action identity checks. Admission stopped at the third cell.
No winner was selected from the partial eight-cell queue. The observed failure
is an output-pipe error, not evidence of a bad loss or failed training quality;
the originating pipe/launcher traceback was not available in the child log.
The original queue/source/status were not modified and nothing was restarted.

## Prepared bounded round

[Protocol](PROTOCOL.md): exactly three new conditions completing the 2×2
positive-term weight × event coefficient factorial, same common warm-start,
selected configuration only, two epochs, all inherited gates.
CPU train-only preparation computed positive weight **7.733163050192017**.
The exact train IDs and data hashes are in `train_class_weight.json`.

Implementation: `graph-safety/scripts/streaming_followup_v1/`.
Registered server run: `runs/streaming-followup-v1/`.
Immutable failed-admission source: `runs/streaming-followup-v1-launch/source/`.
`authorization.json`, `status.json`, and `source_manifest.json` are copied here.
The standalone CPU preparation script is also saved in the server run directory.

Local preparation subsequently corrected the scheduler's Instruct data-audit
constants before its future `load_data` call and added a check of the prospective
class ratio; that path was not reached in failed admission. The immutable failed
source was preserved. Local protocol now records the measured ratio; the original
protocol hash in authorization refers to the preserved launch source version.

At this historical stop, continuation required resolving the incomplete original
eight cells first. Any future scheduler invocation must use a **new run/source version**
(the v1 output intentionally cannot be overwritten). Do not simply restart the
old launcher, whose root-creation guard also intentionally refuses an existing run.
For reliable future orchestration, persist process stdout independently of the SSH
viewer: the original launcher's output forwarding currently lets a broken stdout
pipe terminate its queue and leave an outdated top-level status.

No confirm/external data, thresholds, new labels, tests, lint, installs or commits.
