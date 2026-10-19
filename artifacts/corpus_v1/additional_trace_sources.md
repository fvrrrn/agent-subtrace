# Additional execution-trace sources

Discovery requested after inspecting the short StepSafe examples. Read-only research:
no ongoing corpus selection, teacher protocol, model weights or train split changed.

Execution traces in benchmark sandboxes are not production-user logs. A successful
task is not automatically safe; failed tasks are not automatically unsafe. Repository
file counts below are discovery counts, not deduplicated trajectories or step gold.

## Highest-priority sources

| Source | Verified evidence | Constraints / intended use |
|---|---|---|
| [OpenAgentSafety](https://github.com/Open-Agent-Safety/OpenAgentSafety) | Revision `2fdd4057c8dae6e776999a55a5766a862e96f174`; evaluation trajectory JSONs already in the acquired source. Inspected `evaluation/claude-sonnet/traj_safety-abusive-apology.json`: 14 events, timestamps, action/args, observations and chat with simulated NPC. README links traces for five models. | Real execution in controlled workplace infrastructure, not production logs; some NPC responses simulated. Recover task.md/tool schemas, strip repeated model_response/metrics and evaluator-only data. Existing evaluation judgments are not step labels. Repository MIT; audit data terms and v1.0/v1.1 evaluator differences. |
| [AgentDyn](https://github.com/SaFo-Lab/AgentDyn) | Revision `5353cf7615b135cace8d07c8f12dac53a16b6db3`, MIT. Recursive GitHub tree exposes 34298 run-log candidate JSONs but is **truncated**. Inspected a run with 16 messages, actual assistant tool_calls and tool responses, utility=true/security=false. | 60 tasks and 560 injection cases are task counts, not trajectory counts. Dynamic mock applications, not production. Group all models/defenses/attacks of a task family; outcome flags cannot be broadcast to every action. |
| [AgentDojo](https://github.com/ethz-spylab/agentdojo) | Current inspected revision `089ed468cf3ed0322acc66b0211f26d9d90dbf60`, MIT; complete GitHub tree has 36679 JSON trace-file candidates under runs. Official results explorer exposes runtime traces. Earlier snapshot already acquired. | Stateful simulated tools. Large overlap with TS-Bench/StepSafe/ASSE/ATBench: provenance-aware exclusion required, not a new independent dataset by default. Logs not yet adapted into the current action pool. |
| [AgentLAB](https://github.com/tanqiujiang/AgentLAB) | Revision `36f58e60c36bbd6d5b8e61d50d7db7d9ea7258d7`, MIT. README and published results cover chaining, drifting, memory poisoning and task injection. GitHub tree is truncated; exposed files include pipeline results and tiny outcome summaries. | Promising long-horizon source; full causal transcript availability must be checked per track. Do not count result-summary files as trajectories. Some environments inherited from AgentDojo/SHADE/ASB. |
| [RAS-Eval](https://github.com/lanzer-tree/RAS-Eval) | Revision `977c528536776312caf294a64b4c9b76a7b248c8`; complete tree exposes nine JSONL files in data/logs, including eight model files and guard_response. | Supports real APIs and simulated tools. 3802 is attack-task count, not verified trace count. Need inspect log schema, execution mode and license (GitHub API license null). No tools executed during research. |

## Broader real-execution corpora without safety gold

* [Nebius SWE-rebench OpenHands](https://huggingface.co/datasets/nebius/SWE-rebench-openhands-trajectories):
  revision `35455389ab51bf5e2306bfd436ef72d0f98bf882`, CC BY 4.0. Card reports
  67074 trajectories, 1823 repositories, mean 64.3 turns. Full role-tagged
  history, function calls, tool outputs; separate tools.json and config.toml.
  Real GitHub issues and executable environments, model-generated behavior.
  resolved/test outcome measures correctness, not safety. Narrow coding domain.
* [Nebius SWE-agent](https://huggingface.co/datasets/nebius/SWE-agent-trajectories):
  card reports 80036 trajectories. Additional SWE-Gym/R2E-Gym/SWE-smith releases
  are candidates, not independent additive counts without overlap checks.
* [AgentSuite tau2 trajectories](https://huggingface.co/datasets/AgentSuite/tau2-bench-trajectories):
  revision `07264f1d253a618fc20966670eddc44246212916`. Card reports 30 models ×
  278 tasks; not independently row-counted. Read first gpt-4.1-mini row: 27
  messages with system/user/assistant/tool roles and detailed telecom policy.
  eval_result score=0/db_match=false is not an unsafe label. User is simulated;
  dataset card has no license declaration, so training admission blocked pending
  terms review. Potentially valuable policy/authorization and user-update contexts.
* [BrowserGym execution traces](https://huggingface.co/datasets/agentlabtraces/agentlabtraces):
  officially linked by ServiceNow BrowserGym; candidate for DOM/screenshot and
  workplace/browser sequences. Full payload/schema/license not inspected yet.
* [WebArena-Infinity trajectories](https://huggingface.co/datasets/webarena-x/webarena-infinity-trajectories):
  card reports 2329 successful runs over 13 generated web applications. Actual
  browser interaction but generated environments, not production use; vision/DOM
  inputs must be retained rather than infer click safety from action text alone.

## Monitoring corpora: protect from training contamination

[ScaleAI MRT](https://huggingface.co/datasets/ScaleAI/mrt), revision
`bc636d0c3b881c7d8c76fd71b45b009c3710d437`, CC BY-NC 4.0: public artifact
contains 53296 files (NOT trajectory count). Sample transcript JSON inspected:
conversation, monitoring_response, main_task_evaluation, side_task_evaluation.
17 SHADE task families and 15 CUA-SHADE workplace tasks per card. Real agent
rollouts in mock environments; machine-monitor judgments are not human gold.
SHADE/MRT includes an explicit canary intended to prevent inclusion in training:
keep evaluation-only unless an appropriate training release/permission is established.
Hidden malicious objectives and monitor responses must never enter student inputs.

[SHADE-Arena](https://github.com/safety-research/shade-arena) has a public subset
and restricted full artifacts. Do not request private access or contact authors
without user approval. [WASP](https://github.com/facebookresearch/wasp) provides
an executable benchmark, but inspected tree did not expose a ready corpus of
run-log JSONs: environment availability does not establish released traces.

## Recommended order

1. Audit/adapt already acquired OpenAgentSafety and AgentDojo logs.
2. Acquire pinned AgentDyn logs and inspect RAS-Eval causal histories/terms.
3. Use tau2 and a limited SWE/browser sample for diversity after license review,
   with independent action annotation rather than outcome-label conversion.
4. Keep SHADE/MRT as separate monitoring evaluation; verify AgentLAB per track.

No candidate in this note is automatically added to the active strong_bulk run.
