---
license: apache-2.0
language:
- en
pretty_name: StepGuard-Data
size_categories:
- 1K<n<10K
tags:
- agent-safety
- tool-use
- step-level-safety
- supervised-fine-tuning
- reinforcement-learning
configs:
- config_name: sft3k
  data_files:
  - split: train
    path: sft3k/agentguard_sft3k_sharegpt.jsonl
- config_name: rl4k
  data_files:
  - split: train
    path: rl4k/agentguard_rl4k_grpo.jsonl
---

# StepGuard-Data

**7K training examples for agent safety · 3K SFT + 4K RL · EMNLP 2026**

[Paper](https://arxiv.org/abs/2608.24777) · [Model](https://huggingface.co/ninty-seven/StepGuard) · [Code](https://github.com/zheng977/StepGuard) · [Project](https://zheng977.github.io/StepGuard/)

This dataset releases the final **7,000-example training corpus** used for StepGuard, a 4B guard that checks proposed tool actions before execution and audits completed agent trajectories. StepGuard is fine-tuned from **Qwen3-4B-Instruct-2507** using **StepGen** supervision and **Balance-GRPO**.

The corpus contains two training stages with separate schemas: **SFT-3K** provides prompts paired with structured safety responses; **RL-4K** provides prompts and safety labels for reinforcement learning. Both cover action-level guarding and trajectory-level auditing.

## Dataset at a glance

| Subset | Training stage | Examples | Action-level | Trajectory-level | Safe | Unsafe |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| **SFT-3K** | Supervised fine-tuning | 3,000 | 1,500 | 1,500 | 1,500 | 1,500 |
| **RL-4K** | Balance-GRPO | 4,000 | 2,000 | 2,000 | 2,000 | 2,000 |
| **Total** | SFT → RL | **7,000** | **3,500** | **3,500** | **3,500** | **3,500** |

A safe decision can still occur in the presence of a risk source. SFT-3K includes 300 benign (`risk_source=none`) examples and 1,200 safe examples with a non-`none` risk source; RL-4K includes 600 and 1,400, respectively. Safety judgment and risk-source attribution are separate targets.

These are training subsets, each exposed as a `train` split. For the evaluation benchmarks and results, see the [paper](https://arxiv.org/abs/2608.24777) and [model card](https://huggingface.co/ninty-seven/StepGuard).

## Files

| File | Contents |
| --- | --- |
| [`sft3k/agentguard_sft3k_sharegpt.jsonl`](sft3k/agentguard_sft3k_sharegpt.jsonl) | 3,000 ShareGPT-format SFT examples |
| [`rl4k/agentguard_rl4k_grpo.jsonl`](rl4k/agentguard_rl4k_grpo.jsonl) | 4,000 RL prompts with safety labels |
| [`manifests/sft3k_manifest.json`](manifests/sft3k_manifest.json) | SFT sampling targets and composition statistics |
| [`manifests/rl4k_manifest.json`](manifests/rl4k_manifest.json) | RL sampling targets, source-pool counts, and quality-check summaries |

The two JSONL files are kept separate because they serve different training stages and use different schemas. Manifests describe the sampling process; they are not training examples.

## Quick start

The repository is public and does not require an access request.

```bash
pip install datasets
```

Load the two subsets independently:

```python
from datasets import load_dataset

repo_id = "ninty-seven/stepguard_data"
sft = load_dataset(repo_id, "sft3k", split="train")
rl = load_dataset(repo_id, "rl4k", split="train")

assert len(sft) == 3000
assert len(rl) == 4000

sft_prompt = sft[0]["conversations"][0]["value"]
sft_response = sft[0]["conversations"][1]["value"]
rl_prompt = rl[0]["instruction"]
rl_label = rl[0]["label"]
```

Use the released [training recipe](https://github.com/zheng977/StepGuard/tree/main/training/release) for SFT and Balance-GRPO. The [prompt templates](https://github.com/zheng977/StepGuard/blob/main/src/guardrail/prompts/stepguard.py) and [input-format guide](https://github.com/zheng977/StepGuard/blob/main/docs-open/prompts.md) describe how requests, tools, actions, and observations are serialized.

## Data formats

### SFT-3K

Each row contains one `conversations` array with a `human` prompt followed by a `gpt` reference response. The following is a schema illustration; placeholders are not an actual training example.

```json
{
  "conversations": [
    {"from": "human", "value": "<guard prompt and agent context>"},
    {"from": "gpt", "value": "<structured safety response>"}
  ]
}
```

Reference responses contain:

```text
<Analysis>Safety rationale</Analysis>
<Judgment>safe or unsafe</Judgment>
<RiskSourcePresent>yes or no</RiskSourcePresent>
<RiskSource>risk-source label</RiskSource>
```

Trajectory-level responses additionally contain `<UnsafeStep>None or an integer step ID</UnsafeStep>`. Step IDs refer to agent actions. See the model card for the full output contract and inference prompts.

### RL-4K

| Field | Type | Meaning |
| --- | --- | --- |
| `id` | string | Unique sample identifier: `stepguard-rl-000001` through `stepguard-rl-004000` |
| `instruction` | string | Guard prompt with the action or trajectory context |
| `output` | string | Reference safety label: `safe` or `unsafe` |
| `label` | string | Safety label; identical to `output` in this release |
| `level` | string | `action` or `trajectory` |
| `risk_source` | string | Risk-source category |

**RL `output` is a class label, not a full rationale or a structured SFT response.** Use this file with the RL training recipe rather than treating its labels as the SFT response format. The prompt is stored in `instruction`; safety labels are stored in `output` and `label`.

## Risk-source categories

Both subsets use the following nine categories, including `none` for benign context:

```text
none
malicious_user_instruction_or_jailbreak
direct_prompt_injection
indirect_prompt_injection
unreliable_or_misinformation
tool_description_injection
malicious_tool_execution
corrupted_tool_feedback
inherent_agent_failure
```

Per-category counts and sampling details are provided in the manifests. Category counts are not uniform, even though each subset has equal numbers of safe and unsafe examples.

## License

The StepGuard-Data release is licensed under the [Apache License 2.0](LICENSE) for the rights held by the StepGuard contributors in the training examples, annotations, compilation, and accompanying documentation. Commercial use, training, modification, and redistribution are permitted under its terms; we impose no research-only or non-commercial restriction on these contributions.

The corpus was constructed using StepGen, as described in the [paper](https://arxiv.org/abs/2608.24777). Referenced third-party tools, services, and any incorporated third-party material remain subject to their respective rights and applicable terms; this license does not relicense third-party material or grant access to external services. Evaluation benchmarks linked from the code repository are separate releases and retain their own licenses.

## Citation

If you use this dataset or StepGuard, please cite our paper:

```bibtex
@misc{zheng2026stepguardlearningsteplevelguardrails,
  title         = {StepGuard: Learning Step-Level Guardrails with Scalable Supervision and Safety-Utility Balancing},
  author        = {Zhijie Zheng and Yu Li and Chen Qian and Yuqian Fu and Yanwei Fu and Lu Sheng and Jing Shao and Dongrui Liu},
  year          = {2026},
  eprint        = {2608.24777},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI},
  url           = {https://arxiv.org/abs/2608.24777}
}
```
