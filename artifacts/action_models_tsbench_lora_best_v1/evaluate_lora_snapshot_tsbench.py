"""Evaluate the explicitly locked validation-best interim LoRA on both TS suites."""
from pathlib import Path

import evaluate_action_tsharm as evaluator
import train_action_models as training


if __name__ == '__main__':
    runs = Path('/var/local/danefedov/experiments/agent-safety/runs')
    snapshot = runs / 'action-models-tsbench-lora-best-v1'
    audit = training.read(snapshot / 'snapshot_audit.json')
    assert audit['selection']['epoch'] == 2
    for suite in ('harm', 'dojo'):
        root = runs / ('action-models-ts' + suite + '-v1')
        locked = root / 'locked/lora_validation_epoch2'
        for name, expected in audit['files_sha256'].items():
            assert training.sha(locked / name) == expected
        evaluator.score(root, training.read(locked / 'locked_manifest.json'))
        evaluator.compare(root, 'lora_validation_epoch2')
    training.write(snapshot / 'status.json', {'status': 'complete', 'epoch': 2,
        'selection': audit['selection_basis'], 'training_unchanged': True})
