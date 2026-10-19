"""Locked epoch-one Kev on official TS suites against fixed existing LoRA outputs."""
import argparse
import os
from pathlib import Path
import select

import train_action_models as training
import evaluate_action_tsharm as benchmark


def main(a):
    runs = a.root.parent
    snapshot = runs / 'kev-noul-epoch1-snapshot-v1'
    audit = training.read(snapshot / 'snapshot_audit.json')
    assert audit['selected_epoch'] == 1 and audit['status'] == 'passed'
    for filename, expected in audit['checkpoint_sha256'].items():
        assert training.sha(snapshot / 'checkpoint' / filename) == expected
    a.root.mkdir(parents=True, exist_ok=False)
    selection = training.read(snapshot / 'selection.json')
    assert selection['epoch'] == 1
    for suite in ('dojo', 'harm'):
        reference = runs / ('action-models-ts' + suite + '-v1')
        root = a.root / suite
        root.mkdir()
        (root / 'records.json').symlink_to(reference / 'records.json')
        cfg = training.read(reference / 'protocol.json')
        cfg.update(source_sha256=training.sha(training.__file__),
                   records_sha256=training.sha(root / 'records.json'),
                   comparison='Locked Kev NOUL+ranking epoch1 vs fixed Mamba+LoRA epochs1/2 and original Kev',
                   reference_root=str(reference), checkpoint_snapshot_audit=audit,
                   threshold=.5, no_benchmark_selection=True)
        training.write(root / 'protocol.json', cfg)
        locked = root / 'locked/kev_epoch1'
        locked.mkdir(parents=True)
        (locked / 'checkpoint').symlink_to(snapshot / 'checkpoint', target_is_directory=True)
        item = {'name': 'kev_epoch1', 'kind': 'kev', 'selected': selection,
                'source': str(snapshot), 'checkpoint_sha256': audit['checkpoint_sha256']}
        training.write(root / 'model_manifest.json', item)
        for name in ('lora_epoch1', 'lora_validation_epoch2', 'kev_original'):
            source = reference / 'predictions' / name
            assert (source / 'complete.json').exists()
            (root / 'predictions').mkdir(exist_ok=True)
            (root / 'predictions' / name).symlink_to(source, target_is_directory=True)
    if a.wait_pid:
        try:
            fd = os.pidfd_open(a.wait_pid)
        except ProcessLookupError:
            fd = None
        if fd is not None:
            print('Waiting on GPU3 training PID', a.wait_pid, flush=True)
            select.select([fd], [], [])
            os.close(fd)
    for suite in ('dojo', 'harm'):
        root = a.root / suite
        benchmark.score(root, training.read(root / 'model_manifest.json'))
        benchmark.compare(root, 'epoch1')
        training.write(root / 'status.json', {'status': 'complete'})
    training.write(a.root / 'status.json', {'status': 'complete', 'trained_epoch': 1,
        'threshold': .5, 'training_unchanged': True, 'no_reference_inference_repeated': True})


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--wait-pid', type=int)
    main(p.parse_args())
