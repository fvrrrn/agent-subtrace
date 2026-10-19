"""One finite conditional round; incomplete parent results are a hard stop."""
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import select
import subprocess
import sys
import time

from tqdm import tqdm
import train_weighted_broad_lora_v1 as parent
from prepare_whole_dialogs import write

ROOT = Path(os.environ.get('FOLLOWUP_ROOT', str(parent.ROOT / 'runs/streaming-followup-v1')))
PREVIOUS = Path(os.environ.get('FOLLOWUP_PARENT', str(parent.ROOT / 'runs/weighted-broad-lora-ablation-v3')))
SOURCE = Path(__file__).parent


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return parent.trainer.original.file_hash(path)


def completed(arm):
    status = read(arm/'status.json')
    assert status['status'] == 'complete' and status['epochs'] == 2, (str(arm), status)
    assert status['frozen_weights_unchanged'] and read(arm/'frozen_audit.json')['unchanged']
    assert read(arm/'warmstart_gate.json')['raw_error'] == 0
    gate = read(arm/'gate.json')
    assert gate['status'] == 'passed' and gate['loss_support_passed']
    assert gate['causal_suffix_error'] == gate['reload_error'] == 0
    for epoch in (1, 2):
        assert read(arm/f'epoch{epoch}/status.json')['status'] == 'complete'
    selection = read(arm/'selection.json')
    epoch = selection['epoch']
    assert epoch in (1, 2)
    assert sha(arm/f'epoch{epoch}/head.pt') == selection['head_sha256']
    for name, expected in selection['adapter_sha256'].items():
        assert sha(arm/f'epoch{epoch}/adapter'/name) == expected
    metrics = read(arm/f'dev_epoch{epoch}/metrics.json')
    assert metrics == selection['metrics']
    assert all(math.isfinite(metrics[k]) for k in ('average_precision', 'event_bce'))
    history = read(arm/'history.json')
    assert len(history) == 2
    best = max(history, key=lambda h: (h['validation']['average_precision'], -h['validation']['event_bce'], -h['epoch']))
    assert best['epoch'] == epoch
    events = [json.loads(s) for s in (arm/f'dev_epoch{epoch}/predictions.jsonl').read_text().splitlines()]
    identities = sorted((e['key'], e['dialog'], e['label'], e['ids_sha256'], e['state_sha256']) for e in events)
    assert len(identities) == len(set(identities)) == 886
    return selection, status, events, identities


def main():
    ROOT.mkdir(parents=True, exist_ok=False)
    write(ROOT/'authorization.json', {'user_request': 'Продолжай эксперименты пока не будет какой-то нормальный результат',
        'max_new_conditions': 3, 'epochs_each': 2, 'requires_all_parent_cells': True,
        'parent_launcher_pid': 4134159, 'scheduler_pid': os.getpid(), 'created_unix': time.time(),
        'protocol_sha256': sha(SOURCE/'FOLLOWUP_PROTOCOL.md')})
    write(ROOT/'source_manifest.json', {str(p.relative_to(SOURCE)): sha(p) for p in SOURCE.rglob('*')
        if p.is_file() and '__pycache__' not in p.parts})
    try:
        try:
            fd = os.pidfd_open(4134159)
        except ProcessLookupError:
            print('Parent launcher absent; validating all eight terminal results.', flush=True)
        else:
            try:
                command = Path('/proc/4134159/cmdline').read_bytes()
                assert b'launch_weighted_broad_lora_v1.py' in command, 'PID identity mismatch'
                write(ROOT/'status.json', {'status': 'waiting_pidfd', 'parent_pid': 4134159})
                print('Waiting via pidfd for original eight-cell launcher; no polling.', flush=True)
                poll = select.poll(); poll.register(fd, select.POLLIN); poll.poll()
            finally:
                os.close(fd)
        expected = list(itertools.product(('lora', 'frozen'), ('broad', 'step'), ('adaptive', 'uniform')))
        assert read(PREVIOUS/'authorization.json')['cells'] == [list(c) for c in expected]
        summaries = {}; reference = None; hashes = {}
        for config in tqdm(expected, desc='Validate completed parent cells', unit='cell'):
            name = '_'.join(config)
            selection, status, events, identities = completed(PREVIOUS/name)
            if reference is None:
                reference = identities
            assert identities == reference
            summaries[name] = {'configuration': dict(zip(('adaptation','curve','weighting'), config)),
                'selection': selection, 'seconds': status['seconds']}
            for filename in ('selection.json', 'protocol.json', 'status.json', 'history.json', 'gate.json', 'frozen_audit.json'):
                hashes[f'{name}/{filename}'] = sha(PREVIOUS/name/filename)
        assert read(PREVIOUS/'status.json')['status'] == 'complete'
        winner = min(summaries, key=lambda n: (-summaries[n]['selection']['metrics']['average_precision'],
            summaries[n]['selection']['metrics']['event_bce'], summaries[n]['seconds'], n))
        parent.trainer.original.BASE = parent.BASE
        parent.trainer.original.REVISION = parent.REVISION
        rows, audit = parent.trainer.original.load_data(parent.DATA)
        splits = read(parent.DATA/'splits.json')
        assert len(splits['train_small']) == 2009 and len(splits['dev_select']) == 301
        index = {r['key']: r for r in rows}
        train = [index[k] for k in splits['train_small']]
        unsafe = math.fsum(sum(s['label'] for s in r['spans'])/len(r['spans']) for r in train)
        safe = len(train)-unsafe
        assert unsafe > 0 and safe > 0
        ratio = safe/unsafe
        assert ratio == 7.733163050192017, 'Prospective train class mass changed'
        cells = [{'name': name, 'positive_weight': weight, 'event_weight': coefficient}
            for name, weight, coefficient in [('off_event1', 1., 1.), ('on_event01', ratio, .1), ('on_event1', ratio, 1.)]]
        lock = {'winner': winner, 'winner_configuration': summaries[winner]['configuration'],
            'parent_results': summaries, 'parent_hashes': hashes, 'cells': cells, 'locked_unix': time.time(),
            'effective_safe': safe, 'effective_unsafe': unsafe, 'positive_weight': ratio,
            'train_ids': splits['train_small'], 'dev_ids': splits['dev_select'],
            'data_audit': audit, 'splits_sha256': sha(parent.DATA/'splits.json'),
            'warmstart_sha256': parent.HEAD_SHA, 'kernel_manifest_sha256': sha(SOURCE/'WEIGHTED_LORA_KERNELS.json'),
            'action_identity_sha256': hashlib.sha256(json.dumps(reference).encode()).hexdigest()}
        write(ROOT/'lock.json', lock)
        for cell in tqdm(cells, desc='Locked three-cell follow-up', unit='cell'):
            # Sequential GPU3 reuse only after the parent has fully completed.
            users = subprocess.check_output(['nvidia-smi','-i','3','--query-compute-apps=pid',
                '--format=csv,noheader,nounits'], text=True).strip()
            assert not users, f'GPU3 occupied: {users}'
            command = [sys.executable, '-u', str(SOURCE/'train_followup.py'), '--lock', str(ROOT/'lock.json'),
                '--cell', cell['name'], '--root', str(ROOT/cell['name'])]
            with (ROOT/(cell['name']+'.log')).open('xb') as log:
                child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
                write(ROOT/'status.json', {'status': 'running', 'cell': cell['name'], 'child_pid': child.pid})
                print(f'LAUNCHED {cell["name"]} PID={child.pid} LOG={log.name}', flush=True)
                assert child.wait() == 0, 'Child failed; no automatic retry'
            _, _, _, identities = completed(ROOT/cell['name'])
            assert identities == reference
        report = {}
        import numpy as np
        for name, arm in [(winner, PREVIOUS/winner)] + [(c['name'], ROOT/c['name']) for c in cells]:
            selection, status, events, identities = completed(arm)
            slices = {}
            for source in sorted({s for e in events for s in e['sources']}):
                group = [e for e in events if source in e['sources']]
                metrics = parent.trainer.evaluate([e['label'] for e in group], [e['score'] for e in group])
                metrics.pop('pr_curve', None)
                slices[source] = {'n': len(group), **metrics}
            report[name] = {'selection': selection, 'seconds': status['seconds'], 'source_slices': slices,
                'class_score_quantiles': {str(label): np.quantile([e['score'] for e in events if e['label']==label],
                    [0,.1,.25,.5,.75,.9,1]).tolist() for label in (0,1)}}
        write(ROOT/'comparison.json', {'paired_actions': 886, 'results': report,
            'interpretation': 'Dev-selected continuation factorial, one seed; no held-out claim or threshold fit.'})
        write(ROOT/'status.json', {'status': 'complete', 'new_cells': 3, 'epochs_each': 2})
    except Exception as error:
        write(ROOT/'status.json', {'status': 'blocked_or_failed', 'error': repr(error), 'automatic_retry': False})
        raise


if __name__ == '__main__':
    main()
