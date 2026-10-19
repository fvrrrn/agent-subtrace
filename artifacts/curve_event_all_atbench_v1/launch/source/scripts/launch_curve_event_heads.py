"""Audit and train two curve-plus-event conditions concurrently on GPU2/3."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from prepare_whole_dialogs import write
from graph_safety.metrics import evaluate

ARMS = ('curve_null', 'curve_linear_softmax')


def main(args):
    source = Path(__file__).with_name('train_curve_event_heads.py')
    for phase in ('gate', 'train'):
        children = []
        for gpu, arm in zip((2, 3), ARMS):
            command = [sys.executable, '-u', str(source), phase, '--root', str(args.root),
                       '--data', str(args.data), '--arm', arm, '--epochs', str(args.epochs), '--curve', args.curve]
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
            with (args.root / f'{phase}_{arm}.log').open('w') as stream:
                child = subprocess.Popen(command, env=env, stdout=stream, stderr=subprocess.STDOUT)
            children.append(child)
            write(args.root / f'{phase}_{arm}_launch.json', {'pid': child.pid, 'gpu': gpu, 'command': command})
            print('START', phase, arm, child.pid, flush=True)
        codes = [p.wait() for p in children]
        write(args.root / f'{phase}_status.json', {'returncodes': codes})
        if any(codes):
            write(args.root / 'status.json', {'status': 'failed', 'phase': phase, 'returncodes': codes})
            raise SystemExit(1)
        if phase == 'gate':
            gates = [json.loads((args.root / (arm+'_gate.json')).read_text()) for arm in ARMS]
            assert all(g['status'] == 'passed' for g in gates)
            assert len({g['shared_head_sha256'] for g in gates}) == 1
        print('END', phase, codes, flush=True)
    outputs, selections = {}, {}
    for arm in ARMS:
        folder = args.root / 'arms' / arm
        choice = json.loads((folder / 'selection.json').read_text())
        selections[arm] = choice
        outputs[arm] = {r['key']: r for r in map(json.loads, (folder / f'validation_epoch{choice["epoch"]}.jsonl').read_text().splitlines())}
    common = sorted(set.intersection(*(set(v) for v in outputs.values())))
    assert all(len(v) == len(common) for v in outputs.values())
    for key in common:
        assert len({(v[key]['label'], v[key]['state_sha256'], v[key]['ids_sha256']) for v in outputs.values()}) == 1
    metrics = {}
    for arm, output in outputs.items():
        m = evaluate([output[k]['label'] for k in common], [output[k]['score'] for k in common])
        m.pop('pr_curve')
        metrics[arm] = m
    write(args.root / 'comparison.json', {'common_validation_actions': len(common), 'selections': selections,
        'metrics': metrics, 'limitations': 'Validation-selected one-seed comparison. Token curve is artificial. Both conditions use token-curve BCE PLUS0.1event BCE. No external benchmark run.'})
    write(args.root / 'status.json', {'status': 'complete', 'arms': list(ARMS), 'epochs_each': args.epochs})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--epochs', type=int, choices=range(1, 101), default=2)
    parser.add_argument('--curve', choices=('broad', 'linear', 'step'), default='broad')
    main(parser.parse_args())
