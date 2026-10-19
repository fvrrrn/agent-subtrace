"""Four linear/step conditions; select on validation, then a separate attention-before-Mamba run."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from prepare_whole_dialogs import write


def read(path):
    return json.loads(path.read_text())


def attention(args):
    plan = read(args.root / 'protocol.json')
    arm, curve = plan['arm'], plan['curve']
    script = Path(__file__).with_name('train_curve_event_heads.py')
    for phase in ('gate', 'train'):
        cmd = [sys.executable, '-u', str(script), phase, '--root', str(args.root),
               '--data', str(args.data), '--arm', arm, '--curve', curve, '--epochs', '4', '--attention']
        with (args.root / (phase+'.log')).open('w') as out:
            child = subprocess.Popen(cmd, env=dict(os.environ, CUDA_VISIBLE_DEVICES='2'),
                                     stdout=out, stderr=subprocess.STDOUT)
        write(args.root / (phase+'_launch.json'), {'pid': child.pid, 'gpu': 2, 'command': cmd})
        code = child.wait()
        if code:
            write(args.root / 'status.json', {'status': 'failed', 'phase': phase, 'returncode': code})
            raise SystemExit(code)
    f = args.root / 'arms' / arm
    choice = read(f / 'selection.json')
    history = read(f / 'history.json')
    base = Path(plan['selected_baseline_run']) / 'arms' / arm
    base_choice = read(base / 'selection.json')
    base_history = read(base / 'history.json')
    write(args.root / 'comparison.json', {'selection': choice, 'baseline_selection': base_choice,
        'attention_validation': next(h['validation'] for h in history if h['epoch'] == choice['epoch']),
        'baseline_validation': next(h['validation'] for h in base_history if h['epoch'] == base_choice['epoch']),
        'limitations': 'Validation-selected curve/readout then validation-selected checkpoint; one seed, added capacity. Not independent test. No ATBench selection/evaluation.'})
    write(args.root / 'status.json', {'status': 'complete', 'epochs': 4, 'attention_before_mamba': True})


def shapes(args):
    launcher = Path(__file__).with_name('launch_curve_event_heads.py')
    for curve in ('linear', 'step'):
        folder = args.root / curve
        cmd = [sys.executable, '-u', str(launcher), '--root', str(folder), '--data', str(args.data),
               '--epochs', '4', '--curve', curve]
        with (folder / 'scheduler.log').open('w') as out:
            child = subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT)
        write(folder / 'launch.json', {'pid': child.pid, 'command': cmd})
        code = child.wait()
        if code:
            write(args.root / 'status.json', {'status': 'failed', 'curve': curve, 'returncode': code})
            raise SystemExit(code)
    choices = []
    for curve in ('linear', 'step'):
        for arm in ('curve_null', 'curve_linear_softmax'):
            choice = read(args.root / curve / 'arms' / arm / 'selection.json')
            choices.append({'curve': curve, 'arm': arm, **choice})
    best = min(choices, key=lambda c: (c['validation_event_bce'], c['curve'], c['arm']))
    write(args.root / 'selection.json', {'candidates': choices, 'selected': best,
        'criterion': 'minimum hard event validation BCE; no benchmark labels'})
    dest = args.attention_root
    p = read(dest / 'protocol.json')
    p.update(arm=best['arm'], curve=best['curve'], selection_pending=False,
             selected_baseline_run=str(args.root / best['curve']),
             selected_baseline_checkpoint=best, selection_source=str(args.root / 'selection.json'))
    write(dest / 'protocol.json', p)
    attention_launcher = Path(p['source_root']) / 'scripts' / Path(__file__).name
    cmd = [sys.executable, '-u', str(attention_launcher), 'attention', '--root', str(dest), '--data', str(args.data)]
    with (dest / 'scheduler.log').open('w') as out:
        child = subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
    write(dest / 'launch.json', {'pid': child.pid, 'command': cmd, 'gpu': 2})
    write(args.root / 'status.json', {'status': 'complete', 'conditions': 4,
        'attention_separate_pid': child.pid, 'attention_run': str(dest)})
    # Wait on the separate experiment, without polling, so the notification also copies its final artifacts.
    code = child.wait()
    write(args.root / 'attention_wait.json', {'returncode': code})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('shapes', 'attention'))
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--attention-root', type=Path)
    args = parser.parse_args()
    {'shapes': shapes, 'attention': attention}[args.phase](args)
