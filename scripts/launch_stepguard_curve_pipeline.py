"""Run external evaluation, four-epoch curves, then their fixed-checkpoint evaluation."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def run(root, parent, hard, previous, curves):
    scripts = Path(__file__).parent
    evaluator = scripts / 'evaluate_all_action_stepguard.py'
    launcher = scripts / 'launch_hard_target_models.py'
    logs = root / 'pipeline_logs'; logs.mkdir(exist_ok=False)
    initial = [sys.executable, '-u', str(evaluator), 'initial', '--root', str(root),
               '--parent', str(parent), '--hard', str(hard), '--previous', str(previous), '--curves', str(curves)]
    stages = [('initial_stepguard', initial),
              ('four_epoch_curves', [sys.executable, '-u', str(launcher), 'run', '--root', str(curves)]),
              ('curve_stepguard', [sys.executable, '-u', str(evaluator), 'curves', '--root', str(root)])]
    write(root / 'pipeline_manifest.json', {'pid': os.getpid(), 'started_unix': time.time(),
        'stages': stages, 'gpu': 2, 'curve_epochs': 4,
        'memory_policy': 'Finish LoRA online evaluation before starting concurrent frozen-head jobs.'})
    for name, command in stages:
        print('START ' + name, flush=True)
        with (logs / (name + '.log')).open('w') as output:
            p = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT)
            write(logs / (name + '_launch.json'), {'pid': p.pid, 'command': command})
            code = p.wait()
        write(logs / (name + '_status.json'), {'returncode': code, 'finished_unix': time.time()})
        print(f'END {name} returncode={code}', flush=True)
        if code:
            write(root / 'pipeline_status.json', {'status': 'failed', 'stage': name})
            raise SystemExit(code)
        if name == 'initial_stepguard':
            command = [sys.executable, '-u', str(evaluator), 'trained_kev', '--root', str(root), '--parent', str(parent)]
            with (root / 'trained_kev.log').open('w') as output:
                kev = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
            write(root / 'trained_kev_launch.json', {'pid': kev.pid, 'command': command,
                  'note': 'Waits for existing native Kev training process; does not train or poll.'})
        if name in ('initial_stepguard', 'curve_stepguard'):
            snapshot = root / 'comparisons' / ('initial.json' if name == 'initial_stepguard' else 'with_curves.json')
            print(snapshot.read_text(), flush=True)
    write(root / 'pipeline_status.json', {'status': 'curves_and_ready_models_complete',
          'native_kev': 'independent completion-gated evaluator', 'finished_unix': time.time()})


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'parent', 'hard', 'previous', 'curves'):
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    run(a.root, a.parent, a.hard, a.previous, a.curves)
