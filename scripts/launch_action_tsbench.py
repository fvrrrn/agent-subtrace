"""Evaluate ready Harm models then the fixed Dojo shortlist, on one GPU queue."""
from pathlib import Path
import subprocess
import sys

import train_action_models as training


if __name__ == '__main__':
    runs = Path('/var/local/danefedov/experiments/agent-safety/runs')
    launch = runs / 'action-models-tsbench-v1-launch'
    evaluator = Path(__file__).with_name('evaluate_action_tsharm.py')
    for benchmark in ('harm', 'dojo'):
        command = [sys.executable, '-u', str(evaluator), 'ready', '--benchmark', benchmark,
                   '--root', str(runs / ('action-models-ts' + benchmark + '-v1'))]
        with (launch / (benchmark + '.log')).open('w') as output:
            process = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT)
            training.write(launch / (benchmark + '_launch.json'), {'pid': process.pid, 'command': command})
            code = process.wait()
        training.write(launch / (benchmark + '_status.json'), {'returncode': code})
        print(f'{benchmark} finished with returncode {code}', flush=True)
        if code:
            raise SystemExit(code)
