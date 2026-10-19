"""Run the fixed ATBench replay followed by the restricted Nemotron final-report evaluation."""
from pathlib import Path
import argparse
import subprocess
import sys

import train_action_models as training


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--suffix',default='')
    a=p.parse_args()
    runs=Path('/var/local/danefedov/experiments/agent-safety/runs')
    launch=runs/('external-replay'+a.suffix+'-v1-launch')
    evaluator=Path(__file__).with_name('evaluate_trajectory_actions_run.py')
    for name in ('atbench','nemotron'):
        root=runs/('action-models-'+name+a.suffix+'-v1')
        command=[sys.executable,'-u',str(evaluator),'run','--root',str(root)]
        with (launch/(name+'_evaluation.log')).open('w') as stream:
            p=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT)
            training.write(launch/(name+'_launch.json'),{'pid':p.pid,'command':command})
            code=p.wait()
        training.write(launch/(name+'_status.json'),{'returncode':code})
        print(name,'returncode',code,flush=True)
        if code:
            raise SystemExit(code)
