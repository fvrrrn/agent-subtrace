"""Run fixed ATBench replay heads in isolated processes on GPU2/3."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

from prepare_whole_dialogs import write


def main(root):
    script = Path(__file__).with_name('evaluate_curve_event_atbench.py')
    children = []
    for gpu, arm in zip((2, 3), ('curve_null', 'curve_linear_softmax')):
        command = [sys.executable, '-u', str(script), 'score', '--root', str(root), '--arm', arm]
        with (root / (arm+'.log')).open('w') as stream:
            child = subprocess.Popen(command, env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu)),
                                     stdout=stream, stderr=subprocess.STDOUT)
        children.append(child)
        write(root / (arm+'_launch.json'), {'pid': child.pid, 'gpu': gpu, 'command': command})
    codes = [child.wait() for child in children]
    write(root / 'score_status.json', {'returncodes': codes})
    if any(codes):
        write(root / 'status.json', {'status': 'failed', 'returncodes': codes})
        raise SystemExit(1)
    subprocess.run([sys.executable, '-u', str(script), 'analyze', '--root', str(root)], check=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    main(parser.parse_args().root)
