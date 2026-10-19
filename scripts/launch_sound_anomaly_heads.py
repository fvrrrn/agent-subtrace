"""Wait for matched A via pidfd, then B and C sequentially on GPU 2."""
import json
import os
from pathlib import Path
import select
import subprocess
import sys
import time
import traceback

ROOT = Path('/var/local/danefedov/experiments/agent-safety')
RUN = ROOT / 'runs/sound-anomaly-4b-v1'
SOURCE = Path(__file__).resolve().parents[1]
DEADLINE = 1792345185.243998


def write(name, value):
    (RUN / name).write_text(json.dumps(value, indent=2) + '\n')


def main():
    RUN.mkdir(parents=True, exist_ok=True)
    write('scheduler.json', {'pid': os.getpid(), 'status': 'waiting_for_control', 'control_pid': 2991678,
                            'deadline': DEADLINE, 'gpu': 2, 'source': str(SOURCE)})
    try:
        fd = os.pidfd_open(2991678)
    except ProcessLookupError:
        fd = None
    if fd is not None:
        try:
            ready, _, _ = select.select([fd], [], [], max(0, DEADLINE - time.time()))
            if not ready:
                raise TimeoutError('Deadline while waiting for A')
        finally:
            os.close(fd)
    control = ROOT / 'runs/sound-anomaly-4b-control-v1/candidates/control_a'
    assert json.loads((control / 'fit/complete.json').read_text())['status'] == 'complete'
    deadline = min(DEADLINE, time.time() + 2700)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='2', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
               HF_HOME=str(ROOT / 'models/huggingface'),
               PYTHONPATH=f'{SOURCE}/src:{ROOT}/archive/legacy-shm/danefedov-qwen-mamba-v1/deps')
    for arm in ('B', 'C'):
        for phase in ('gate', 'train'):
            remaining = deadline - time.time()
            if remaining <= 0:
                raise TimeoutError('Wave budget exhausted')
            command = [sys.executable, '-u', str(SOURCE / 'scripts/train_sound_anomaly_heads.py'), phase,
                       '--arm', arm, '--root', str(RUN / arm), '--data', str(ROOT / 'datasets/whole-dialog-4b-v2'),
                       '--cache', str(ROOT / 'runs/whole-dialog-4b-v3/features'),
                       '--splits', str(ROOT / 'runs/autoresearch-sixhour-v1/data/splits.json'),
                       '--control-gate', str(control / 'gate.json'), '--deadline-unix', str(deadline)]
            with (RUN / f'{arm}_{phase}.log').open('x') as log:
                child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
                write('status.json', {'status': 'running', 'arm': arm, 'phase': phase, 'pid': child.pid,
                                     'scheduler_pid': os.getpid(), 'command': command, 'deadline': deadline})
                try:
                    code = child.wait(timeout=remaining)
                except subprocess.TimeoutExpired:
                    child.terminate()
                    child.wait()
                    raise
                if code:
                    raise RuntimeError(f'{arm} {phase} failed with exit {code}; no retry')
    write('status.json', {'status': 'complete', 'finished_unix': time.time(), 'scheduler_pid': os.getpid()})


if __name__ == '__main__':
    try:
        main()
    except Exception:
        write('failure.json', {'status': 'failed', 'traceback': traceback.format_exc()})
        raise
