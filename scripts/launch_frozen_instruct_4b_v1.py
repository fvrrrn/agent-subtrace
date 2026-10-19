"""Sequential GPU3 pipeline with hard deadline, no automatic retries."""
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

from prepare_frozen_instruct_4b_v1 import ROOT, RUN, DEADLINE
from prepare_whole_dialogs import write
from prepare_autoresearch_prompt import file_hash, read


def run_stage(name, script, *arguments):
    remaining = DEADLINE - time.time()
    if remaining <= 0:
        raise TimeoutError('Original training deadline reached')
    # Each stage starts only when no other CUDA process occupies this GPU.
    users = subprocess.check_output(['nvidia-smi', '-i', '3', '--query-compute-apps=pid',
                                     '--format=csv,noheader,nounits'], text=True).strip()
    if users:
        raise RuntimeError(f'GPU3 is occupied; refusing launch: {users}')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='3', OMP_NUM_THREADS='4',
               CUBLAS_WORKSPACE_CONFIG=':4096:8', TOKENIZERS_PARALLELISM='false')
    source = Path(__file__).parent
    overlay = ROOT / 'archive/legacy-shm/danefedov-qwen-mamba-v1/deps'
    env['PYTHONPATH'] = os.pathsep.join((str(source), str(source / 'src'), str(overlay)))
    command = [sys.executable, str(source / script), *map(str, arguments)]
    with (RUN / (name + '.log')).open('x') as log:
        child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True)
        write(RUN / 'status.json', {'status': 'running', 'stage': name,
              'launcher_pid': os.getpid(), 'child_pid': child.pid, 'gpu': 3, 'command': command,
              'deadline_unix': DEADLINE})
        print({'stage': name, 'pid': child.pid}, flush=True)
        try:
            code = child.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
            raise TimeoutError(f'Deadline interrupted own stage {name}')
        if code:
            raise RuntimeError(f'{name} exited with {code}; no retry')


def main():
    assert read(RUN / 'data/data_audit.json')['launch_allowed']
    source = Path(__file__).parent
    manifest = {str(p.relative_to(source)): file_hash(p) for p in source.rglob('*')
                if p.is_file() and '__pycache__' not in p.parts}
    write(source.parent / 'source_manifest.json', manifest)
    write(RUN / 'authorization.json', {'conditions': ['A_broad', 'A_linear', 'A_step', 'B', 'C'],
          'epochs': 2, 'frozen': True, 'lora': False, 'added_attention': False,
          'splits': {'train_small': 2009, 'dev_select': 301}, 'deadline_unix': DEADLINE,
          'global_deadline_unix': 1792348785, 'threshold': .5, 'selection': 'AP; tie lower BCE',
          'source_manifest_sha256': file_hash(source.parent / 'source_manifest.json')})
    run_stage('cache', 'cache_frozen_instruct_4b_v1.py')
    for arm in ('A_broad', 'A_linear', 'A_step', 'B', 'C'):
        common = ['--root', RUN / arm, '--data', RUN / 'data', '--cache', RUN / 'cache',
                  '--splits', RUN / 'data/splits.json', '--arm', arm,
                  '--deadline-unix', DEADLINE,
                  '--control-gate', ROOT / 'runs/sound-anomaly-4b-control-v1/candidates/control_a/gate.json']
        for phase in ('gate', 'train'):
            run_stage(arm + '_' + phase, 'train_frozen_instruct_4b_v1.py', phase, *common)
        assert read(RUN / arm / 'fit/complete.json')['epochs'] == 2
    write(RUN / 'status.json', {'status': 'complete', 'conditions': 5, 'epochs_each': 2})


if __name__ == '__main__':
    try:
        main()
    except Exception:
        write(RUN / f'failure_{time.time_ns()}.json', {'status': 'failed', 'traceback': traceback.format_exc()})
        write(RUN / 'status.json', {'status': 'failed', 'traceback': traceback.format_exc()})
        raise
