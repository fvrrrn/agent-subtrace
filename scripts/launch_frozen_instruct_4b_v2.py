"""User-authorized identity repair restart; frozen v1 cache is read-only."""
import os
import signal
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

from tqdm import tqdm
from prepare_frozen_instruct_4b_v1 import ROOT, DEADLINE
from prepare_whole_dialogs import write
from prepare_autoresearch_prompt import file_hash, read

RUN = ROOT / 'runs/frozen-instruct-4b-v2'
OLD = ROOT / 'runs/frozen-instruct-4b-v1'


def run_stage(name, script, *arguments, gpu=True):
    remaining = DEADLINE - time.time()
    assert remaining > 0, 'Original training deadline reached'
    if gpu:
        users = subprocess.check_output(['nvidia-smi', '-i', '3', '--query-compute-apps=pid',
                                        '--format=csv,noheader,nounits'], text=True).strip()
        assert not users, f'GPU3 occupied; refusing launch: {users}'
    source = Path(__file__).parent
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='3' if gpu else '', OMP_NUM_THREADS='4',
               CUBLAS_WORKSPACE_CONFIG=':4096:8', TOKENIZERS_PARALLELISM='false', PYTHONDONTWRITEBYTECODE='1')
    env['PYTHONPATH'] = os.pathsep.join((str(source), str(source / 'src'),
        str(ROOT / 'archive/legacy-shm/danefedov-qwen-mamba-v1/deps')))
    command = [sys.executable, '-u', str(source / script), *map(str, arguments)]
    with (RUN / (name + '.log')).open('xb') as log:
        child = subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 start_new_session=True)
        def stream():
            while chunk := os.read(child.stdout.fileno(), 8192):
                log.write(chunk)
                log.flush()
                sys.stdout.buffer.write(chunk)
                sys.stdout.buffer.flush()
        reader = threading.Thread(target=stream)
        reader.start()
        write(RUN / 'status.json', {'status': 'running', 'stage': name, 'launcher_pid': os.getpid(),
              'child_pid': child.pid, 'gpu': 3 if gpu else None, 'command': command, 'deadline_unix': DEADLINE})
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
        finally:
            reader.join()
        assert code == 0, f'{name} exited {code}; no retry'


def main():
    source = Path(__file__).parent
    manifest = {str(p.relative_to(source)): file_hash(p) for p in source.rglob('*')
                if p.is_file() and '__pycache__' not in p.parts}
    write(source.parent / 'source_manifest.json', manifest)
    write(RUN / 'authorization.json', {'user_request': 'Fix cache identity and restart intended five Instruct arms',
          'conditions': ['A_broad', 'A_linear', 'A_step', 'B', 'C'], 'epochs': 2,
          'frozen': True, 'lora': False, 'added_attention': False, 'deadline_unix': DEADLINE,
          'cache_read_only': str(OLD / 'cache'), 'selection': 'AP; tie lower BCE', 'threshold': .5,
          'source_manifest_sha256': file_hash(source.parent / 'source_manifest.json')})
    with tqdm(total=11, desc='Instruct v2 stages', unit='stage') as progress:
        run_stage('cpu_cache_admission', 'verify_frozen_instruct_4b_v2.py', '--output', RUN, gpu=False)
        progress.update()
        assert read(RUN / 'cache_admission.json')['count'] == 2310
        for arm in ('A_broad', 'A_linear', 'A_step', 'B', 'C'):
            common = ['--root', RUN / arm, '--data', OLD / 'data', '--cache', OLD / 'cache',
                      '--splits', OLD / 'data/splits.json', '--arm', arm, '--deadline-unix', DEADLINE,
                      '--control-gate', ROOT / 'runs/sound-anomaly-4b-control-v1/candidates/control_a/gate.json']
            for phase in ('gate', 'train'):
                run_stage(arm + '_' + phase, 'train_frozen_instruct_4b_v1.py', phase, *common)
                progress.update()
            assert read(RUN / arm / 'fit/complete.json')['epochs'] == 2
    write(RUN / 'status.json', {'status': 'complete', 'conditions': 5, 'epochs_each': 2})


if __name__ == '__main__':
    try:
        main()
    except Exception:
        failure = {'status': 'failed', 'traceback': traceback.format_exc()}
        write(RUN / f'failure_{time.time_ns()}.json', failure)
        write(RUN / 'status.json', failure)
        raise
