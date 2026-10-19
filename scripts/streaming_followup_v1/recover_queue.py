"""Recover precisely six original conditions, then one registered three-cell round."""
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from tqdm import tqdm
from prepare_whole_dialogs import write
from launch_followup import completed, sha

BASE = Path('/var/local/danefedov/experiments/agent-safety')
OLD = BASE/'runs/weighted-broad-lora-ablation-v3'
RUN = BASE/'runs/weighted-broad-recovery-v1'
FOLLOWUP = BASE/'runs/streaming-followup-v2'
SOURCE = Path(__file__).parent


def main():
    RUN.mkdir(exist_ok=False)
    try:
        for pid in (4134159, 473298):
            assert not Path(f'/proc/{pid}').exists(), f'Old PID still exists: {pid}'
        cells = list(itertools.product(('lora','frozen'), ('broad','step'), ('adaptive','uniform')))
        original_source = OLD.parent/'weighted-broad-lora-ablation-v3-launch/source'
        hashes = {}
        for path in original_source.rglob('*'):
            if path.is_file() and '__pycache__' not in path.parts:
                relative = path.relative_to(original_source)
                assert sha(SOURCE/relative) == sha(path), f'Original source changed: {relative}'
                hashes[str(relative)] = sha(path)
        write(RUN/'original_source_equality.json', hashes)
        write(RUN/'source_manifest.json', {str(p.relative_to(SOURCE)):sha(p) for p in SOURCE.rglob('*')
            if p.is_file() and '__pycache__' not in p.parts})
        write(RUN/'authorization.json', {'user_request':'Продолжай эксперименты; recover original six then prepared three',
            'cells':cells,'epochs_each':2,'new_conditions':3,'original_conditions_recovered':6,
            'scheduler_pid':os.getpid(),'locked_unix':time.time(),'no_automatic_retries':True,
            'training_source_unchanged':True,'logging':'regular files, detached session, DEVNULL stdin'})
        aggregate = RUN/'completed_eight'; aggregate.mkdir()
        write(aggregate/'authorization.json', json.loads((OLD/'authorization.json').read_text()))
        reference = None; manifest = {}
        for config in tqdm(cells, desc='Original locked eight: reuse2 recover6', unit='cell'):
            name = '_'.join(config)
            reuse = config in cells[:2]
            arm = OLD/name if reuse else RUN/name
            if not reuse:
                users = subprocess.check_output(['nvidia-smi','-i','3','--query-compute-apps=pid',
                    '--format=csv,noheader,nounits'], text=True).strip()
                assert not users, f'GPU3 occupied: {users}'
                command = [sys.executable,'-u',str(SOURCE/'train_weighted_broad_lora_v1.py'),
                    '--root',str(arm),'--adaptation',config[0],'--curve',config[1],'--weighting',config[2]]
                with (RUN/(name+'.log')).open('xb') as log:
                    child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
                    write(RUN/'status.json', {'status':'running','cell':name,'child_pid':child.pid,'scheduler_pid':os.getpid()})
                    print(f'LAUNCHED {name} PID={child.pid} LOG={log.name}', flush=True)
                    code = child.wait()
                    assert code == 0, f'{name} failed ({code}), no retry'
            selection, status, events, identity = completed(arm)
            if reference is None:
                reference = identity
            assert identity == reference
            (aggregate/name).symlink_to(arm, target_is_directory=True)
            manifest[name] = {'path':str(arm),'reused':reuse,
                'selection_sha256':sha(arm/'selection.json'),'protocol_sha256':sha(arm/'protocol.json'),
                'status_sha256':sha(arm/'status.json')}
            write(RUN/'completed_manifest.json',manifest)
        write(aggregate/'status.json', {'status':'complete','cells':8,'epochs_each':2})
        subprocess.run([sys.executable,'-u',str(SOURCE/'summarize_weighted_broad_lora_v1.py'),str(aggregate)], check=True)
        env = dict(os.environ, FOLLOWUP_ROOT=str(FOLLOWUP), FOLLOWUP_PARENT=str(aggregate))
        with (RUN/'followup_scheduler.log').open('xb') as log:
            child = subprocess.Popen([sys.executable,'-u',str(SOURCE/'launch_followup.py')], env=env,
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            write(RUN/'status.json', {'status':'followup','child_pid':child.pid,'scheduler_pid':os.getpid()})
            assert child.wait() == 0, 'Follow-up stopped; no retry'
        write(RUN/'status.json', {'status':'complete','recovered_cells':6,'new_followup_cells':3})
    except BaseException as error:
        write(RUN/'failure.json', {'error':repr(error),'traceback':traceback.format_exc(),'automatic_retry':False})
        write(RUN/'status.json', {'status':'failed','error':repr(error)})
        raise


if __name__ == '__main__':
    main()
