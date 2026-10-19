"""Finite locked eight-cell queue, streamed tqdm, GPU3 ownership checked per cell."""
import os
from pathlib import Path
import subprocess
import sys
import time
from tqdm import tqdm
from prepare_whole_dialogs import write
from train_weighted_broad_lora_v1 import ROOT, HEAD_SHA
from train_whole_dialogs import file_hash

RUN = ROOT/'runs/weighted-broad-lora-ablation-v3'


def main():
    RUN.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).parent
    cells = [(a,c,w) for a in ('lora','frozen') for c in ('broad','step') for w in ('adaptive','uniform')]
    manifest = {str(p.relative_to(source)):file_hash(p) for p in source.rglob('*')
                if p.is_file() and '__pycache__' not in p.parts}
    write(RUN/'source_manifest.json', manifest)
    write(RUN/'authorization.json', {'user_request':'Бюджет забей. Также сделай ablation по поводу кривой и остального',
        'cells':cells,'epochs_each':2,'deadline':None,'warmstart_sha256':HEAD_SHA,
        'locked_unix':time.time(),'protocol_sha256':file_hash(source/'WEIGHTED_LORA_PROTOCOL.md')})
    for adaptation,curve,weighting in tqdm(cells,desc='Locked continuation factorial',unit='arm'):
        name = '_'.join((adaptation,curve,weighting))
        users = subprocess.check_output(['nvidia-smi','-i','3','--query-compute-apps=pid',
                                        '--format=csv,noheader,nounits'],text=True).strip()
        assert not users, f'GPU3 occupied: {users}'
        command = [sys.executable,'-u',str(source/'train_weighted_broad_lora_v1.py'),
            '--root',str(RUN/name),'--adaptation',adaptation,'--curve',curve,'--weighting',weighting]
        with (RUN/(name+'.log')).open('xb') as log:
            child = subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
            write(RUN/'status.json',{'status':'running','cell':name,'launcher_pid':os.getpid(),'child_pid':child.pid})
            print(f'LAUNCHED {name} PID={child.pid}',flush=True)
            while chunk := os.read(child.stdout.fileno(),8192):
                log.write(chunk); log.flush()
                sys.stdout.buffer.write(chunk); sys.stdout.buffer.flush()
            code = child.wait()
            if code:
                write(RUN/'status.json',{'status':'failed','cell':name,'returncode':code})
                raise RuntimeError(f'{name} failed; no automatic retry')
    write(RUN/'status.json',{'status':'complete','cells':8,'epochs_each':2})


if __name__ == '__main__':
    main()
