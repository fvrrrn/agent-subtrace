"""Run immutable candidate evaluator with its exact source imports and a wall deadline."""
import argparse
import hashlib
import json
from pathlib import Path
import runpy
import signal
import sys
import time


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lock',type=Path,required=True)
    parser.add_argument('--name',required=True)
    parser.add_argument('--split',required=True)
    parser.add_argument('--store',type=Path,required=True)
    args=parser.parse_args()
    lock=json.loads(args.lock.read_text());entry=next(e for e in lock['selected'] if e['name']==args.name)
    assert entry['mode']=='frozen'
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    assert sha(entry['head']['path'])==entry['head']['sha256']
    assert sha(entry['training_protocol']['path'])==entry['training_protocol']['sha256']
    selection=json.loads((Path(entry['root'])/'fit/selection.json').read_text())
    assert selection==entry['selection']
    remaining=lock['deadline_unix']-time.time()-180
    assert remaining>0, 'Deadline reached before frozen evaluation'
    def expired(signum,frame):
        raise TimeoutError('Locked frozen evaluation reached persistence reserve')
    signal.signal(signal.SIGALRM,expired);signal.setitimer(signal.ITIMER_REAL,remaining)
    source=Path(entry['source_root']);sys.path.insert(0,str(source/'src'));sys.path.insert(0,str(source/'scripts'))
    cfg=entry['selection']['config'];root=args.lock.parent.parent
    script=source/'scripts/train_autoresearch_heads.py'
    sys.argv=[str(script),'eval','--root',entry['root'],'--data',str(args.store/'datasets/whole-dialog-4b-v2'),
        '--cache',str(args.store/'runs/whole-dialog-4b-v3/features'),'--splits',str(root/'data/splits.json'),
        '--readout',entry['readout'],'--curve',entry['curve'],'--event-weight',str(entry['event_weight']),
        '--train-split',entry['train_split'],'--seed',str(cfg['seed']),'--checkpoint',entry['head']['path'],
        '--eval-split',args.split,'--deadline-unix',str(lock['deadline_unix'])]
    if entry['attention']:sys.argv.append('--attention')
    runpy.run_path(str(script),run_name='__main__')


if __name__=='__main__':main()
