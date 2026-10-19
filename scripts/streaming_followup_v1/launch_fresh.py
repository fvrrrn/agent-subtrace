"""Validate all prior 8+3, lock four fresh-init cells, execute once sequentially."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import numpy as np
from tqdm import tqdm
from prepare_whole_dialogs import write
from launch_followup import completed, sha
from graph_safety.metrics import evaluate

BASE=Path('/var/local/danefedov/experiments/agent-safety')
RUN=BASE/'runs/fresh-factorial-v1'
SOURCE=Path(__file__).parent


def read(p):
    return json.loads(p.read_text())


def main():
    RUN.mkdir(exist_ok=False)
    try:
        assert read(BASE/'runs/weighted-broad-recovery-v1/status.json')['status']=='complete'
        follow=BASE/'runs/streaming-followup-v2'
        assert read(follow/'status.json')['status']=='complete'
        previous=BASE/'runs/weighted-broad-recovery-v1/completed_eight'
        names=['_'.join(c) for c in read(previous/'authorization.json')['cells']]
        candidates=[c['name'] for c in read(follow/'lock.json')['cells']]
        results={}; reference=None; hashes={}
        for arm in tqdm([previous/n for n in names]+[follow/n for n in candidates],desc='Admit complete prior 8+3',unit='arm'):
            selected,status,events,identity=completed(arm)
            if reference is None: reference=identity
            assert identity==reference
            results[arm.name]={'selection':selected,'seconds':status['seconds']}
            hashes[str(arm/'selection.json')]=sha(arm/'selection.json')
            hashes[str(arm/'protocol.json')]=sha(arm/'protocol.json')
        # User's specified rule: coefficient chosen among the three follow-up
        # cells, not retroactively selecting a different coefficient from control.
        best=min(candidates,key=lambda n:(-results[n]['selection']['metrics']['average_precision'],
            results[n]['selection']['metrics']['event_bce'],results[n]['seconds'],n))
        coefficient=next(c['event_weight'] for c in read(follow/'lock.json')['cells'] if c['name']==best)
        cells=[[a,c] for a in ('lora','frozen') for c in ('step','broad')]
        lock={'cells':cells,'epochs_each':2,'event_weight':coefficient,'coefficient_selected_from':best,
            'positive_weight':1.,'weighting':'uniform','readout':'endpoint','initialization':'fresh shared head, fresh adapters',
            'criterion':'dev AP, ties lower unweighted event BCE, then measured cost',
            'prior_results':results,'prior_hashes':hashes,'locked_unix':time.time(),
            'protocol_sha256':sha(SOURCE/'FRESH_PROTOCOL.md'),
            'source_manifest':{str(p.relative_to(SOURCE)):sha(p) for p in SOURCE.rglob('*')
                if p.is_file() and '__pycache__' not in p.parts},
            'data_hashes':read(BASE/'runs/fresh-factorial-research-v1/diagnostics.json')['data_hashes']}
        write(RUN/'lock.json',lock)
        for adaptation,curve in tqdm(cells,desc='Locked fresh 2x2 factorial',unit='arm'):
            name=adaptation+'_'+curve
            assert not subprocess.check_output(['nvidia-smi','-i','3','--query-compute-apps=pid',
                '--format=csv,noheader,nounits'],text=True).strip(), 'GPU3 occupied'
            with (RUN/(name+'.log')).open('xb') as log:
                child=subprocess.Popen([sys.executable,'-u',str(SOURCE/'train_fresh.py'),'--root',str(RUN/name),
                    '--lock',str(RUN/'lock.json'),'--adaptation',adaptation,'--curve',curve],
                    stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                write(RUN/'status.json',{'status':'running','cell':name,'scheduler_pid':os.getpid(),'child_pid':child.pid})
                print(f'LAUNCHED {name} PID={child.pid} LOG={log.name}',flush=True)
                assert child.wait()==0, 'Training failed; no automatic retry'
            assert read(RUN/name/'status.json')['status']=='complete'
        report={}
        for adaptation,curve in cells:
            name=adaptation+'_'+curve; arm=RUN/name; selection=read(arm/'selection.json'); epoch=selection['epoch']
            assert read(arm/'initialization.json')==read(RUN/'shared_initialization.json')
            assert read(arm/'frozen_audit.json')['unchanged']
            assert sha(arm/f'epoch{epoch}/head.pt')==selection['head_sha256']
            events=[json.loads(s) for s in (arm/f'dev_epoch{epoch}/predictions.jsonl').read_text().splitlines()]
            identity=sorted((e['key'],e['dialog'],e['label'],e['ids_sha256'],e['state_sha256']) for e in events)
            assert identity==reference
            slices={}
            for source in sorted({s for e in events for s in e['sources']}):
                group=[e for e in events if source in e['sources']]
                m=evaluate([e['label'] for e in group],[e['score'] for e in group]); m.pop('pr_curve',None)
                slices[source]=m
            report[name]={'selection':selection,'status':read(arm/'status.json'),'source_slices':slices,
                'class_score_quantiles':{str(label):np.quantile([e['score'] for e in events if e['label']==label],
                    [0,.1,.25,.5,.75,.9,1]).tolist() for label in (0,1)}}
        write(RUN/'comparison.json',{'paired_actions':886,'results':report,
            'interpretation':'one-seed dev-selected fresh-init factorial; prior continuation has additional training budget, not matched initialization'})
        write(RUN/'status.json',{'status':'complete','cells':4,'epochs_each':2})
    except BaseException as error:
        write(RUN/'failure.json',{'error':repr(error),'traceback':traceback.format_exc()})
        write(RUN/'status.json',{'status':'failed','error':repr(error),'automatic_retry':False})
        raise


if __name__=='__main__':
    main()
