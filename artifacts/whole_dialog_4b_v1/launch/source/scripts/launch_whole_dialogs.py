"""Two-GPU schedule: shared frozen cache, frozen +/- ranking, then LoRA +/- ranking."""
import argparse
import json
import os
from pathlib import Path
import select
import subprocess
import sys

from prepare_whole_dialogs import write


def main(a):
    try:
        fd=os.pidfd_open(a.gate_pid)
    except ProcessLookupError:
        fd=None
    if fd is not None:
        print('Waiting for whole-dialog CUDA gate',a.gate_pid,flush=True)
        select.select([fd],[],[]);os.close(fd)
    gate=json.loads((a.root/'cuda_gate.json').read_text())
    assert gate['status']=='passed'
    script=Path(__file__).with_name('train_whole_dialogs.py')
    stages=[('cache',[['cache','--shard','0'],['cache','--shard','1']]),
            ('frozen',[['train','--arm','frozen_linear'],['train','--arm','frozen_linear_mono']]),
            ('lora',[['train','--arm','lora_linear'],['train','--arm','lora_linear_mono']])]
    for stage,jobs in stages:
        children=[]
        for gpu,arguments in zip((2,3),jobs):
            command=[sys.executable,'-u',str(script),*arguments,'--root',str(a.root),'--data',str(a.data)]
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu))
            with (a.root/f'{stage}_gpu{gpu}.log').open('w') as out:
                p=subprocess.Popen(command,env=env,stdout=out,stderr=subprocess.STDOUT)
            children.append(p)
            write(a.root/f'{stage}_gpu{gpu}_launch.json',{'pid':p.pid,'command':command,'gpu':gpu})
            print('START',stage,gpu,p.pid,flush=True)
        codes=[p.wait() for p in children]
        write(a.root/f'{stage}_status.json',{'returncodes':codes})
        if any(codes):
            write(a.root/'status.json',{'status':'failed','stage':stage,'returncodes':codes})
            raise SystemExit(1)
        if stage=='cache':
            first=json.loads((a.root/'features/complete0.json').read_text())
            second=json.loads((a.root/'features/complete1.json').read_text())
            assert first['weights_sha256']==second['weights_sha256']
            assert first['template_sha256']==second['template_sha256']
        print('END',stage,codes,flush=True)
    from graph_safety.metrics import evaluate
    outputs={};selections={};initials=set()
    for arm in ('frozen_linear','frozen_linear_mono','lora_linear','lora_linear_mono'):
        folder=a.root/'arms'/arm
        selection=json.loads((folder/'selection.json').read_text());selections[arm]=selection
        initials.add(json.loads((folder/'run.json').read_text())['initial_head_sha256'])
        outputs[arm]={r['key']:r for r in map(json.loads,(folder/f"validation_epoch{selection['epoch']}.jsonl").read_text().splitlines())}
    assert len(initials)==1
    common=sorted(set.intersection(*(set(v) for v in outputs.values())))
    assert all(len(v)==len(common) for v in outputs.values())
    for key in common:
        assert len({(v[key]['label'],v[key]['state_sha256'],v[key]['ids_sha256']) for v in outputs.values()})==1
    metrics={}
    for arm,out in outputs.items():
        m=evaluate([out[k]['label'] for k in common],[out[k]['score'] for k in common]);m.pop('pr_curve');metrics[arm]=m
    write(a.root/'comparison.json',{'common_actions':len(common),'metrics':metrics,'selections':selections,
          'limitations':'Validation-selected one-seed comparison. Soft curves are artificial token targets. No benchmark evaluated.'})
    write(a.root/'status.json',{'status':'complete','arms':4,'epochs_each':2})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--gate-pid',type=int,required=True)
    main(p.parse_args())
