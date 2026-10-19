"""Replay locked dev outputs before untouched confirmation; no adaptive model selection."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import select
import signal

import torch

from prepare_whole_dialogs import write
from report_autoresearch_scores import load


def read(path):
    return json.loads(path.read_text())


def paths(entry, split):
    folder=Path(entry['root'])/'evaluations'/split
    return folder/('events.jsonl' if entry['mode']=='frozen' else 'predictions.jsonl'),folder/'tokens'


def command(entry, split, args):
    source=Path(__file__).parent
    if entry['mode']=='frozen':
        return [sys.executable,'-u',str(source/'evaluate_autoresearch_frozen_snapshot.py'),
                '--lock',str(args.root/'final_lock/lock.json'),'--name',entry['name'],
                '--split',split,'--store',str(args.store)]
    return [sys.executable,'-u',str(source/'evaluate_autoresearch_lora_dev.py'),
            '--train-root',entry['root'],'--mode',entry['mode'],'--split',split,
            '--lock',str(args.root/'final_lock/lock.json'),'--name',entry['name'],
            '--deadline-unix',str(args.deadline_unix)]


def replay_audit(entry):
    ref=Path(entry['dev_events']);new,new_tokens=paths(entry,'dev_select')
    a,b=load(ref),load(new)
    assert set(a)==set(b)
    max_score=0.
    for key in a:
        for field in ('label','state_sha256','ids_sha256'):
            assert a[key][field]==b[key][field]
        max_score=max(max_score,abs(a[key]['score']-b[key]['score']))
    assert max_score<=1e-6, f'Event-score replay drift {max_score}'
    old_tokens=ref.parent/'tokens';max_raw=0.;count=0
    for p in old_tokens.glob('*.pt'):
        x=torch.load(p,weights_only=True,map_location='cpu')
        y=torch.load(new_tokens/p.name,weights_only=True,map_location='cpu')
        assert x['ids_sha256']==y['ids_sha256']
        for oldkey,newkey in (('raw_token_logits','raw_token_logits'),('raw_logits','raw_logits'),('null_logits','null_logits')):
            if oldkey not in x or x[oldkey] is None:continue
            if newkey not in y:newkey='raw_logits' if oldkey=='raw_token_logits' else 'raw_token_logits'
            error=(x[oldkey]-y[newkey]).abs().max().item()
            max_raw=max(max_raw,error)
        count+=1
    assert count>0 and max_raw==0.,f'Raw token replay drift {max_raw}'
    return {'status':'passed','common_events':len(a),'dialogs':count,'max_event_error':max_score,'max_raw_logit_error':max_raw}


def main(args):
    lock=read(args.root/'final_lock/lock.json');entries=lock['selected']
    args.deadline_unix=lock['deadline_unix']
    out=args.root/'confirmation';out.mkdir(exist_ok=False)
    try:
        for split in ('dev_select','dev_confirm','original_validation'):
            assert time.time()+360<args.deadline_unix, 'Insufficient confirmation phase budget'
            children=[]
            for gpu,entry in enumerate(entries):
                cmd=command(entry,split,args)
                with (out/(entry['name']+'_'+split+'.log')).open('w') as stream:
                    child=subprocess.Popen(cmd,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1'),
                                           stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                children.append((entry,child))
                write(out/(entry['name']+'_'+split+'_launch.json'),{'pid':child.pid,'gpu':gpu,'command':cmd})
            codes={}
            for entry,child in children:
                try:fd=os.pidfd_open(child.pid)
                except ProcessLookupError:fd=None
                if fd is not None:
                    ready=select.select([fd],[],[],max(0,args.deadline_unix-time.time()-180))[0]
                    if not ready:os.killpg(child.pid,signal.SIGINT)
                    if not ready and not select.select([fd],[],[],60)[0]:os.killpg(child.pid,signal.SIGTERM)
                    os.close(fd)
                codes[entry['name']]=child.wait()
            write(out/(split+'_returncodes.json'),codes)
            assert not any(codes.values()),f'{split} failed: {codes}'
            if split=='dev_select':
                audits={e['name']:replay_audit(e) for e in entries}
                write(out/'replay_audit.json',audits)
        for entry in entries:
            script=Path(__file__).with_name('report_autoresearch_scores.py')
            cmd=[sys.executable,str(script),'--select',entry['dev_events'],'--evaluation',
                 str(paths(entry,'dev_confirm')[0]),str(paths(entry,'original_validation')[0]),
                 '--locked-threshold',entry['threshold_file'],'--expected-threshold-sha256',entry['threshold_sha256'],
                 '--deadline-unix',str(args.deadline_unix),
                 '--output',str(out/(entry['name']+'_locked_threshold_report.json'))]
            subprocess.run(cmd,check=True,stdout=subprocess.DEVNULL)
        if len(entries)==2:
            assert time.time()+120<args.deadline_unix, 'Insufficient bootstrap budget'
            cmd=[sys.executable,str(Path(__file__).with_name('bootstrap_autoresearch_comparison.py')),
                 '--events-a',str(paths(entries[0],'dev_confirm')[0]),'--events-b',str(paths(entries[1],'dev_confirm')[0]),
                 '--manifest',str(args.root/'data/dev_confirm.manifest.json'),
                 '--threshold-a',str(entries[0]['threshold']),'--threshold-b',str(entries[1]['threshold']),
                 '--repetitions','1000','--output',str(out/'paired_component_bootstrap.json')]
            subprocess.run(cmd,check=True,stdout=subprocess.DEVNULL)
        write(out/'status.json',{'status':'complete','models':[e['name'] for e in entries],
            'confirm_used_for_model_selection':False,'thresholds_fit_on':'dev_select_only'})
        for entry in entries:
            report=read(out/(entry['name']+'_locked_threshold_report.json'))
            print(entry['name'],json.dumps(report['evaluations'],indent=2),flush=True)
    except Exception as error:
        write(out/'failure.json',{'status':'failed','error':repr(error),'no_automatic_retry':True})
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--store',type=Path,required=True)
    main(parser.parse_args())
