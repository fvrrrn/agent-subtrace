"""Final-only six-suite queue on four GPUs, after locked checkpoint replay passes."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from prepare_whole_dialogs import write


def read(path):
    return json.loads(path.read_text())


def worker(args):
    root=args.root/'external'
    mapping={0:(0,('atbench',)),1:(1,('atbench',)),2:(0,('rjudge','dojo')),3:(1,('rjudge','dojo'))}
    index,suites=mapping[args.gpu]
    lock=read(args.root/'final_lock/lock.json')
    if index>=len(lock['selected']):return
    name=lock['selected'][index]['name'];folder=root/name
    codes={}
    for suite in suites:
        if time.time()+300>=lock['deadline_unix']:
            write(root/f'gpu{args.gpu}_budget_stop.json',{'status':'budget_stopped','before_suite':suite})
            break
        cmd=[sys.executable,'-u',str(Path(__file__).with_name('evaluate_autoresearch_external.py')),
             '--plan',str(args.root/'final_lock'/f'finalist{index}_external_plan.json'),
             '--root',str(folder),'--suite',suite]
        with (root/(name+'_'+suite+'.log')).open('w') as out:
            child=subprocess.Popen(cmd,stdout=out,stderr=subprocess.STDOUT)
        write(root/(name+'_'+suite+'_launch.json'),{'pid':child.pid,'gpu':args.gpu,'command':cmd})
        codes[suite]=child.wait()
        if codes[suite]:break
    write(root/f'gpu{args.gpu}_returncodes.json',codes)


def main(args):
    assert read(args.root/'confirmation/status.json')['status']=='complete'
    root=args.root/'external';root.mkdir(exist_ok=False)
    children=[]
    for gpu in range(4):
        cmd=[sys.executable,'-u',str(Path(__file__)),'worker','--root',str(args.root),'--gpu',str(gpu)]
        with (root/f'gpu{gpu}_worker.log').open('w') as out:
            child=subprocess.Popen(cmd,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2'),
                                   stdout=out,stderr=subprocess.STDOUT)
        children.append(child)
        write(root/f'gpu{gpu}_launch.json',{'pid':child.pid,'command':cmd,'gpu':gpu})
    codes=[p.wait() for p in children]
    lock=read(args.root/'final_lock/lock.json');reports={};all_complete=True
    for entry in lock['selected']:
        reports[entry['name']]={}
        for suite in ('atbench','rjudge','dojo'):
            folder=root/entry['name']/suite
            if (folder/'complete.json').exists():
                done=read(folder/'complete.json');summary=read(folder/'summary.json')
                reports[entry['name']][suite]={'complete':done,'summary':summary}
                block=summary['results']['all_fully_evaluated']['action_gold' if suite=='dojo' else 'all_events']
                print(entry['name'],suite,json.dumps({k:block[k] for k in ('control_0.5','dev_select_threshold')},indent=2),flush=True)
            else:
                all_complete=False
                reports[entry['name']][suite]={'status':'incomplete','failure':read(folder/'failure.json') if (folder/'failure.json').exists() else None}
    write(root/'report.json',reports)
    write(root/'status.json',{'status':'complete' if all_complete else 'incomplete','worker_returncodes':codes,
        'no_threshold_or_model_fitting_on_external':True})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('run','worker'))
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--gpu',type=int)
    args=parser.parse_args()
    {'run':main,'worker':worker}[args.phase](args)
