"""Explicit bounded waves of isolated frozen-head autoresearch candidates."""
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
    plan = read(args.plan)
    job = next(j for j in plan['jobs'] if j['name'] == args.name)
    root = Path(plan['root']) / job['name']
    root.mkdir(parents=True, exist_ok=False)
    script = Path(__file__).with_name('train_autoresearch_heads.py')
    flags = ['--root', str(root), '--data', job.get('data', plan['data']), '--cache', job.get('cache', plan['cache']),
             '--splits', job.get('splits', plan['splits']), '--readout', job['readout'],
             '--curve', job.get('curve','broad'), '--event-weight', str(job.get('event_weight',.1)),
             '--epochs', str(job.get('epochs',4)), '--train-split', job.get('train_split','train_small'),
             '--deadline-unix', str(plan['deadline_unix'])]
    if job.get('attention'):
        flags.append('--attention')
    write(root/'candidate.json',job)
    for phase in ('gate','train'):
        if time.time() >= plan['deadline_unix']:
            write(root/'status.json',{'status':'budget_stopped','before_phase':phase})
            return
        command=[sys.executable,'-u',str(script),phase,*flags]
        with (root/(phase+'.log')).open('w') as stream:
            child=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT)
        write(root/(phase+'_launch.json'),{'pid':child.pid,'command':command})
        code=child.wait()
        if code:
            write(root/'status.json',{'status':'failed','phase':phase,'returncode':code})
            return
    complete=root/'fit/complete.json'
    write(root/'status.json',{'status':'complete' if complete.exists() else 'budget_stopped'})


def main(args):
    plan=read(args.plan)
    out=Path(plan['root']);out.mkdir(parents=True,exist_ok=True)
    parallel=plan.get('max_parallel',4)
    for start in range(0,len(plan['jobs']),parallel):
        children=[]
        for job in plan['jobs'][start:start+parallel]:
            command=[sys.executable,'-u',str(Path(__file__)),'worker','--plan',str(args.plan),'--name',job['name']]
            with (out/(job['name']+'_worker.log')).open('w') as stream:
                child=subprocess.Popen(command,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(job['gpu'])),
                                       stdout=stream,stderr=subprocess.STDOUT)
            children.append(child)
            write(out/(job['name']+'_launch.json'),{'pid':child.pid,'gpu':job['gpu'],'command':command})
        codes=[c.wait() for c in children]
        write(out/f'wave{start//parallel}_returncodes.json',codes)
    result=[]
    for job in plan['jobs']:
        root=out/job['name'];status=read(root/'status.json') if (root/'status.json').exists() else {'status':'failed_worker'}
        entry={'candidate':job,'status':status}
        if status['status']=='complete':
            entry['selection']=read(root/'fit/selection.json')
            history=read(root/'fit/history.json')
            entry['dev_select']=next(h['dev_select'] for h in history if h['epoch']==entry['selection']['epoch'])
        result.append(entry)
    write(out/'results.json',result)
    write(out/'status.json',{'status':'complete','successful':sum(e['status']['status']=='complete' for e in result),
                           'candidates':len(result),'deadline_unix':plan['deadline_unix']})
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('run','worker'))
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--name')
    args=parser.parse_args()
    {'run':main,'worker':worker}[args.phase](args)
