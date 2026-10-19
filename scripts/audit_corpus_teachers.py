"""Check annotation teachers on existing interpreter-labeled controls, never on new test labels."""
import asyncio
import argparse
from collections import defaultdict
import json
import time
import httpx
from annotate_safety_corpus import ROOT,DATA,call,stable


async def main(thinking,guard=False,split='train'):
    path=ROOT/'archive/legacy-shm/danefedov-controlled-monotonic-verified-v1/results/records.json'
    if not path.exists():path=ROOT/'archive/legacy-shm/danefedov-controlled-monotonic-verified-v1/records.json'
    data=json.loads(path.read_text())
    rows=[]
    for label in (0,1):
        subset=sorted((r for r in data if r['split']==split and r['label']==label),key=lambda r:stable('teacher-control:'+r['key']))[:64]
        rows.extend({'key':r['key'],'state':json.loads(r['state']),'label':r['label']} for r in subset)
    assert len(rows)==128
    name='oracle_controls_guard' if guard else 'oracle_controls_reasoning' if thinking else 'oracle_controls'
    folder=DATA/'annotations'/(name+('_validation' if split=='validation' else ''));folder.mkdir(exist_ok=False)
    workers=json.loads((ROOT/'runs/corpus-v1/serving-v3/readiness.json').read_text());sems={w['gpu']:asyncio.Semaphore(4) for w in workers}
    if thinking:workers=[w for w in workers if w['gpu'] in (1,3)]
    if thinking and split=='validation':workers=[w for w in workers if w['gpu']==3]
    if guard:
        workers=[json.loads((ROOT/'runs/corpus-v1/serving-gptoss/readiness.json').read_text())]
        assert workers[0]['status']=='ready'
    outputs=[];start=time.monotonic()
    async with httpx.AsyncClient(timeout=900,limits=httpx.Limits(max_connections=24)) as client:
        with (folder/'calls.jsonl').open('w') as stream:
            async def one(w,r):
                async with sems[w['gpu']]:result=await call(client,w,r,thinking=thinking)
                # Oracle label is attached only AFTER the independent request.
                result['oracle_label']=r['label'];outputs.append(result)
                stream.write(json.dumps(result)+'\n');stream.flush()
            await asyncio.gather(*(one(w,r) for w in workers for r in rows))
    summary={}
    for w in workers:
        rr=[r for r in outputs if r['gpu']==w['gpu']]
        judged=[r for r in rr if r['status']=='complete' and r['result']['verdict']!='uncertain']
        correct=sum(int(r['result']['verdict']=='unsafe')==r['oracle_label'] for r in judged)
        summary[str(w['gpu'])]={'model':w['model'],'mtp':w['mtp'],'attempts':len(rr),'binary_verdicts':len(judged),
            'correct':correct,'accuracy_on_binary':correct/len(judged) if judged else None,
            'grounded_evidence':sum(r.get('evidence_grounded',False) for r in rr)}
    result={'controls':f'128 balanced original verified synthetic {split} actions, interpreter labels withheld from requests',
        'models':summary,'thinking':thinking,'seconds':time.monotonic()-start,'limitation':'Implementation/explicit-policy check, not general or human-annotated accuracy on the new corpus.'}
    (folder/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--thinking',action='store_true');p.add_argument('--guard',action='store_true');p.add_argument('--split',choices=('train','validation'),default='train');a=p.parse_args()
    asyncio.run(main(a.thinking,a.guard,a.split))
