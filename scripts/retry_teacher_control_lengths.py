"""One longer-output retry of length failures only; preserve every initial control call."""
import asyncio
from collections import Counter,defaultdict
import json
import httpx
from annotate_safety_corpus import ROOT,DATA,call
from audit_structured_evidence import support


async def main():
    folder=DATA/'annotations/control_length_retry';folder.mkdir(exist_ok=False)
    original=json.loads((ROOT/'archive/legacy-shm/danefedov-controlled-monotonic-verified-v1/results/records.json').read_text())
    states={r['key']:json.loads(r['state']) for r in original}
    guard=json.loads((ROOT/'runs/corpus-v1/serving-gptoss/readiness.json').read_text())
    qwen=next(w for w in json.loads((ROOT/'runs/corpus-v1/serving-v3/readiness.json').read_text()) if w['gpu']==3)
    all_calls={};todo=[]
    for suffix in ('','_validation'):
        for name,w in [('oracle_controls_guard',guard),('oracle_controls_reasoning',qwen)]:
            rows=[json.loads(s) for s in (DATA/f'annotations/{name}{suffix}/calls.jsonl').read_text().splitlines()]
            rows=[r for r in rows if r['teacher']==w['model']]
            for r in rows:
                key=(suffix,r['key'],r['teacher']);all_calls[key]=r
                choices=(r.get('raw_response') or {}).get('choices',[])
                if r['status']=='error' and choices and choices[0]['finish_reason']=='length':todo.append((key,w,r))
    sem={0:asyncio.Semaphore(4),3:asyncio.Semaphore(4)}
    async with httpx.AsyncClient(timeout=900) as client:
        with (folder/'calls.jsonl').open('w') as stream:
            async def one(key,w,old):
                row={'key':old['key'],'state':states[old['key']]}
                async with sem[w['gpu']]:r=await call(client,w,row,thinking=True,max_tokens=4096)
                r['oracle_label']=old['oracle_label'];r['source_control_split']='validation' if key[0] else 'train'
                all_calls[key]=r;stream.write(json.dumps(r)+'\n');stream.flush()
            await asyncio.gather(*(one(*item) for item in todo))
    summary={}
    for suffix in ('','_validation'):
        groups=defaultdict(list)
        for (split,key,teacher),r in all_calls.items():
            if split==suffix:groups[key].append(r)
        agreed=[];accepted=[]
        for key,rr in groups.items():
            if len(rr)!=2 or any(r['status']!='complete' or r['result']['verdict']=='uncertain' for r in rr):continue
            if rr[0]['result']['verdict']!=rr[1]['result']['verdict']:continue
            result={'key':key,'label':rr[0]['oracle_label'],'prediction':int(rr[0]['result']['verdict']=='unsafe')};agreed.append(result)
            if all(r['result']['evidence'] and all(support(q,states[key]) for q in r['result']['evidence']) for r in rr):accepted.append(result)
        summary[suffix or 'train']={'agreement':len(agreed),'agreement_errors':[r for r in agreed if r['label']!=r['prediction']],
            'accepted':len(accepted),'accepted_classes':dict(Counter(r['label'] for r in accepted)),
            'accepted_errors':[r for r in accepted if r['label']!=r['prediction']]}
    report={'retry_count':len(todo),'max_tokens':4096,'results':summary,'no_semantic_disagreement_retry':True}
    (folder/'summary.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':asyncio.run(main())
