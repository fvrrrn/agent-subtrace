"""Offline evidence check: exact substrings or exact visible JSON field/value pairs."""
import json
from collections import Counter,defaultdict
from pathlib import Path
from annotate_safety_corpus import ROOT,DATA,strings


def field_pairs(value):
    if isinstance(value,dict):
        for k,v in value.items():
            if isinstance(v,(str,int,float,bool)) or v is None:yield str(k),v
            yield from field_pairs(v)
    elif isinstance(value,list):
        for v in value:yield from field_pairs(v)


def support(quote,state):
    if not isinstance(quote,str) or len(quote)<8:return None
    if any(quote in text for text in list(strings(state))+[json.dumps(state,ensure_ascii=False)]):return 'literal'
    if ':' not in quote:return None
    key,text=quote.split(':',1);key=key.strip().strip('"');text=text.strip()
    for k,value in field_pairs(state):
        if key!=k:continue
        representations=[json.dumps(value,ensure_ascii=False)]
        if isinstance(value,str):representations.append(value)
        if text in representations:return 'exact_json_field_value'
    return None


def main():
    folder=DATA/'annotations/oracle_controls'
    original=json.loads((ROOT/'archive/legacy-shm/danefedov-controlled-monotonic-verified-v1/results/records.json').read_text())
    states={r['key']:json.loads(r['state']) for r in original}
    rows=[json.loads(s) for s in (folder/'calls.jsonl').read_text().splitlines()]
    groups=defaultdict(dict)
    for r in rows:
        quotes=r.get('result',{}).get('evidence',[])
        r['structured_support']=[support(q,states[r['key']]) for q in quotes]
        groups[r['key']][r['gpu']]=r
    result={}
    for a,b in ((0,2),(1,3)):
        accepted=[]
        for key,g in groups.items():
            xx=[g[a],g[b]]
            if not all(x['status']=='complete' and x['finish_reason']=='stop' and x['structured_support'] and all(x['structured_support']) for x in xx):continue
            v=xx[0]['result']['verdict']
            if v=='uncertain' or v!=xx[1]['result']['verdict']:continue
            accepted.append({'key':key,'label':xx[0]['oracle_label'],'prediction':int(v=='unsafe')})
        errors=[r for r in accepted if r['label']!=r['prediction']]
        result[str((a,b))]={'n':len(accepted),'classes':dict(Counter(r['label'] for r in accepted)),
            'errors':errors,'accuracy':1-len(errors)/len(accepted) if accepted else None}
    report={'criterion':'Literal substring OR exact existing scalar JSON key:value. No semantic fuzzy matching, paraphrases, ellipses or arithmetic inference.',
        'results':result,'warning':'Post-hoc diagnostic on these 128 controls, not independent validation. Original outputs and original acceptance remain unchanged.'}
    (folder/'structured_evidence_audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':main()
