"""Conservative lexical near-duplicate grouping before teacher annotation or training."""
import json
from pathlib import Path
import re
from datasketch import MinHash,MinHashLSH
from build_safety_candidates import ROOT,family,digest


def shingles(text):
    words=re.findall(r'\w+|<\w+>',family(text))
    width=min(5,len(words))
    return {' '.join(words[i:i+width]) for i in range(len(words)-width+1)} if width else {''}


def main():
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--input-name',default='candidates_v2');args=parser.parse_args()
    folder=ROOT/'derived'/args.input_name
    requests={};groups={}
    for line in (folder/'actions.jsonl').open():
        r=json.loads(line);h=digest(family(r['state']['user_request']))
        requests[h]=r['state']['user_request'];groups.setdefault(h,set()).add(r['group'])
    evals=json.loads((ROOT/'evaluation_requests.json').read_text())
    index=MinHashLSH(threshold=.75,num_perm=64);stored={};parent={};blocked=set();matches=[]
    def find(x):
        parent.setdefault(x,x)
        if parent[x]!=x:parent[x]=find(parent[x])
        return parent[x]
    def union(a,b):
        a,b=find(a),find(b)
        if a!=b:parent[max(a,b)]=min(a,b)
    for i,(key,text) in enumerate([(f'eval:{i}',t) for i,t in enumerate(evals)]+sorted(requests.items())):
        s=shingles(text);m=MinHash(num_perm=64,seed=20261018);m.update_batch([v.encode() for v in sorted(s)])
        hits=[]
        for other in index.query(m):
            t=stored[other];jac=len(s&t)/len(s|t)
            if jac>=.85:hits.append((other,jac))
        if not key.startswith('eval:'):
            own=sorted(groups[key])
            for g in own[1:]:union(own[0],g)
            for other,jac in hits:
                if other.startswith('eval:'):
                    blocked.update(own);matches.append({'request_family':key,'evaluation':other,'jaccard':jac})
                else:
                    for g in groups[other]:union(own[0],g)
        stored[key]=s;index.insert(key,m)
        if i%5000==0:print('request audit',i,flush=True)
    mapping={g:find(g) for values in groups.values() for g in values}
    blocked_components={find(g) for g in blocked}
    report={'method':'64-permutation MinHash candidate retrieval + exact 5-word-shingle Jaccard >= .85; entity-masked requests',
        'request_families':len(requests),'group_mapping':mapping,'blocked_components':sorted(blocked_components),
        'evaluation_matches':matches,'limitations':['Lexical overlap audit does not prove semantic independence. Tool/schema reuse is allowed; scenario family overlap is grouped.']}
    (folder/'overlap_audit.json').write_text(json.dumps(report,indent=2));print('audit complete',len(matches),'evaluation matches',len(blocked_components),'blocked components',flush=True)


if __name__=='__main__':main()
