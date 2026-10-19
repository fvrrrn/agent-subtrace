"""Unsafe-increasing and safe-decreasing NOUL training, with frozen references."""
import argparse
from collections import Counter, defaultdict
from pathlib import Path
import shutil

from .io import read_json, read_jsonl, write_json, sha256
from .metrics import evaluate


def prepare(root, reference, ablation, raw):
    root.mkdir(exist_ok=False)
    cfg=read_json(reference/'protocol.json')
    rows=read_json(reference/'records.json')
    if sha256(reference/'records.json')!=cfg['records_sha256']:
        raise ValueError('Reference data hash mismatch')
    source_audit=read_json(raw.parent/'source_audit.json')
    sources={}
    for entry in source_audit['files']:
        p=raw/entry['file']
        if sha256(p)!=entry['sha256']: raise ValueError('Raw data changed')
        for i,r in enumerate(read_json(p)): sources[f'{entry["file"]}:{i}']=r
    groups=defaultdict(list)
    for r in rows: groups[tuple(r['trajectory'])].append(r)
    ambiguous=[]
    for items in groups.values():
        for late in items:
            late['previous_safe']=None
            if late['label']: continue
            obs=sources[late['key']]
            candidates=[r for r in items if not r['label'] and r['segment']<late['segment']
                and r['group']==late['group']
                and obs['history'].startswith(sources[r['key']]['history'])
                and sources[r['key']]['current_action'] in obs['history'][len(sources[r['key']]['history']):]]
            if candidates:
                nearest=[r for r in candidates if r['segment']==max(x['segment'] for x in candidates)]
                if len(nearest)==1: late['previous_safe']=nearest[0]['key']
                else: ambiguous.append(late['key'])
    indexed={r['key']:r for r in rows}
    for r in rows:
        for field,label in [('previous_safe',0),('previous_unsafe',1)]:
            if r[field]:
                p=indexed[r[field]]
                assert r['label']==p['label']==label and r['split']==p['split'] and p['segment']<r['segment']
    write_json(root/'records.json',rows)
    shutil.copy2(reference/'atbench_cases.json',root/'atbench_cases.json')
    shutil.copytree(reference/'baseline',root/'baseline')
    for a,p in {'increase':reference,'decrease':ablation/'decrease','noul_only':ablation/'noul_only'}.items():
        dest=root/'references'/a
        dest.mkdir(parents=True)
        shutil.copytree(p/'trained',dest/'trained')
        for name in ('selected_checkpoint.json','training_environment.json','admission.json'):
            shutil.copy2(p/name,dest/name)
        epoch=read_json(p/'selected_checkpoint.json')['epoch']
        shutil.copy2(p/f'epoch_{epoch}.json',dest/f'epoch_{epoch}.json')
    cfg.update(records_sha256=sha256(root/'records.json'),rank_direction='increase',safe_rank_weight=.1,
        loss='Mean NOUL CE + 0.1 * mean unsafe hinge(0.2 + early - late) + 0.1 * mean safe hinge(0.2 + late - early), raw yes-minus-no logodds. Each pair class normalized separately.',
        parent_records_sha256=sha256(reference/'records.json'),
        code_sha256={p.name:sha256(p) for p in Path(__file__).parent.glob('*.py')})
    cfg['limitations'] += ['Safe means source label 0, not necessarily remediation.',
        'Extra safe-pair forwards change dropout RNG consumption versus previous runs; same seed but not identical stochastic path.',
        'Total regularizer weight increases from 0.1 to 0.2; comparison does not isolate sign structure from total regularization weight.']
    write_json(root/'protocol.json',cfg)
    write_json(root/'safe_pairs.json',{'counts':dict(Counter(r['split'] for r in rows if r['previous_safe'])),
        'ambiguous_not_used':ambiguous})


def compare(root):
    locations={'baseline':root/'baseline','bidirectional':root/'trained',
               **{a:root/'references'/a/'trained' for a in ('increase','decrease','noul_only')}}
    rows={a:{r['key']:r for r in read_jsonl(p/'records.jsonl')} for a,p in locations.items()}
    planned={r['key'] for r in read_json(root/'atbench_cases.json')}
    if any(set(v)!=planned for v in rows.values()): raise ValueError('Incomplete evaluation')
    envs=[read_json(p/'environment.json') for p in locations.values()]
    if any(e['environment']!=envs[0]['environment'] or e['temperature']!=envs[0]['temperature'] for e in envs):
        raise ValueError('Evaluation environment mismatch')
    if read_json(root/'admission.json')!=read_json(root/'references/increase/admission.json'):
        raise ValueError('Data admission differs')
    if read_json(root/'training_environment.json')!=read_json(root/'references/increase/training_environment.json'):
        raise ValueError('Initialization/environment differs')
    common=sorted(k for k in planned if all(v[k]['status']=='complete' for v in rows.values()))
    for k in common:
        ref=rows['baseline'][k]
        for v in rows.values():
            if v[k]['label']!=ref['label'] or [s['input_sha256'] for s in v[k]['steps']]!=[s['input_sha256'] for s in ref['steps']]:
                raise ValueError('Inputs differ')
    write_json(root/'all_arms_comparison.json',{'common_count':len(common),
        'labels':dict(Counter(rows['baseline'][k]['label'] for k in common)),
        'arms':{a:evaluate([v[k]['label'] for k in common],[v[k]['score'] for k in common]) for a,v in rows.items()},
        'exclusions':{a:[r for r in v.values() if r['status']!='complete'] for a,v in rows.items()},
        'limitations':read_json(root/'protocol.json')['limitations']})
    # Assess both constraints using only saved, selected validation predictions.
    folders={'bidirectional':root,**{a:root/'references'/a for a in ('increase','decrease','noul_only')}}
    vals={}
    for a,p in folders.items():
        epoch=read_json(p/'selected_checkpoint.json')['epoch']
        vals[a]={r['key']:r for r in read_json(p/f'epoch_{epoch}.json')['validation']}
    temperature=envs[0]['temperature']
    report={}
    for field,sign in [('previous_unsafe',1),('previous_safe',-1)]:
        pairs=[(r[field],r['key']) for r in read_json(root/'records.json') if r['split']=='validation'
               and r[field] and all(r[field] in v and r['key'] in v for v in vals.values())]
        report[field]={}
        for a,v in vals.items():
            details=[]
            for early,late in pairs:
                x,y=v[early],v[late]
                delta=((y['logits'][1]-y['logits'][0])-(x['logits'][1]-x['logits'][0]))*temperature
                details.append({'early':early,'late':late,'early_score':x['score'],'late_score':y['score'],
                                'delta_raw_logodds':delta,'desired_delta':sign*delta})
            report[field][a]={'n':len(details),'strict_desired_direction':sum(r['desired_delta']>0 for r in details),
                'ties':sum(r['desired_delta']==0 for r in details),'margin_met':sum(r['desired_delta']>=.2 for r in details),
                'pairs':details}
    write_json(root/'validation_monotonicity.json',report)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=('prepare','compare'));p.add_argument('--root',type=Path,required=True)
    p.add_argument('--reference',type=Path);p.add_argument('--ablation',type=Path);p.add_argument('--raw',type=Path)
    a=p.parse_args()
    if a.phase=='prepare':prepare(a.root,a.reference,a.ablation,a.raw)
    else:compare(a.root)


if __name__=='__main__':main()
