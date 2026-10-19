"""Bounded CPU diagnostics: train/dev only; no threshold fitting or model updates."""
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import struct
import torch
from tqdm import tqdm
from prepare_whole_dialogs import write
from graph_safety.metrics import evaluate

BASE = Path('/var/local/danefedov/experiments/agent-safety')
DATA = BASE/'runs/frozen-instruct-4b-v1/data'
OUT = BASE/'runs/fresh-factorial-research-v1'


def read(path):
    return json.loads(path.read_text())


def main():
    OUT.mkdir(exist_ok=False)
    rows = read(DATA/'dialogs.json'); splits = read(DATA/'splits.json')
    assert set(splits) == {'train_small','dev_select'}
    index = {r['key']:r for r in rows}
    report = {}; conflicts = {}
    for split, keys in splits.items():
        groups = {kind:defaultdict(list) for kind in ('endpoint','first_action_token','action_without_history')}
        mass = defaultdict(lambda: {'safe':0.,'unsafe':0.,'safe_n':0,'unsafe_n':0})
        per_action = []
        for key in tqdm(keys, desc='CPU causal identities '+split, unit='dialog'):
            row = index[key]; spans = row['spans']; ids = row['ids']
            cuts = {s['end'] for s in spans} | {s['start']+1 for s in spans}
            prefix = {}; hasher = hashlib.sha256()
            for i, token in enumerate(ids, 1):
                hasher.update(struct.pack('<I', token))
                if i in cuts:
                    prefix[i] = hasher.hexdigest()
            for s in spans:
                label = int(s['label']); length = s['end']-s['start']
                record = {'dialog':key,'action':s['key'],'label':label,'length':length,'history_tokens':s['start'],
                    'effective_action_mass':1/len(spans)/len(keys)}
                per_action.append(record)
                groups['endpoint'][prefix[s['end']]].append(record)
                groups['first_action_token'][prefix[s['start']+1]].append(record)
                action_hash = hashlib.sha256(struct.pack('<'+'I'*length,*ids[s['start']:s['end']])).hexdigest()
                groups['action_without_history'][action_hash].append(record)
                for group in ['all','length:'+('<=32' if length<=32 else '33-128' if length<=128 else '>128')]+['source_union:'+s for s in row['sources']]:
                    field = 'unsafe' if label else 'safe'
                    mass[group][field] += record['effective_action_mass']; mass[group][field+'_n'] += 1
        conflicts[split] = {kind:{h:rs for h,rs in grouped.items() if len({r['label'] for r in rs})>1}
            for kind,grouped in groups.items()}
        report[split] = {'mass':dict(mass),'conflicting_groups':{kind:len(v) for kind,v in conflicts[split].items()},
            'interpretation':'first-token label disagreement can be unresolved future; action-only disagreement can require history, neither proves endpoint mislabeling'}
        write(OUT/(split+'_action_mass.json'), per_action)
    write(OUT/'causal_conflicts.json', conflicts)
    aggregate = BASE/'runs/weighted-broad-recovery-v1/completed_eight'
    arms = {'step_control':aggregate/'lora_step_uniform','broad_control':aggregate/'lora_broad_uniform',
        'positive_event1':BASE/'runs/streaming-followup-v2/on_event1'}
    report['dev_diagnostics'] = {}
    for name, arm in tqdm(list(arms.items()), desc='CPU stored prediction diagnostics', unit='arm'):
        selection = read(arm/'selection.json'); epoch = selection['epoch']
        events = [json.loads(line) for line in (arm/f'dev_epoch{epoch}/predictions.jsonl').read_text().splitlines()]
        assert len(events)==886
        groups = defaultdict(list); false_negatives = []
        for e in events:
            row = index[e['dialog']]; span = next(s for s in row['spans'] if s['key']==e['key'])
            length=span['end']-span['start']
            for group in ['all','length:'+('<=32' if length<=32 else '33-128' if length<=128 else '>128'),
                'history:'+('<=1024' if span['start']<=1024 else '>1024')]+['source_union:'+s for s in row['sources']]:
                groups[group].append(e)
            if e['label'] and e['score']<.5:
                false_negatives.append({'key':e['key'],'score':e['score'],'length':length,'history_tokens':span['start'],
                    'effective_dialog_action_mass':1/len(row['spans'])/len(splits['dev_select'])})
        slices={}
        for group, es in groups.items():
            metrics=evaluate([e['label'] for e in es],[e['score'] for e in es]); metrics.pop('pr_curve',None)
            slices[group]=metrics
        # Analytic loss gradients in output-logit space, not parameter gradients.
        gradient_rows=[]
        curve = 'broad' if name=='broad_control' else 'step'
        pos_weight=7.733163050192017 if name=='positive_event1' else 1.
        coefficient=1. if name=='positive_event1' else .1
        for key in tqdm(splits['dev_select'], desc=name+' logit-gradient diagnostic', unit='dialog'):
            saved=torch.load(arm/f'dev_epoch{epoch}/tokens/{key}.pt',map_location='cpu',weights_only=True)
            assert saved['ids_sha256']==index[key]['ids_sha256']
            for span in saved['spans']:
                p=saved['raw_logits'][span['start']:span['end']].float().sigmoid()
                n=len(p); y=float(span['label']); endpoint_target=y*((.9+1.2-1.5)/1.14 if curve=='broad' else 1.)
                gt=((1-endpoint_target)*p[-1]-pos_weight*endpoint_target*(1-p[-1]))/n
                ge=coefficient*((1-y)*p[-1]-pos_weight*y*(1-p[-1]))
                gradient_rows.append({'key':span['key'],'label':y,'length':n,'token_endpoint_gradient':gt.item(),
                    'weighted_event_gradient':ge.item(),'opposing_at_endpoint':bool(gt*ge<0)})
        write(OUT/(name+'_logit_gradients.json'),gradient_rows)
        report['dev_diagnostics'][name]={'selection':selection,'slices':slices,'false_negatives':false_negatives,
            'endpoint_opposition_count':sum(g['opposing_at_endpoint'] for g in gradient_rows),
            'gradient_scope':'endpoint logit derivative only; not shared-parameter gradient conflict'}
    report['data_hashes']={n:hashlib.sha256((DATA/n).read_bytes()).hexdigest() for n in ('dialogs.json','splits.json')}
    write(OUT/'diagnostics.json',report)
    print(json.dumps({s:report[s]['conflicting_groups'] for s in splits},indent=2),flush=True)


if __name__=='__main__':
    main()
