"""Causal action candidates; preserve label provenance and quarantine evaluation overlap."""
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import re

ROOT=Path('/var/local/danefedov/experiments/agent-safety/datasets/corpus-v1')


def dump(x):return json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':'))
def digest(x):return hashlib.sha256(dump(x).encode()).hexdigest()
def normalized(x):return ' '.join(str(x).lower().split())
def family(x):
    s=normalized(x)
    s=re.sub(r'\b[0-9a-f]{8}-[0-9a-f-]{27,}\b','<id>',s)
    s=re.sub(r'https?://\S+','<url>',s)
    s=re.sub(r'[\w.+-]+@[\w.-]+','<email>',s)
    return re.sub(r'\b\d+(?:\.\d+)?\b','<number>',s)


def main():
    import argparse
    import pyarrow.parquet as pq
    parser=argparse.ArgumentParser();parser.add_argument('--output-name',default='candidates')
    args=parser.parse_args()
    raw=ROOT/'raw';out=ROOT/'derived'/args.output_name;out.mkdir(parents=True,exist_ok=False)
    holdout=json.loads((ROOT/'holdout_blocklist.json').read_text())
    blocked=set(holdout['request_hashes']);blocked_family=set(holdout['request_family_hashes'])
    candidates={};quarantine=[];counts=Counter();trajectory_counts=Counter()
    def add(source,trajectory,position,tools,user,history,action,label=None,quality='unlabelled',domain=None,family_id=None,trajectory_label=None):
        state={'tool_interfaces':tools,'user_request':user,'observed_history':history,'proposed_action':action}
        h=digest(state);request_hash=digest(normalized(user));group=digest(family(user))
        provenance={'source':source,'trajectory':trajectory,'position':position,'source_label':label,'label_quality':quality,'domain':domain,'family_id':family_id or str(trajectory),'trajectory_label':trajectory_label}
        if request_hash in blocked or group in blocked_family:
            quarantine.append({'input_sha256':h,'reason':'evaluation_request_overlap','provenance':provenance});return
        if h in candidates:
            candidates[h]['provenance'].append(provenance);counts['exact_duplicate_actions']+=1;return
        candidates[h]={'key':source+':'+str(trajectory)+':'+str(position),'state':state,'input_sha256':h,
            'group':group,'request_hash':request_hash,'provenance':[provenance],
            'split':'validation' if int(group[:8],16)%20==0 else 'train','label':None,'status':'awaiting_review'}
        counts[source]+=1
    align=json.loads((raw/'jc-ryan--AgentAlign/agent_align_data_v3.json').read_text())
    for r in align:
        history=[];user=next((m.get('content','') for m in r['messages'] if m['role']=='user'),'')
        tools=[{k:v for k,v in t.items() if k!='category'} for t in r['tools']]
        for i,m in enumerate(r['messages']):
            if m['role']=='assistant':add('AgentAlign',r['id'],i,tools,user,history,dict(m),None,'aligned_response_not_action_gold',r.get('category'))
            history=history+[m]
        trajectory_counts['AgentAlign']+=1
    safety=json.loads((raw/'jinjinyien--ToolSafety/toolsafety.json').read_text())
    for n,r in enumerate(safety):
        history=[];user=next((m.get('value','') for m in r['conversations'] if m['from'] in ('human','user')),'')
        for i,m in enumerate(r['conversations']):
            if m['from'] in ('assistant','gpt'):
                add('ToolSafety',str(n),i,r.get('system',''),user,history,m['value'],None,'aligned_response_not_action_gold')
            history=history+[{'role':m['from'],'content':m['value']}]
        trajectory_counts['ToolSafety']+=1
    # Reuse the inspected official parser; its trajectory-level output is NOT action gold.
    script=raw/'carriegu0818--StepSafe/snapshot/scripts/build_data.py'
    spec=importlib.util.spec_from_file_location('stepsafe_source_parser',script)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    for config in ('AgentDoG-BinarySafety','AgentDoG-FineGrainedTaxonomy'):
        source_rows=json.loads((raw/'AI45Research--AgentDoG1.0-Training-Data'/config/'train.json').read_text())
        for n,r in enumerate(source_rows):
            try:events,profile,goal=module.convert(r)
            except (ValueError,KeyError,IndexError,StopIteration) as error:
                quarantine.append({'reason':'agentdog_parse_error','source':config,'index':n,'error':repr(error)});continue
            history=[]
            for e in events:
                visible={k:v for k,v in e.items() if k in ('surface','agent_id','agent_role','tool_name','tool_args','tool_result','content','target_agent','target_role') and v is not None}
                if e['agent_role']=='agent':
                    add('AgentDoG',f'{config}:{n}',e['sequence_num'],profile,goal,history,
                        {k:v for k,v in visible.items() if k!='tool_result'},None,'trajectory_label_not_broadcast',config,
                        trajectory_label=r['output'] if config=='AgentDoG-BinarySafety' else 'unsafe_by_source_config')
                history=history+[visible]
            trajectory_counts['AgentDoG']+=1
    ss=ROOT/'derived/stepsafe_rebuilt/rl'
    tasks=pq.read_table(ss/'train.parquet').to_pylist();ids={r['env_kwargs']['audit_id'] for r in tasks}
    for line in (ss/'trajectory_annotations.jsonl').open():
        r=json.loads(line)
        if r['audit_id'] not in ids:continue
        events=r['trajectory'];events=json.loads(events) if isinstance(events,str) else events
        labels={s['sequence_num']:s for s in r['step_labels']};history=[]
        user=next((e.get('content','') for e in events if e.get('agent_role')=='user'),'')
        task=next((t['env_kwargs'] for t in tasks if t['env_kwargs']['audit_id']==r['audit_id']),{})
        user=task.get('goal') or user
        tools=task.get('profile') or r.get('profile') or task.get('role_prompts') or ''
        for e in events:
            visible={k:v for k,v in e.items() if k in ('surface','agent_id','agent_role','tool_name','tool_args','tool_result','content','target_agent','target_role') and v is not None}
            seq=e['sequence_num']
            if seq in labels:
                lab=labels[seq];target=lab.get('current_action_target');violated=lab.get('policy_violated')
                value=1 if target=='STOP' and violated is True else 0 if target=='CONTINUE' and violated is False else None
                action={k:v for k,v in visible.items() if k!='tool_result'}
                add('StepSafe',r['audit_id'],seq,tools,user,history,action,value,'published_current_action_annotation' if value is not None else 'ambiguous_intervention_target',r.get('domain'),r.get('contrastive_pair_id') or r.get('root_source_audit_id') or r['audit_id'])
            history=history+[visible]
        trajectory_counts['StepSafe']+=1
    base=raw/'glo26--stepshield/snapshot/data'
    for part in ('train','generated_benign'):
        for path in sorted((base/part).rglob('*.jsonl')):
            try:r=json.loads(path.read_text())
            except json.JSONDecodeError:continue
            history=[];task=r.get('task',{});user=dump(task)
            for i,s in enumerate(r.get('steps',[])):
                label=1 if s.get('label') not in (None,'','safe','SAFE',0) else None
                add('StepShield',r['trajectory_id'],s.get('step',i),'Tool names and arguments are provided in the observed actions.',user,history,
                    {'name':s.get('action'),'arguments':s.get('arguments'),'thought':s.get('thought')},label,
                    'published_divergence_marker' if label==1 else 'unlabelled_not_assumed_safe',part,re.sub(r'-(CLEAN|ROGUE)$','',r['trajectory_id']))
                history=history+[{'role':'agent','action':s.get('action'),'arguments':s.get('arguments'),'thought':s.get('thought')},
                    {'role':'tool','content':s.get('observation')}]
            trajectory_counts['StepShield']+=1
    # ToolSafe is retained separately: published eval data are not silently relabelled as train.
    rows=list(candidates.values());conflicts=[]
    # Keep request families AND source counterfactual families in one connected split.
    parent={}
    def find(x):
        parent.setdefault(x,x)
        if parent[x]!=x:parent[x]=find(parent[x])
        return parent[x]
    def union(a,b):
        a,b=find(a),find(b)
        if a!=b:parent[max(a,b)]=min(a,b)
    for r in rows:
        for p in r['provenance']:union('request:'+r['group'],p['source']+':family:'+p['family_id'])
    for r in rows:
        r['group']=digest(find('request:'+r['group']))
        r['split']='validation' if int(r['group'][:8],16)%20==0 else 'train'
    with (out/'actions.jsonl').open('w') as f:
        for r in rows:
            labels={p['source_label'] for p in r['provenance'] if p['source_label'] is not None}
            if len(labels)>1:r['status']='source_label_conflict';conflicts.append(r['key'])
            f.write(dump(r)+'\n')
    (out/'quarantine.jsonl').write_text(''.join(dump(r)+'\n' for r in quarantine))
    report={'unique_action_candidates':len(rows),'contributions':dict(counts),'source_trajectories':dict(trajectory_counts),
        'splits':dict(Counter(r['split'] for r in rows)),'published_label_candidates':sum(any(p['source_label'] is not None for p in r['provenance']) for r in rows),
        'evaluation_overlap_quarantined':len(quarantine),'source_label_conflicts':conflicts,
        'training_ready':False,'counterfactual_family_components_grouped':True,'limitations':['Not all samples are labeled; candidate count is not accepted training count.',
            'Exact full-input dedup and normalized/entity-masked request grouping applied. Semantic near-duplicate audit still required.',
            'StepSafe labels were made with hindsight; local evidence and incomplete-context audit needed.',
            'StepShield null labels are not interpreted as safe. No commands from trajectories are executed.']}
    (out/'summary.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
