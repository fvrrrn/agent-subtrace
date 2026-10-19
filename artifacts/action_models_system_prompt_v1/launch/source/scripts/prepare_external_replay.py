"""Independent evaluation-only ATBench and Nemotron final-report adapters."""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil

import train_action_models as training
from build_safety_candidates import normalized, family
from graph_safety.annotation_cohort import flatten

BASE = Path('/var/local/danefedov/experiments/agent-safety')


def overlap(cases):
    bundle = BASE / 'datasets/data-bundle-v1/training/records.json'
    known = [json.loads(r['state'])['user_request'] for r in training.read(bundle)]
    norm, masked = {normalized(x) for x in known}, {family(x) for x in known}
    hits = []
    for case in cases:
        for message in case['messages']:
            if message['role'] == 'user' and (normalized(message['content']) in norm or family(message['content']) in masked):
                hits.append(case['key']); break
    return {'bundle_sha256': training.sha(bundle), 'matching_cases': hits,
            'limitation': 'Lexical normalized/masked requests, not proof of semantic independence.'}


def atbench(out):
    raw = BASE / 'datasets/corpus-v1/raw/AI45Research--ATBench'
    manifest = training.read(raw / 'SOURCE_MANIFEST.json')
    path = raw / 'ATBench/test.json'
    assert training.sha(path) == '80c534b5f3517c872b528a2e6b64e34495206968edd184f4aed563d39f0cca09'
    cases = []
    for index, row in enumerate(training.read(path)):
        messages = [{k: v for k,v in m.items() if k in ('role','content','thought','action')}
                    for m in flatten(row['contents'])]
        assert all(m['role'] in ('user','agent','environment') for m in messages)
        assert row['label'] in (0,1)
        cases.append({'key':f'atbench:{index}:{row["id"]}', 'label':row['label'],
                      'tools':[{k:t[k] for k in ('name','description','parameters')} for t in row['tool_used']],
                      'messages':messages, 'source_split':'ATBench/test'})
    assert len(cases)==1000
    audit=overlap(cases)
    out.mkdir(parents=True,exist_ok=False)
    training.write(out/'overlap_audit.json',audit)
    assert not audit['matching_cases'], 'Training overlap needs review before inference'
    training.write(out/'cases.json',cases)
    training.write(out/'manifest.json',{'source':manifest,'file_sha256':training.sha(path),
        'cases':len(cases),'label_counts':dict(Counter(c['label'] for c in cases)),
        'evaluation_only':True,'label_scope':'trajectory','excluded_fields':['reason','risk_source','failure_mode','real_world_harm','tool._source','tool._original_description'],
        'protocol':'All actions incl Complete primary; tool-only sensitivity fixed before inference. No action labels inferred.'})


def decode(value):
    for _ in range(3):
        if isinstance(value,str):
            try:
                value=json.loads(value);continue
            except ValueError:
                pass
        break
    return value


def nemotron(out):
    import pyarrow.parquet as pq
    source=BASE/'datasets/corpus-v1/raw/nvidia--Nemotron-AIQ-Agentic-Safety-Dataset-1.0'
    out.mkdir(parents=True,exist_ok=False)
    raw=out/'raw';shutil.copytree(source,raw)
    source_manifest=training.read(raw/'SOURCE_MANIFEST.json')
    for entry in source_manifest['files']:
        assert training.sha(raw/entry['path'])==entry['sha256']
    cases,rejected,inventory=[],[],{}
    allowed=('topic','report_organization','search_web','rag_collection','num_queries','llm_name','reflection_count','report_size')
    for config in ('safety','security'):
        for split in ('with_defense','without_defense'):
            path=raw/f'data/{config}_data/{split}/data-00000-of-00001.parquet'
            file=pq.ParquetFile(path);counts=Counter()
            for batch in file.iter_batches(batch_size=16):
                for row in batch.to_pylist():
                    snapshot=decode(row['attack_snapshot']);trace=decode(row['trace'])
                    key=f'nemotron:{config}:{split}:{row["trace_id"]}'
                    result=snapshot['result'];counts[str(result.get('success'))]+=1
                    if config=='security':
                        continue  # Node risk/attack success is not binary action-safety gold.
                    roots=[s for s in trace if s.get('name')=='ai_researcher' and not s.get('parent_id')]
                    reason=None
                    if len(roots)!=1:reason='ambiguous_root'
                    elif result.get('error') not in (None,'','None'):reason='source_error'
                    elif type(result.get('success')) is not bool:reason='no_binary_final_outcome'
                    else:
                        root=roots[0];request=decode(root.get('attributes.input.value'))
                        output=decode(root.get('attributes.output.value'))
                        if not isinstance(request,dict) or not isinstance(request.get('topic'),str):reason='missing_visible_request'
                        elif not isinstance(output,str) or not output.strip():reason='missing_final_report'
                        elif root.get('status_code') not in ('OK',None):reason='root_error'
                    if reason:
                        rejected.append({'key':key,'reason':reason});continue
                    user=json.dumps({k:request[k] for k in allowed if k in request},ensure_ascii=False)
                    cases.append({'key':key,'label':int(result['success']), 'tools':[],
                        'messages':[{'role':'user','content':user},
                                    {'role':'agent','action':json.dumps({'response':output},ensure_ascii=False)}],
                        'source_split':f'{config}/{split}'})
            inventory[f'{config}/{split}']={'rows':file.metadata.num_rows,'source_success_counts':dict(counts)}
    audit=overlap(cases)
    training.write(out/'overlap_audit.json',audit)
    training.write(out/'inventory.json',inventory)
    training.write(out/'rejections.json',rejected)
    assert len({c['key'] for c in cases})==len(cases)
    assert not audit['matching_cases'], 'Training overlap needs review'
    training.write(out/'cases.json',cases)
    training.write(out/'manifest.json',{'source':source_manifest,'verified_raw_copy':True,'evaluation_only':True,
        'cases':len(cases),'source_inventory':inventory,'rejected':len(rejected),
        'label_scope':'Safety source success: attack reached final workflow span; not gold for every action or entire execution.',
        'target':'Retrospective final-report safety, conditioned on visible root request. No claimed reconstruction of internal causal tool actions.',
        'input_policy':'Only source root topic and visible workflow configuration, plus final report as proposed text action. No attack snapshots, labels, judge rationales, ground_truth, threat IDs or category fields.',
        'security_status':'Preserved and inventoried, not assigned fabricated binary safety labels.',
        'comparability':'Not the unpublished 100 manually relabeled execution-trace subset in StepGuard Table8. No direct reproduction claim.',
        'limitations':'Source final-attack labels may not capture all unsafe behavior. Internal history/tool schemas not reconstructed in final-report adapter; this measures response screening, not proactive tool-use prevention.'})
    print(json.dumps({'cases':len(cases),'rejected':len(rejected),'inventory':inventory},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('dataset',choices=('atbench','nemotron'))
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    {'atbench':atbench,'nemotron':nemotron}[a.dataset](a.output)
