"""Select/lock two distinct configurations using dev-select only, before confirmation."""
import argparse
import json
from pathlib import Path
import time

from graph_safety.metrics import evaluate
from prepare_whole_dialogs import write
from train_whole_dialogs import file_hash
from report_autoresearch_scores import load


def read(path):
    return json.loads(path.read_text())


def checked(path, expected):
    assert file_hash(path)==expected, path
    return {'path':str(path),'sha256':expected}


def collect(root):
    result=[]
    for stage in ('round1v2','round2','prompt_round','full_frozen'):
        path=root/stage/'results.json'
        if not path.exists():
            continue
        for entry in read(path):
            if entry['status']['status']!='complete':
                continue
            candidate=entry['candidate'];folder=root/stage/candidate['name'];s=read(folder/'fit/selection.json');c=s['config']
            source_name={'round1v2':'autoresearch-sixhour-round1v2-launch','round2':'autoresearch-sixhour-round2-launch',
                         'full_frozen':'autoresearch-sixhour-round2-launch','prompt_round':'autoresearch-sixhour-prompt-launch'}[stage]
            result.append({'name':stage+'_'+candidate['name'],'root':str(folder),'mode':'frozen',
                'source_root':str(root.parent/source_name/'source'),
                'training_protocol':checked(folder/'fit/run.json',file_hash(folder/'fit/run.json')),
                'readout':c['readout'],'attention':c['attention_before_mamba'],'curve':c['curve'],
                'event_weight':c['event_weight'],'train_split':c['train_split'],'epoch':s['epoch'],
                'ap':entry['dev_select']['average_precision'],'event_bce':entry['dev_select']['event_bce'],
                'head':checked(folder/'fit/selected_head.pt',s['head_sha256']),
                'dev_events':str(folder/'fit'/f'dev_select_epoch{s["epoch"]}'/'events.jsonl'),
                'prompt':stage=='prompt_round','selection':s})
    for name in ('lora_screen_best','lora_screen_mil','lora_last8','lora_full_best','fullft_small'):
        folder=root/name
        if not (folder/'status.json').exists() or read(folder/'status.json')['status']!='complete':
            continue
        s=read(folder/'selection.json');p=read(folder/'protocol.json');epoch=s['epoch'];cp=folder/f'epoch{epoch}'
        entry={'name':name,'root':str(folder),'mode':'fullft' if name=='fullft_small' else 'lora',
            'training_protocol':checked(folder/'protocol.json',file_hash(folder/'protocol.json')),
            'readout':p['readout'],'attention':p.get('attention',False),'curve':p['curve'],
            'event_weight':p['event_weight'],'train_split':p.get('train_split','train_small'),
            'lora_last_layers':p.get('lora_last_layers',0),
            'epoch':epoch,'ap':s['metrics']['average_precision'],'event_bce':s['metrics']['event_bce'],
            'dev_events':str(folder/f'dev_epoch{epoch}'/'predictions.jsonl'),'prompt':False,'selection':s}
        if name=='fullft_small':
            entry['head']=checked(cp/'head.pt',s['checkpoint_hashes']['head.pt'])
            entry['backbone']=checked(cp/'backbone.pt',s['checkpoint_hashes']['backbone.pt'])
        else:
            entry['head']=checked(cp/'head.pt',s['head_sha256'])
            files={str(f.relative_to(cp/'adapter')):file_hash(f) for f in sorted((cp/'adapter').rglob('*')) if f.is_file()}
            assert files==s['adapter_sha256']
            entry['adapter']={'path':str(cp/'adapter'),'files_sha256':files}
        result.append(entry)
    return sorted(result,key=lambda e:(-e['ap'],e['event_bce'],e['name']))


def main(args):
    all_candidates=collect(args.root)
    assert all_candidates
    selected=[];seen=set()
    for entry in all_candidates:
        signature=tuple(entry[k] for k in ('mode','readout','attention','curve','event_weight','prompt'))+(entry.get('lora_last_layers',0),)
        if signature in seen:
            continue
        selected.append(entry);seen.add(signature)
        if len(selected)==2:
            break
    out=args.root/'final_lock'
    out.mkdir(exist_ok=False)
    auth=read(args.root/'authorization.json')
    parent=args.root.parent/'whole-dialog-benchmarks-v1'
    spans=args.root.parent/'whole-dialog-token-replay-v1'
    template=read(parent/'atbench/audit.json')['template_sha256']
    weights=read(parent/'atbench/frozen_linear/complete.json')['frozen_weights_sha256']
    suites={}
    for suite in ('atbench','rjudge','dojo'):
        folder=parent/suite
        suites[suite]={k:{'path':str(p),'sha256':file_hash(p)} for k,p in {
            'cases':folder/'cases.json','spans':spans/suite/'spans.json',
            'audit':folder/'audit.json','shared_rejections':folder/'shared_rejections.json'}.items()}
        suites[suite]['cache']=str(folder/'features')
    for i,entry in enumerate(selected):
        # No confirmation predictions may preexist for the selected run.
        assert not (Path(entry['root'])/'evaluations/dev_confirm').exists()
        rows=load(Path(entry['dev_events']))
        metrics=evaluate([r['label'] for r in rows.values()],[r['score'] for r in rows.values()])
        curve=metrics.pop('pr_curve')
        points=[p for p in curve if p.get('threshold') is not None and p['recall'] is not None]
        best=max(points,key=lambda p:(min(p['precision'],p['recall']),p['f1'],p['recall'],p['threshold']))
        threshold_file=out/f'finalist{i}_threshold.json'
        write(threshold_file,{'split':'dev_select','threshold':best['threshold'],
            'criterion':'max min(P,R),ties F1/recall/higher threshold; no test fitting',
            'events_path':entry['dev_events'],'events_sha256':file_hash(Path(entry['dev_events'])),
            'n':len(rows),'fitted_point':best,'optimistic_on_dev_select':True})
        entry['threshold']=best['threshold']
        entry['threshold_file']=str(threshold_file)
        entry['threshold_sha256']=file_hash(threshold_file)
        entry['dev_events_sha256']=file_hash(Path(entry['dev_events']))
        plan={**{k:entry[k] for k in ('mode','head','readout','attention')},'final_only':True,
            'prompt':None,'template_sha256':template,'frozen_weights_sha256':weights,
            'threshold':best['threshold'],'threshold_source':{'split':'dev_select',
                'path':str(threshold_file),'sha256':file_hash(threshold_file)},
            'deadline_unix':auth['deadline_unix'],'deadline_reserve_seconds':180,
            'sequence_budget_seconds':180,'suites':suites,'candidate_name':entry['name']}
        if entry['mode']=='lora':plan['adapter']=entry['adapter']
        if entry['mode']=='fullft':plan['backbone']=entry['backbone']
        if entry['prompt']:
            raise RuntimeError('Prompted finalist requires independently prepared native external suite; no original-cache fallback')
        write(out/f'finalist{i}_external_plan.json',plan)
    write(out/'lock.json',{'locked_unix':time.time(),'criterion':'dev_select AP,ties eventBCE; best2distinct model/readout/curve/event-weight configurations, same-config small/full variants collapsed',
        'selected':selected,'all_completed_candidates':all_candidates,'confirm_used_for_selection':False,
        'external_used_for_selection':False,'deadline_unix':auth['deadline_unix']})
    print(json.dumps({'locked':[{'name':e['name'],'ap':e['ap'],'threshold':e['threshold']} for e in selected]},indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    main(parser.parse_args())
