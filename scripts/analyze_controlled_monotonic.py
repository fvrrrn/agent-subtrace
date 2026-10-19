"""Offline matched step metrics; no model calls and no threshold selection."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import statistics

from graph_safety.io import read_json, read_jsonl, write_json
from graph_safety.metrics import evaluate


def metrics(rows):
    result=evaluate([r['label'] for r in rows],[r['score'] for r in rows])
    return {k:result[k] for k in ('n_examples','n_positive','positive_prevalence','average_precision','roc_auc')}


def interval(values):
    ordered=sorted(values)
    return [ordered[int(.025*(len(ordered)-1))],ordered[int(.975*(len(ordered)-1))]]


def paired_bootstrap(reference,candidate,repetitions=1000):
    keys=sorted(set(reference)&set(candidate))
    groups=defaultdict(list)
    for k in keys:
        assert reference[k]['label']==candidate[k]['label']
        assert reference[k]['input_sha256']==candidate[k]['input_sha256']
        groups[reference[k]['group']].append(k)
    names=sorted(groups);rng=random.Random(20261017)
    diffs=defaultdict(list)
    for _ in range(repetitions):
        sample=[k for name in rng.choices(names,k=len(names)) for k in groups[name]]
        a=metrics([reference[k] for k in sample]);b=metrics([candidate[k] for k in sample])
        for m in ('average_precision','roc_auc'):
            if a[m] is not None and b[m] is not None:diffs[m].append(b[m]-a[m])
    return {m:{'percentile_95_ci':interval(v),'bootstrap_samples':len(v)} for m,v in diffs.items()}


def mean_seed_bootstrap(rows,common,seeds,arm,repetitions=1000):
    groups=defaultdict(list)
    for k in sorted(common):groups[rows[f'noul_{seeds[0]}'][k]['group']].append(k)
    names=sorted(groups);rng=random.Random(20261017);diffs=defaultdict(list)
    for _ in range(repetitions):
        sample=[k for name in rng.choices(names,k=len(names)) for k in groups[name]]
        estimates=defaultdict(list)
        for seed in seeds:
            a=metrics([rows[f'noul_{seed}'][k] for k in sample])
            b=metrics([rows[f'{arm}_{seed}'][k] for k in sample])
            for m in ('average_precision','roc_auc'):
                if a[m] is not None and b[m] is not None:estimates[m].append(b[m]-a[m])
        for m,v in estimates.items():diffs[m].append(statistics.mean(v))
    return {m:{'percentile_97_5_ci_two_contrasts':[
        sorted(v)[int(.0125*(len(v)-1))],sorted(v)[int(.9875*(len(v)-1))]],
        'bootstrap_samples':len(v),'fixed_seeds':seeds} for m,v in diffs.items()}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--stage',choices=('pilot','full'),required=True)
    a=parser.parse_args();cfg=read_json(a.root/'protocol.json');folder=a.root/a.stage
    records={r['key']:r for r in read_json(a.root/'records.json')}
    summary={'stage':a.stage,'primary':'Step-level AP, original direction; ROC-AUC co-reported.',
             'no_trajectory_label_broadcast':True,'splits':{}}
    for split in ('validation',) if a.stage=='pilot' else ('test','external_test'):
        paths={'baseline':folder/f'baseline_{split}.jsonl'}
        for arm in cfg['arms']:
            for seed in cfg['seeds'][:1] if a.stage=='pilot' else cfg['seeds']:
                name=f'{arm}_{seed}';p=folder/name
                if a.stage=='pilot':
                    epoch=read_json(p/'checkpoint_audit.json')['selected_epoch'];path=p/f'validation_epoch{epoch}.jsonl'
                else:path=p/f'{split}.jsonl'
                paths[name]=path
        raw={name:list(read_jsonl(path)) for name,path in paths.items()}
        rows={name:{r['key']:r for r in rr if r['status']=='complete'} for name,rr in raw.items()}
        common=set.intersection(*(set(v) for v in rows.values()))
        expected={r['key'] for r in records.values() if r['split']==split}
        assert all({r['key'] for r in rr}==expected for rr in raw.values())
        output={'common_n':len(common),'planned_n':len(expected),'models':{},'contrasts':{},
                'rejections':{n:[r for r in rr if r['status']!='complete'] for n,rr in raw.items()}}
        for name,rr in rows.items():
            for k in common:
                assert rr[k]['label']==rows['baseline'][k]['label']
                assert rr[k]['input_sha256']==rows['baseline'][k]['input_sha256']
            result={'all_actions':metrics([rr[k] for k in sorted(common)]),'mechanisms':{}}
            for mechanism in sorted({records[k]['mechanism'] for k in common}):
                result['mechanisms'][mechanism]=metrics([rr[k] for k in common if records[k]['mechanism']==mechanism])
            if split in ('test','validation'):
                if cfg.get('verified_ordering'):
                    result['ordering_regime_slices']={regime:metrics([rr[k] for k in common
                        if int(records[k]['group'].split(':')[1])%3==j])
                        for j,regime in enumerate(('increasing','reversed','constant'))}
                result['action_slices']={}
                for action in ('fetch','copy','combine','deliver','charge','respond'):
                    subset=[rr[k] for k in common if json.loads(records[k]['state'])['proposed_action']['name']==action]
                    if subset:
                        safe=[r for r in subset if not r['label']]
                        result['action_slices'][action]={**metrics(subset),'safe_n':len(safe),
                            'false_alarms_at_fixed_0_5':sum(r['score']>=.5 for r in safe)}
                decisions=[k for k in common if json.loads(records[k]['state'])['proposed_action']['name'] in ('deliver','charge')]
                result['transmit_and_spend_only']=metrics([rr[k] for k in decisions])
                twins=defaultdict(list)
                for k in decisions:
                    r=records[k];twins[(r['group'],r['position'],r['action_sha256'])].append(k)
                pairs=[v for v in twins.values() if len(v)==2 and {rr[k]['label'] for k in v}=={0,1}]
                comparisons=[rr[next(k for k in pair if rr[k]['label'])]['score']-rr[next(k for k in pair if not rr[k]['label'])]['score'] for pair in pairs]
                result['counterfactuals']={'n':len(comparisons),'correct_direction':sum(v>0 for v in comparisons),
                                         'ties':sum(v==0 for v in comparisons)}
                result['constraint_compliance_not_severity_gold']={}
                for field,sign in (('previous_unsafe',1),('previous_safe',-1)):
                    pairs=[(records[k][field],k) for k in common if records[k][field] in common]
                    deltas=[sign*((rr[b]['logits'][1]-rr[b]['logits'][0])-(rr[e]['logits'][1]-rr[e]['logits'][0])) for e,b in pairs]
                    result['constraint_compliance_not_severity_gold'][field]={'n':len(deltas),'strict_direction':sum(d>0 for d in deltas)}
                if cfg.get('verified_ordering'):
                    previous={};probes=defaultdict(list)
                    for r in records.values():
                        k=r['key']
                        if k not in common:continue
                        trajectory=r['trajectory'];action=json.loads(r['state'])['proposed_action']['name']
                        if action not in ('deliver','charge'):
                            previous.pop(trajectory,None);continue
                        early=previous.get(trajectory)
                        if early and early['label']==r['label'] and early['ordering_value'] is not None and r['ordering_value'] is not None:
                            e=early['key'];stage_delta=r['ordering_value']-early['ordering_value']
                            prediction_delta=rr[k]['score']-rr[e]['score']
                            logodds_delta=(rr[k]['logits'][1]-rr[k]['logits'][0])-(rr[e]['logits'][1]-rr[e]['logits'][0])
                            direction='increasing' if stage_delta>0 else 'decreasing' if stage_delta<0 else 'constant'
                            probes[f'{action}:label{r["label"]}:{direction}'].append((stage_delta,prediction_delta,logodds_delta))
                        previous[trajectory]=r
                    result['ordinal_proxy_probe_including_reverse_and_equal']={
                        name:{'n':len(values),'strict_logodds_concordance':sum(x*z>0 for x,_,z in values) if not name.endswith('constant') else None,
                              'mean_absolute_probability_change':statistics.mean(abs(y) for _,y,_ in values)}
                        for name,values in probes.items()}
            output['models'][name]=result
        if a.stage=='full':
            for arm in ('unsafe_up','bidirectional'):
                for seed in cfg['seeds']:
                    x=f'noul_{seed}';y=f'{arm}_{seed}'
                    ref={k:rows[x][k] for k in common};cand={k:rows[y][k] for k in common}
                    output['contrasts'][f'{y}_minus_{x}']=paired_bootstrap(ref,cand)
                    output['contrasts'][f'{y}_minus_{x}']['point_differences']={
                        scope:{m:output['models'][y][scope][m]-output['models'][x][scope][m]
                               for m in ('average_precision','roc_auc')}
                        for scope in ('all_actions','transmit_and_spend_only') if scope in output['models'][y]}
                output['contrasts'][f'mean_{arm}_minus_noul']=mean_seed_bootstrap(rows,common,cfg['seeds'],arm)
            output['seed_means']={arm:{m:statistics.mean(output['models'][f'{arm}_{seed}']['all_actions'][m] for seed in cfg['seeds'])
                                      for m in ('average_precision','roc_auc')} for arm in cfg['arms']}
            output['inference_note']='Per-seed paired group-bootstrap intervals do not establish training-seed robustness; only three seeds. Two main loss-vs-NOUL contrasts and two evaluation domains must not be cherry-picked.'
        summary['splits'][split]=output
    write_json(folder/'comparison.json',summary)
    print(json.dumps({s:{n:v['all_actions'] for n,v in out['models'].items()} for s,out in summary['splits'].items()},indent=2))


if __name__=='__main__':main()
