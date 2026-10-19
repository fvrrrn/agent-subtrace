"""Frozen controls for the completed Kev-0.8B unsafe-step training experiment."""
import argparse
from collections import Counter
from pathlib import Path
import shutil

from .io import read_json, read_jsonl, write_json, sha256
from .metrics import evaluate


def prepare(root, reference):
    root.mkdir(exist_ok=False)
    cfg = read_json(reference/'protocol.json')
    for name, field in [('records.json','records_sha256'),('atbench_cases.json','atbench_sha256')]:
        if sha256(reference/name) != cfg[field]: raise ValueError('Reference data changed')
    shared = root/'reference'
    shared.mkdir()
    for name in ('protocol.json','admission.json','training_environment.json','selected_checkpoint.json'):
        shutil.copy2(reference/name,shared/name)
    for arm in ('baseline','trained'):
        shutil.copytree(reference/arm,shared/arm)
    for arm in ('decrease','noul_only'):
        folder = root/arm
        folder.mkdir()
        for name in ('records.json','atbench_cases.json'):
            shutil.copy2(reference/name,folder/name)
        shutil.copytree(reference/'baseline',folder/'baseline')
        protocol = dict(cfg)
        protocol.update(rank_direction='decrease' if arm=='decrease' else 'increase',
                        rank_weight=.1 if arm=='decrease' else 0.,
                        ablation_arm=arm, reference_protocol_sha256=sha256(reference/'protocol.json'),
                        code_sha256={p.name:sha256(p) for p in Path(__file__).parent.glob('*.py')})
        protocol['loss'] = ('NOUL CE + 0.1 * mean relu(0.2 + later_unsafe_raw_logodds - earlier_unsafe_raw_logodds)'
                            if arm=='decrease' else 'NOUL cross-entropy only; ranking coefficient exactly zero')
        protocol['ablation_control'] = 'Same initialization, data/order, seed, optimizer, two epochs and validation selection. Earlier-step forwards retained even with zero rank weight to match dropout RNG consumption. Baseline predictions reused from reference after matched-input/environment checks.'
        write_json(folder/'protocol.json',protocol)
    write_json(root/'protocol.json',{'arms':['baseline','increase','decrease','noul_only'],
        'reference':str(reference), 'reference_protocol_sha256':sha256(reference/'protocol.json'),
        'new_runs':['decrease','noul_only'], 'no_threshold_search':True,
        'design':'One fixed seed; increase and baseline reuse completed results. New controls run sequentially on GPU 3.'})


def compare(root):
    locations={'baseline':root/'reference/baseline','increase':root/'reference/trained',
               'decrease':root/'decrease/trained','noul_only':root/'noul_only/trained'}
    original = read_json(root/'reference/protocol.json')
    admission = read_json(root/'reference/admission.json')
    parent = read_json(root/'reference/training_environment.json')
    for arm in ('decrease','noul_only'):
        cfg=read_json(root/arm/'protocol.json')
        for key in original:
            if key in ('loss','rank_weight','code_sha256'): continue
            if cfg[key]!=original[key]: raise ValueError(f'Unmatched protocol: {arm}/{key}')
        if read_json(root/arm/'admission.json')!=admission: raise ValueError('Admission differs')
        env=read_json(root/arm/'training_environment.json')
        if env!=parent: raise ValueError('Training initialization or environment differs')
    rows={a:{r['key']:r for r in read_jsonl(p/'records.jsonl')} for a,p in locations.items()}
    planned={r['key'] for r in read_json(root/'decrease/atbench_cases.json')}
    if any(set(v)!=planned for v in rows.values()): raise ValueError('Incomplete evaluation')
    envs=[read_json(p/'environment.json') for p in locations.values()]
    if any(e['environment']!=envs[0]['environment'] or e['temperature']!=envs[0]['temperature'] for e in envs):
        raise ValueError('Evaluation environment mismatch')
    common=sorted(k for k in planned if all(v[k]['status']=='complete' for v in rows.values()))
    for k in common:
        ref=rows['baseline'][k]
        for v in rows.values():
            if v[k]['label']!=ref['label'] or [s['input_sha256'] for s in v[k]['steps']]!=[s['input_sha256'] for s in ref['steps']]:
                raise ValueError('Evaluation input mismatch')
    y=[rows['baseline'][k]['label'] for k in common]
    result={'common_count':len(common),'labels':dict(Counter(y)),
        'arms':{a:evaluate(y,[v[k]['score'] for k in common]) for a,v in rows.items()},
        'selected_epochs':{'increase':read_json(root/'reference/selected_checkpoint.json'),
                          **{a:read_json(root/a/'selected_checkpoint.json') for a in ('decrease','noul_only')}},
        'exclusions':{a:[r for r in v.values() if r['status']!='complete'] for a,v in rows.items()},
        'limitations':['Single seed, exploratory previously inspected ATBench; no significance or replication claim.',
                      'Original increasing arm reused; new arms trained in separate launches.',
                      'NOUL-only retains unused predecessor forward calls to control dropout RNG, not an optimized throughput baseline.']}
    write_json(root/'comparison.json',result)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=('prepare','compare'))
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--reference',type=Path)
    args=p.parse_args()
    if args.phase=='prepare': prepare(args.root,args.reference)
    else: compare(args.root)


if __name__=='__main__': main()
