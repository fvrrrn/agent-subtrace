"""User-requested early test peek at one already selected, frozen seed."""
import argparse
import gc
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from graph_safety.controlled_training import compact, model_environment, predict
from graph_safety.io import read_json, sha256, write_json
from graph_safety.step_training import load


def compare(output, split, predictions, records):
    indexed={arm:{r['key']:r for r in rows} for arm,rows in predictions.items()}
    planned={r['key'] for r in records.values() if r['split']==split}
    assert all(set(rows)==planned for rows in indexed.values())
    common=sorted(k for k in planned if all(rows[k]['status']=='complete' for rows in indexed.values()))
    result={'split':split,'planned_n':len(planned),'common_n':len(common),'models':{},
            'note':'Early user-requested single-seed peek. No test-driven checkpoint, threshold, data or hyperparameter selection.'}
    for arm,rows in indexed.items():
        for k in common:
            assert rows[k]['input_sha256']==indexed['baseline'][k]['input_sha256']
            assert rows[k]['label']==indexed['baseline'][k]['label']
        model={'all_actions':compact([rows[k] for k in common]),
               'rejections':[r for r in rows.values() if r['status']!='complete']}
        if split=='test':
            decisions=[k for k in common if json.loads(records[k]['state'])['proposed_action']['name'] in ('deliver','charge')]
            model['transmit_and_spend_only']=compact([rows[k] for k in decisions])
            twins=defaultdict(list)
            for k in decisions:
                r=records[k];twins[(r['group'],r['position'],r['action_sha256'])].append(k)
            pairs=[v for v in twins.values() if len(v)==2 and {rows[k]['label'] for k in v}=={0,1}]
            differences=[rows[next(k for k in pair if rows[k]['label'])]['score']-
                         rows[next(k for k in pair if not rows[k]['label'])]['score'] for pair in pairs]
            model['counterfactuals']={'n':len(differences),'correct_direction':sum(x>0 for x in differences),
                                     'ties':sum(x==0 for x in differences)}
        result['models'][arm]=model
    write_json(output/f'{split}_comparison.json',result)
    print(json.dumps({'split':split,'common_n':len(common),
        'metrics':{a:{m:r['all_actions'][m] for m in ('average_precision','roc_auc')}
                   for a,r in result['models'].items()}),flush=True)


def main():
    import torch
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seed',type=int,required=True)
    args=parser.parse_args();root=args.root;output=args.output
    output.mkdir(exist_ok=False)
    cfg=read_json(root/'protocol.json')
    assert sha256(root/'records.json')==cfg['records_sha256']
    assert read_json(root/'pilot_gate.json')['code_sha256']=={
        p.name:sha256(p) for p in (root.parent/'src/graph_safety').glob('*.py')}
    rows=read_json(root/'records.json');records={r['key']:r for r in rows}
    checkpoints={arm:root/'full'/f'{arm}_{args.seed}'/'checkpoint' for arm in cfg['arms']}
    frozen={arm:{'selection':read_json(p.parent/'checkpoint_audit.json'),
                'adapter':sha256(p/'adapter_model.safetensors'),'head':sha256(p/'head.pt')}
            for arm,p in checkpoints.items()}
    assert all(read_json(p.parent/'reload_audit.json')['passed'] for p in checkpoints.values())
    write_json(output/'snapshot_audit.json',{'seed':args.seed,'records_sha256':cfg['records_sha256'],
        'frozen_checkpoints':frozen,'opened_at':datetime.now(timezone.utc).isoformat(),
        'deviation':'User explicitly requested first-seed test measurement before all seeds finish. This is an early test peek; no checkpoint, data or hyperparameter changes follow from it.'})
    write_json(output/'status.json',{'phase':'running','seed':args.seed})
    for split in ('test','external_test'):
        predictions={};reference=None
        for arm,spec in {'baseline':cfg['checkpoint'],**checkpoints}.items():
            cp,tok,model=load(spec);environment=model_environment(cp,model)
            signature={k:v for k,v in environment.items() if k not in ('initial_weights_sha256','initial_head_sha256')}
            if reference is None:reference=signature
            assert signature==reference
            folder=output/arm
            write_json(folder/f'{split}_environment.json',environment)
            # Preserve the same predictor, prompt, limits, temperature and native readout.
            predictions[arm],_=predict(model,tok,[r for r in rows if r['split']==split],cfg,folder,split)
            del model;gc.collect();torch.cuda.empty_cache()
        compare(output,split,predictions,records)
    after={arm:{'adapter':sha256(p/'adapter_model.safetensors'),'head':sha256(p/'head.pt')}
           for arm,p in checkpoints.items()}
    assert all(after[a]=={k:frozen[a][k] for k in ('adapter','head')} for a in after)
    write_json(output/'checkpoint_unchanged_after_test.json',after)
    write_json(output/'status.json',{'phase':'complete','seed':args.seed})


if __name__=='__main__':main()
