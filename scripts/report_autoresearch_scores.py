"""Apply a dev-select-only operating threshold to fixed evaluation event outputs."""
import argparse
import json
from pathlib import Path
import time

from graph_safety.metrics import evaluate
from prepare_whole_dialogs import write
from train_whole_dialogs import file_hash


def load(path):
    rows=[json.loads(line) for line in path.read_text().splitlines()]
    result={}
    for row in rows:
        span=row.get('span',row)
        key=span['key']
        assert key not in result
        result[key]={'label':span['label'],'score':row['score'],
                     'dialog':row.get('dialog'),'ids_sha256':row['ids_sha256'],
                     'state_sha256':span['state_sha256']}
    return result


def summary(rows,threshold):
    m=evaluate([v['label'] for v in rows.values()],[v['score'] for v in rows.values()],threshold)
    curve=m.pop('pr_curve')
    for fpr in (.01,.05):
        m[f'recall_at_fpr_{fpr}']=max((p['recall'] for p in curve if p['recall'] is not None
            and p.get('specificity') is not None and 1-p['specificity']<=fpr+1e-12),default=0.)
    return m,curve


def main(args):
    assert time.time()+60<args.deadline_unix, 'Insufficient reporting budget'
    select=load(args.select)
    baseline,curve=summary(select,.5)
    if args.locked_threshold is not None:
        assert file_hash(args.locked_threshold)==args.expected_threshold_sha256
        record=json.loads(args.locked_threshold.read_text())
        assert record['split']=='dev_select' and record['events_sha256']==file_hash(args.select)
        assert Path(record['events_path'])==args.select
        best=record['fitted_point']
        assert best['threshold']==record['threshold']
    else:
        candidates=[p for p in curve if p.get('threshold') is not None and p['recall'] is not None]
        best=max(candidates,key=lambda p:(min(p['precision'],p['recall']),p['f1'],p['recall'],p['threshold']))
    threshold=best['threshold']
    result={'criterion':'dev_select max min(P,R),ties F1/recall/higher threshold; all ties retained',
            'threshold':threshold,'selection_source':str(args.select),'selection_sha256':file_hash(args.select),
            'dev_select_at_0_5':baseline,'dev_select_fitted':summary(select,threshold)[0],
            'warning':'Dev-select fit is optimistic; evaluation thresholds never fitted to evaluation labels.',
            'evaluations':{}}
    for path in args.evaluation:
        rows=load(path)
        result['evaluations'][str(path)]={'sha256':file_hash(path),'at_0_5':summary(rows,.5)[0],
                                        'at_locked_dev_threshold':summary(rows,threshold)[0]}
    write(args.output,result)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--select',type=Path,required=True)
    parser.add_argument('--evaluation',type=Path,nargs='*',default=[])
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--locked-threshold',type=Path)
    parser.add_argument('--expected-threshold-sha256')
    parser.add_argument('--deadline-unix',type=float,default=float('inf'))
    main(parser.parse_args())
