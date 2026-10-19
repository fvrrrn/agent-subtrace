"""CPU-only paired factorial report; requires all eight completed cells."""
import argparse
import json
from pathlib import Path
from tqdm import tqdm
from prepare_whole_dialogs import write


def main(root):
    read = lambda p: json.loads(p.read_text())
    lock = read(root/'authorization.json')
    results = {}; reference = None
    for adaptation,curve,weighting in tqdm(lock['cells'],desc='Paired factorial report'):
        name = '_'.join((adaptation,curve,weighting)); arm = root/name
        assert read(arm/'status.json')['status'] == 'complete'
        selection = read(arm/'selection.json'); epoch = selection['epoch']
        events = [json.loads(line) for line in (arm/f'dev_epoch{epoch}/predictions.jsonl').read_text().splitlines()]
        identity = sorted((e['key'],e['dialog'],e['label'],e['ids_sha256'],e['state_sha256']) for e in events)
        if reference is None:
            reference = identity
        assert identity == reference and len(identity) == len(set(identity)) == 886
        results[name] = {'epoch':epoch, **selection['metrics']}
    contrasts = {}
    for a,c,w in lock['cells']:
        for factor, other, label in [('adaptation',('lora',c,w),'lora-minus-frozen'),
                                     ('curve',(a,'broad',w),'broad-minus-step'),
                                     ('weighting',(a,c,'adaptive'),'adaptive-minus-uniform')]:
            if (factor=='adaptation' and a!='frozen') or (factor=='curve' and c!='step') or (factor=='weighting' and w!='uniform'):
                continue
            before = '_'.join((a,c,w)); after = '_'.join(other)
            contrasts[f'{label}: {after} - {before}'] = {m:results[after][m]-results[before][m]
                for m in ('average_precision','pr_auc_trapezoid','roc_auc','event_bce')}
    write(root/'comparison.json',{'paired_actions':len(reference),'results':results,'contrasts':contrasts,
        'interpretation':'One seed, dev-selected shared weighted-broad warmstart; continuation effects only.'})
    lines = ['# Continuation factorial','', '886 paired actions; one seed, dev-selected common warm-start.','',
        '| Cell | Epoch | AP | PR-AUC (trap.) | ROC-AUC | P | R | F1 | Acc | FP | FN | BCE |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name,r in results.items():
        o = r['operating_point']
        vals = [r['average_precision'],r['pr_auc_trapezoid'],r['roc_auc'],o['precision'],o['recall'],o['f1'],o['accuracy']]
        lines.append(f'| {name} | {r["epoch"]} | '+' | '.join('undefined' if v is None else f'{v:.5f}' for v in vals)+f' | {o["fp"]} | {o["fn"]} | {r["event_bce"]:.5f} |')
    lines += ['', 'Fixed threshold .5. Complete paired factorial contrasts: `comparison.json`.']
    (root/'COMPARISON.md').write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('root',type=Path)
    main(parser.parse_args().root)
