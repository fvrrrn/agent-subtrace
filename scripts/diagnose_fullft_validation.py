"""CPU-only, immutable matched dev-select diagnosis; no inference or threshold fit."""
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np
import torch


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metrics(rows):
    y = np.array([r['label'] for r in rows])
    s = np.array([r['score'] for r in rows])
    p, n = int(y.sum()), int((1-y).sum())
    order = np.argsort(-s, kind='stable')
    ys, ss = y[order], s[order]
    ends = np.r_[np.flatnonzero(np.diff(ss)), len(ss)-1]
    tp_curve = np.r_[0, np.cumsum(ys)[ends]]
    fp_curve = np.r_[0, np.cumsum(1-ys)[ends]]
    recall = tp_curve / p if p else np.zeros(len(tp_curve))
    precision = np.r_[1., tp_curve[1:] / (tp_curve[1:]+fp_curve[1:])]
    fpr = fp_curve / n if n else np.zeros(len(fp_curve))
    tp, fp = int(((s >= .5) & (y == 1)).sum()), int(((s >= .5) & (y == 0)).sum())
    fn, tn = p-tp, n-fp
    return dict(n=len(y), unsafe=p, AP=float(np.sum(np.diff(recall)*precision[1:])) if p else None,
                PR_AUC_trapezoid=float(np.trapezoid(precision, recall)) if p else None,
                ROC_AUC=float(np.trapezoid(recall, fpr)) if p and n else None,
                BCE=float(np.mean(-(y*np.log(np.clip(s,1e-15,1-1e-15))+(1-y)*np.log(np.clip(1-s,1e-15,1-1e-15))))),
                tp=tp, fp=fp, fn=fn, tn=tn, precision=tp/(tp+fp) if tp+fp else 0.,
                recall=tp/p if p else None, FPR=fp/n if n else None,
                F1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0., accuracy=(tp+tn)/len(y),
                recall_at_FPR_01=float(recall[fpr <= .01].max()) if p and n else None,
                recall_at_FPR_05=float(recall[fpr <= .05].max()) if p and n else None,
                class_score_quantiles={str(c): dict(zip(['min','q10','q25','median','q75','q90','max'],
                    np.quantile(s[y==c],[0,.1,.25,.5,.75,.9,1]).tolist())) if (y==c).any() else None for c in [0,1]})


def main(args):
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=False)
    inputs = {}
    def load(path):
        inputs[str(path)] = sha(path)
        return json.loads(path.read_text())
    root = args.root / 'runs'
    base = root / 'autoresearch-sixhour-v1'
    manifest = load(base/'data/dev_select.manifest.json')
    metadata = load(base/'data/action_metadata_audit.json')
    dialogs = load(base/'data/dev_select.json')
    assert len(dialogs) == 301
    manifests = {r['key']: r for r in manifest['rows']}
    canonical = {}
    for d in dialogs:
        assert d['ids_sha256'] == manifests[d['key']]['ids_sha256']
        assert [s['key'] for s in d['spans']] == manifests[d['key']]['action_keys']
        for i, span in enumerate(d['spans']):
            assert span['key'] not in canonical
            assert span['state_sha256'] == metadata[span['key']]['state_sha256']
            canonical[span['key']] = dict(span, dialog=d['key'], ids_sha256=d['ids_sha256'],
                dialog_tokens=d['tokens'], action_position=i+1, number_actions=len(d['spans']))
    assert len(canonical) == 886 and sum(r['label'] for r in canonical.values()) == 147
    fits = {'frozen_A': root/'sound-anomaly-4b-control-v1/candidates/control_a/fit',
            'frozen_original': base/'round1v2/broad_endpoint/fit'}
    paths = {'fullft': base/'fullft_small/dev_epoch1/predictions.jsonl',
             'lora': base/'lora_screen_best/dev_epoch1/predictions.jsonl'}
    selections = {}
    for name, fit in fits.items():
        selections[name] = load(fit/'selection.json')
        paths[name] = fit/f"dev_select_epoch{selections[name]['epoch']}"/'events.jsonl'
    models = {}
    checks = {}
    for name, path in paths.items():
        inputs[str(path)] = sha(path)
        rows = {}
        sigmoid_error = 0.
        for line in path.read_text().splitlines():
            raw = json.loads(line)
            span = raw.get('span', raw)
            key = span['key']
            assert key not in rows
            c = canonical[key]
            for field in ['label','state_sha256']:
                assert span[field] == c[field], (name,key,field)
            for field in ['dialog','ids_sha256']:
                assert raw[field] == c[field], (name,key,field)
            assert np.isfinite(raw['event_logit']) and 0 <= raw['score'] <= 1
            sigmoid_error = max(sigmoid_error, abs(float(torch.sigmoid(torch.tensor(raw['event_logit']))) - raw['score']))
            rows[key] = dict(c, score=raw['score'], event_logit=raw['event_logit'])
        assert rows.keys() == canonical.keys()
        models[name] = rows
        checks[name] = {'events': len(rows), 'max_sigmoid_error': sigmoid_error}
    keys = sorted(canonical)
    # Deterministic cases: first key within label x source-signature strata, never by score/error.
    strata = defaultdict(list)
    for k in keys:
        strata[(canonical[k]['label'], tuple(sorted(metadata[k]['sources'])))].append(k)
    sample_keys = sorted(v[0] for v in strata.values())
    curves = []
    for name, path in paths.items():
        endpoint_errors, score_errors, count = [], [], 0
        for d in dialogs:
            token_path = path.parent/'tokens'/f"{d['key']}.pt"
            inputs[str(token_path)] = sha(token_path)
            t = torch.load(token_path, map_location='cpu', weights_only=False)
            assert t['ids_sha256'] == d['ids_sha256'] and t['spans'] == d['spans']
            if 'ids' in t:
                assert t['ids'] == d['ids']
            logits = t.get('raw_logits', t.get('raw_token_logits')).float()
            assert len(logits) == d['tokens'] and bool(torch.isfinite(logits).all())
            for span in d['spans']:
                k = span['key']; z = logits[span['end']-1].item()
                endpoint_errors.append(abs(z-models[name][k]['event_logit']))
                score_errors.append(abs(float(torch.sigmoid(torch.tensor(z)))-models[name][k]['score']))
                count += 1
                if k in sample_keys:
                    a = logits[span['start']:span['end']]
                    curves.append(dict(model=name,key=k,label=span['label'],sources=metadata[k]['sources'],
                        tokens=len(a), endpoint_logit=z, mean_logit=float(a.mean()), max_logit=float(a.max()),
                        endpoint_minus_mean=z-float(a.mean()),
                        decile_mean_logits=[float(v.mean()) for v in torch.tensor_split(a,min(10,len(a)))],
                        raw_action_logits=a.tolist()))
        checks[name].update(token_endpoints_checked=count,max_token_endpoint_logit_error=max(endpoint_errors),
                            max_token_endpoint_score_error=max(score_errors))
    def bin_name(value, boundaries):
        lower = 0
        for upper in boundaries:
            if value <= upper:
                return f'{lower+1}-{upper}'
            lower = upper
        return f'>{lower}'
    groups = defaultdict(lambda: defaultdict(list))
    for k,c in canonical.items():
        for source in metadata[k]['sources'] or ['__unattributed__']:
            groups['source'][source].append(k)
        for domain in metadata[k]['domains'] or ['__unattributed__']:
            groups['domain'][domain].append(k)
        for category, value, bounds in [('dialog_tokens',c['dialog_tokens'],[1024,2048,4096,8192]),
                ('action_tokens',c['end']-c['start'],[64,128,256,512]),
                ('action_position',c['action_position'],[1,2,4]),('number_actions',c['number_actions'],[1,2,4])]:
            groups[category][bin_name(value,bounds)].append(k)
    aggregate = {n: metrics(list(rows.values())) for n,rows in models.items()}
    slices = {category: {group: {'keys':sorted(ks),'models':{n:metrics([rows[k] for k in ks])
                for n,rows in models.items()}} for group,ks in gs.items()} for category,gs in groups.items()}
    paired = {}
    for name in ['lora','frozen_A','frozen_original']:
        changes = []
        for k in keys:
            a,b = models['fullft'][k],models[name][k]
            changes.append(dict(key=k,label=a['label'],fullft_score=a['score'],comparison_score=b['score'],
                delta_score=a['score']-b['score'],fullft_positive=a['score']>=.5,comparison_positive=b['score']>=.5,
                fullft_correct=(a['score']>=.5)==a['label'],comparison_correct=(b['score']>=.5)==a['label']))
        paired[name] = {'all_event_pairs':changes,'changed_decisions':sum(r['fullft_positive'] != r['comparison_positive'] for r in changes),
            'fullft_only_correct':sum(r['fullft_correct'] and not r['comparison_correct'] for r in changes),
            'comparison_only_correct':sum(r['comparison_correct'] and not r['fullft_correct'] for r in changes),
            'class_mean_score_delta':{str(y):float(np.mean([r['delta_score'] for r in changes if r['label']==y])) for y in [0,1]}}
    histories = {n:load(p/'history.json') for n,p in {'fullft':base/'fullft_small','lora':base/'lora_screen_best',**fits}.items()}
    report = dict(aggregate=aggregate,checks=checks,slices=slices,paired=paired,histories=histories,
        selected_epochs={n:s['epoch'] for n,s in selections.items()}, exact_keys=keys,
        sample_keys=sample_keys, sample_rule='Lexicographically first key per label x original-source-signature stratum.',
        threshold=.5, threshold_optimization=False,
        caveats=['Source/domain slices overlap; do not sum them. Small slices are descriptive, without uncertainty intervals.',
            'Single stopped FullFT epoch versus selected checkpoints; unequal training budget and selection opportunity.',
            'Train losses are online training averages; dev losses use a final checkpoint and may have different weighting.',
            'Train/dev gaps are compatible with generalization failure, but do not establish overfitting or its cause.',
            'Token curves are artificial-target scores, not localized ground-truth evidence or calibrated risk.',
            'No confirm/external data, model inference, GPU work, or threshold optimization. Only threshold 0.5 is used; no final locked thresholds supplied.',
            'FullFT/LoRA token files store input hashes rather than literal IDs; literal token equality verified for frozen files, hash equality for all.',
            'Stored tensors/outputs can establish readout consistency but cannot independently establish correct training or a causal explanation.'])
    inputs[str(Path(__file__).resolve())] = sha(Path(__file__))
    for filename,obj in [('diagnostics.json',report),('token_curves.json',curves),('input_hashes.json',inputs)]:
        with (args.output/filename).open('x') as stream:
            json.dump(obj,stream,ensure_ascii=False,indent=2,allow_nan=False)
            stream.write('\n')
    print(json.dumps({'aggregate':aggregate,'checks':checks,'sample_keys':len(sample_keys)},indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('/var/local/danefedov/experiments/agent-safety'))
    parser.add_argument('--output',type=Path,required=True)
    main(parser.parse_args())
