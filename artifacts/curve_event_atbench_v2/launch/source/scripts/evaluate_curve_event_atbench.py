"""Fixed validation-selected curve/event heads on the existing ATBench causal replay."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import torch
from tqdm.auto import tqdm

from prepare_whole_dialogs import REVISION, digest, write
from train_curve_event_heads import Head, ARMS, event_logit
from graph_safety.metrics import evaluate


def read(p):
    return json.loads(p.read_text())


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def score(args):
    protocol = read(args.root / 'protocol.json')
    parent = Path(protocol['parent'])
    suite = parent / 'atbench'
    assert sha(suite / 'cases.json') == protocol['cases_sha256']
    spans_path = Path(protocol['spans'])
    assert sha(spans_path) == protocol['spans_sha256']
    spans = read(spans_path)
    locked = args.root / 'locked' / args.arm
    assert sha(locked / 'selected_head.pt') == protocol['heads'][args.arm]['head_sha256']
    head = Head(args.arm).cuda()
    head.load_state_dict(torch.load(locked / 'selected_head.pt', weights_only=True, map_location='cuda'))
    head.eval().requires_grad_(False)
    out = args.root / args.arm
    out.mkdir(exist_ok=False)
    (out / 'tokens').mkdir()
    count = 0
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    try:
        with (out / 'predictions.jsonl').open('w') as stream, torch.inference_mode():
            for case in tqdm(sorted(read(suite / 'cases.json'), key=lambda c: (-c['tokens'], c['key'])),
                             desc=args.arm, unit='case', mininterval=3):
                results, tensors = {}, []
                for seq in case['sequences']:
                    cache = suite / 'features' / (digest(seq['key'])+'.pt')
                    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
                        saved = torch.load(cache, weights_only=True, map_location='cpu')
                    identity = saved['identity']
                    assert identity['ids_sha256'] == seq['ids_sha256'] == digest(seq['ids'])
                    assert identity['revision'] == REVISION
                    assert identity['torch'] == str(torch.__version__)
                    assert identity['template_sha256'] == protocol['template_sha256']
                    assert identity['weights_sha256'] == protocol['frozen_weights_sha256']
                    hidden = saved['hidden'].cuda()
                    del saved
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        raw, null = head(hidden)
                    z = raw[0, :seq['tokens']].float()
                    n = null[0, :seq['tokens']].float() if null is not None else None
                    assert torch.isfinite(z).all() and (n is None or torch.isfinite(n).all())
                    for index, endpoint in zip(seq['action_indices'], seq['endpoints']):
                        a = case['actions'][index]
                        span = spans[case['key']][index]
                        assert a['key'] == span['key'] and a['state_sha256'] == span['state_sha256']
                        assert span['end'] == endpoint+1
                        assert digest(seq['ids'][:span['end']]) == span['prefix_ids_sha256']
                        e = event_logit(z, n, span, args.arm).item()
                        p = z[span['start']:span['end']].sigmoid()
                        assert math.isfinite(e)
                        results[index] = {'key': a['key'], 'kind': a['kind'], 'state_sha256': a['state_sha256'],
                            'score': torch.tensor(e).sigmoid().item(), 'event_logit': e,
                            'endpoint': p[-1].item(), 'token_max': p.max().item(),
                            'action_tokens': len(p), 'null_logit': n[endpoint].item() if n is not None else None}
                    tensors.append({'key': seq['key'], 'ids': torch.tensor(seq['ids'], dtype=torch.int32),
                        'ids_sha256': seq['ids_sha256'], 'raw_logits': z.cpu(),
                        'null_logits': n.cpu() if n is not None else None,
                        'action_spans': [spans[case['key']][i] for i in seq['action_indices']]})
                    del hidden, raw, null, z, n, p
                assert sorted(results) == list(range(len(case['actions'])))
                torch.save({'key': case['key'], 'sequences': tensors}, out / 'tokens' / (digest(case['key'])+'.pt'))
                stream.write(json.dumps({'key': case['key'], 'label': case['label'],
                    'ids_sha256': case['ids_sha256'], 'actions': [results[i] for i in sorted(results)]})+'\n')
                stream.flush()
                count += 1
    except Exception as error:
        write(out / 'failure.json', {'error': repr(error), 'completed_cases': count, 'no_retries': True})
        raise
    assert count == 981
    write(out / 'complete.json', {'cases': count, 'seconds': time.monotonic()-started,
        'peak_cuda_bytes': torch.cuda.max_memory_allocated(), 'torch': str(torch.__version__),
        'cuda': torch.version.cuda, 'device': torch.cuda.get_device_name(), 'qwen_loaded': False,
        'no_truncation': True, 'all_token_logits_saved': True})


def summarize(labels, scores):
    m = evaluate(labels, scores)
    m.pop('pr_curve')
    dist = {}
    for y in (0, 1):
        x = torch.tensor([s for a, s in zip(labels, scores) if a == y], dtype=torch.float64)
        dist[str(y)] = {'n': len(x), 'mean': x.mean().item(),
            'quantiles': dict(zip(('min', 'p10', 'p25', 'median', 'p75', 'p90', 'p99', 'max'),
                torch.quantile(x, x.new_tensor([0, .1, .25, .5, .75, .9, .99, 1])).tolist())),
            'fraction_ge_0.5': (x >= .5).double().mean().item()}
    return {'metrics': m, 'score_distribution_by_trajectory_gold': dist}


def analyze(args):
    protocol = read(args.root / 'protocol.json')
    outputs = {}
    for arm in ARMS:
        assert read(args.root / arm / 'complete.json')['cases'] == 981
        outputs[arm] = {r['key']: r for r in map(json.loads, (args.root / arm / 'predictions.jsonl').read_text().splitlines())}
    reference = Path(protocol['reference'])
    assert sha(reference) == protocol['reference_sha256']
    old = {r['key']: r for r in map(json.loads, reference.read_text().splitlines())}
    outputs['previous_frozen_endpoint'] = old
    keys = sorted(old)
    assert len(keys) == 981 and all(set(v) == set(keys) for v in outputs.values())
    for k in keys:
        assert len({(v[k]['label'], v[k]['ids_sha256']) for v in outputs.values()}) == 1
        assert all([(a['key'], a['state_sha256']) for a in v[k]['actions']] ==
                   [(a['key'], a['state_sha256']) for a in old[k]['actions']] for v in outputs.values())
    result = {}
    for mode in ('all_events', 'tool_only'):
        selected = [k for k in keys if mode == 'all_events' or any(a['kind'] == 'tool_call' for a in old[k]['actions'])]
        result[mode] = {arm: summarize([v[k]['label'] for k in selected],
            [max(a['score'] for a in v[k]['actions'] if mode == 'all_events' or a['kind'] == 'tool_call') for k in selected])
            for arm, v in outputs.items()}
    write(args.root / 'comparison.json', {'common_cases': len(keys), 'results': result,
        'limitations': 'Previously viewed ATBench981 causal replay, not official full1000 evaluator. Trajectory gold only. Scores uncalibrated. Fixed validation-selected checkpoints, no threshold fitting. Different curve/readout/training duration: not isolated event-loss ablation.'})
    write(args.root / 'status.json', {'status': 'complete', 'cases': len(keys), 'arms': list(ARMS)})
    print(json.dumps(result['all_events'], indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('score', 'analyze'))
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--arm', choices=ARMS)
    args = parser.parse_args()
    {'score': score, 'analyze': analyze}[args.phase](args)
