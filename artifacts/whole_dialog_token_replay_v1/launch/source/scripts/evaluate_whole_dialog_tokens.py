"""Token readout replay of fixed whole-dialog heads, with frozen and LoRA queues."""
import argparse
from collections import Counter
import gc
import json
import math
from pathlib import Path
import time

import evaluate_whole_dialog_benchmarks as original
from prepare_whole_dialogs import BASE, REVISION, digest, write

ARMS = ('frozen_linear', 'frozen_linear_mono', 'lora_linear', 'lora_linear_mono')
METHODS = ('endpoint', 'token_max', 'top10_mean', 'tail8_mean')


def read(p):
    return json.loads(p.read_text())


def prepare(root, parent):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(BASE, revision=REVISION, local_files_only=True)
    root.mkdir(parents=True, exist_ok=False)
    write(root / 'parent.json', {'root': str(parent), 'snapshot_sha256': original.sha(parent / 'snapshot_audit.json')})
    audit = {}
    for suite in original.SUITES:
        folder = root / suite
        folder.mkdir()
        assert original.sha(parent / suite / 'cases.json') == read(parent / suite / 'audit.json')['cases_sha256']
        cases = read(parent / suite / 'cases.json')
        spans = {}
        lengths = []
        sequences = 0
        for case in cases:
            actions = []
            for action in case['actions']:
                text = action['native_text']
                marker = '<|im_start|>assistant\n'
                start = text.rfind(marker) + len(marker)
                assert start >= len(marker)
                empty_think = '<think>\n\n</think>\n\n'
                if text[start:].startswith(empty_think):
                    start += len(empty_think)
                encoded = tok(text, add_special_tokens=False, return_offsets_mapping=True)
                positions = [i for i, (a, b) in enumerate(encoded['offset_mapping']) if b > a and a >= start and b <= len(text)]
                assert positions and positions == list(range(positions[0], positions[-1]+1))
                assert positions[-1] == len(encoded['input_ids'])-1
                actions.append({'key': action['key'], 'start': positions[0], 'end': positions[-1]+1,
                    'action_tokens': len(positions), 'prefix_ids_sha256': digest(encoded['input_ids']),
                    'state_sha256': action['state_sha256']})
                lengths.append(len(positions))
            for seq in case['sequences']:
                sequences += 1
                cache = parent / suite / 'features' / (digest(seq['key']) + '.pt')
                assert cache.is_file(), f'Frozen feature cache missing: {cache}'
                for index, endpoint in zip(seq['action_indices'], seq['endpoints']):
                    span = actions[index]
                    assert span['end'] == endpoint+1
                    assert digest(seq['ids'][:span['end']]) == span['prefix_ids_sha256']
                    # The source endpoint excludes im_end/role-closing tokens;
                    # retain the same native action body as the training mask.
                    assert tok.convert_tokens_to_ids('<|im_end|>') not in seq['ids'][span['start']:span['end']]
            spans[case['key']] = actions
        write(folder / 'spans.json', spans)
        audit[suite] = {'cases': len(cases), 'actions': len(lengths), 'sequences': sequences,
            'action_tokens': sum(lengths), 'min_action_tokens': min(lengths), 'max_action_tokens': max(lengths),
            'source_cases_sha256': original.sha(parent / suite / 'cases.json'),
            'spans_sha256': original.sha(folder / 'spans.json')}
    write(root / 'input_gate.json', {'status': 'passed', 'suites': audit,
        'action_mask': 'Native assistant body after role prefix and empty-think prefix, before im_end; same mask convention as training. Native tool-call serialization retained.',
        'no_truncation': True, 'no_new_cases_or_labels': True, 'frozen_cache_complete': True})
    print(json.dumps(audit, indent=2), flush=True)


def score(root, arm):
    import torch
    from tqdm.auto import tqdm
    from train_whole_dialogs import Head, encode
    parent = Path(read(root / 'parent.json')['root'])
    assert original.sha(parent / 'snapshot_audit.json') == read(root / 'parent.json')['snapshot_sha256']
    assert read(root / 'input_gate.json')['status'] == 'passed'
    model = None
    if arm.startswith('frozen'):
        locked = parent / 'locked' / arm
        for name, expected in read(parent / 'snapshot_audit.json')[arm]['hashes'].items():
            assert original.sha(locked / name) == expected
        head = Head().cuda()
        head.load_state_dict(torch.load(locked / 'selected_head.pt', weights_only=True, map_location='cuda'))
        head.eval().requires_grad_(False)
    else:
        model, head = original.load_head(parent, arm)
    for suite in original.SUITES:
        folder = root / suite
        gate = read(root / 'input_gate.json')['suites'][suite]
        assert original.sha(parent / suite / 'cases.json') == gate['source_cases_sha256']
        assert original.sha(folder / 'spans.json') == gate['spans_sha256']
        spans = read(folder / 'spans.json')
        reference = {r['key']: r for r in map(json.loads, (parent / suite / arm / 'predictions.jsonl').read_text().splitlines())}
        out = folder / arm
        out.mkdir(exist_ok=False)
        (out / 'tokens').mkdir()
        started = time.monotonic()
        max_error = 0.
        max_raw_error = 0.
        count = 0
        try:
            with (out / 'predictions.jsonl').open('w') as stream, torch.inference_mode():
                cases = sorted(read(parent / suite / 'cases.json'), key=lambda c: (-c['tokens'], c['key']))
                for case in tqdm(cases, desc=f'tokens {arm}/{suite}', unit='case', mininterval=3):
                    action_results, token_sequences = [], []
                    for seq in case['sequences']:
                        if model is None:
                            cache = parent / suite / 'features' / (digest(seq['key']) + '.pt')
                            saved = torch.load(cache, weights_only=True, map_location='cpu')
                            identity = saved['identity']
                            assert identity['ids_sha256'] == seq['ids_sha256']
                            assert identity['revision'] == REVISION and identity['torch'] == str(torch.__version__)
                            assert identity['template_sha256'] == read(parent / suite / 'audit.json')['template_sha256']
                            assert identity['weights_sha256'] == read(parent / suite / arm / 'complete.json')['frozen_weights_sha256']
                            hidden = saved['hidden'].cuda()
                            del saved
                        else:
                            hidden = encode(model, seq)
                        with torch.autocast('cuda', dtype=torch.bfloat16):
                            logits = head(hidden)[0, :seq['tokens']].float()
                        assert torch.isfinite(logits).all()
                        for index in seq['action_indices']:
                            action = case['actions'][index]
                            span = spans[case['key']][index]
                            assert span['key'] == action['key']
                            z = logits[span['start']:span['end']]
                            p = z.sigmoid()
                            methods = {'endpoint': p[-1].item(), 'token_max': p.max().item(),
                                'top10_mean': p.topk(max(1, math.ceil(.1*len(p)))).values.mean().item(),
                                'tail8_mean': p[-8:].mean().item()}
                            old = next(a for a in reference[case['key']]['actions'] if a['key'] == action['key'])
                            error = abs(methods['endpoint']-old['score'])
                            raw_error = abs(z[-1].item()-old['raw_logodds'])
                            max_error = max(max_error, error)
                            max_raw_error = max(max_raw_error, raw_error)
                            assert error <= 1e-6 and raw_error <= 1e-5, f'Endpoint drift: {arm}/{action["key"]}: {error}/{raw_error}'
                            action_results.append({'key': action['key'], 'kind': action['kind'],
                                'state_sha256': action['state_sha256'], 'methods': methods,
                                'action_tokens': len(p), 'peak_offset': int(p.argmax().item()),
                                'peak_minus_endpoint': methods['token_max']-methods['endpoint']})
                        token_sequences.append({'key': seq['key'], 'ids': torch.tensor(seq['ids'], dtype=torch.int32),
                            'ids_sha256': seq['ids_sha256'], 'raw_logits': logits.cpu().clone(),
                            'action_spans': [spans[case['key']][i] for i in seq['action_indices']]})
                        del hidden, logits, z, p
                    assert [a['key'] for a in action_results] == [a['key'] for a in case['actions']]
                    token_file = digest(case['key']) + '.pt'
                    torch.save({'case_key': case['key'], 'sequences': token_sequences}, out / 'tokens' / token_file)
                    stream.write(json.dumps({'key': case['key'], 'label': case['label'], 'ids_sha256': case['ids_sha256'],
                        'token_file': token_file, 'actions': action_results}) + '\n')
                    stream.flush()
                    count += 1
        except Exception as error:
            write(out / 'failure.json', {'error': type(error).__name__ + ': ' + str(error),
                'completed_cases': count, 'stopped_without_retries': True})
            raise
        write(out / 'complete.json', {'cases': count, 'max_endpoint_probability_error': max_error,
            'max_endpoint_raw_error': max_raw_error, 'seconds': time.monotonic()-started,
            'torch': str(torch.__version__), 'cuda': torch.version.cuda, 'device': torch.cuda.get_device_name(),
            'base_revision': REVISION, 'snapshot': read(parent / 'snapshot_audit.json')[arm],
            'frozen_head_only': model is None, 'all_sequence_token_logits_saved': True})
    del model, head
    gc.collect()
    torch.cuda.empty_cache()


def analyze(root, group):
    from graph_safety.metrics import evaluate
    parent = Path(read(root / 'parent.json')['root'])
    arms = ARMS[:2] if group == 'frozen' else ARMS[2:]
    result = {'group': group, 'methods': METHODS, 'arms': arms, 'suites': {},
        'limitations': 'Exploratory fixed aggregation grid on previously viewed benchmarks; no token gold. Post-hoc thresholds are not held-out calibration. No method is selected for deployment.'}
    lines = [f'# Token aggregation: {group}', '', result['limitations'], '',
        'Precision/recall/F1 ниже — оптимистический post-hoc максимум min(P,R); ROC/AP/PR-AUC не зависят от порога. R@FPR1% также оценивается по кривой на этих же данных.', '']
    for suite in original.SUITES:
        plans = {r['key']: r for r in read(parent / suite / 'planned.json')}
        outputs = {}
        for arm in arms:
            assert (root / suite / arm / 'complete.json').exists()
            outputs[arm] = {r['key']: r for r in map(json.loads, (root / suite / arm / 'predictions.jsonl').read_text().splitlines())}
        common = sorted(set.intersection(*(set(v) for v in outputs.values())))
        assert len(common) == read(parent / suite / 'comparison.json')['common_cases']
        for key in common:
            assert len({(v[key]['label'], v[key]['ids_sha256']) for v in outputs.values()}) == 1
        analyses = [('main', common, False)]
        if suite == 'harm':
            analyses += [('loose', common, False)]
        if suite == 'stepguard':
            analyses = [(split, [k for k in common if plans[k]['split'] == split], False)
                        for split in ('external_test', 'external_dev')]
        if suite in ('atbench', 'rjudge'):
            tools = [k for k in common if any(a['kind'] == 'tool_call' for a in outputs[arms[0]][k]['actions'])]
            analyses += [('tool_only', tools, True), ('all_events_on_tool_cases', tools, False)]
        blocks = {}
        for name, keys, tool_only in analyses:
            if not keys:
                continue
            labels = [int(plans[k]['original_score'] == 1) if name == 'loose' else plans[k]['label'] for k in keys]
            block = {'n': len(keys), 'metrics': {}, 'length_slices': {}}
            show = name in ('main', 'external_test', 'loose')
            if show:
                lines += [f'## {suite}/{name}: {len(keys)} cases', '',
                    '| Model | Aggregation | AP | PR-AUC | ROC-AUC | P*, % | R*, % | F1*, % | R@FPR1%, % |',
                    '|---|---|---:|---:|---:|---:|---:|---:|---:|']
            for arm, output in outputs.items():
                block['metrics'][arm] = {}
                for method in METHODS:
                    scores = [max(a['methods'][method] for a in output[k]['actions']
                                  if not tool_only or a['kind'] == 'tool_call') for k in keys]
                    m = evaluate(labels, scores)
                    curve = m.pop('pr_curve')
                    candidates = [p for p in curve if p['threshold'] is not None and p['recall'] is not None]
                    best = max(candidates, key=lambda p: (min(p['precision'], p['recall']), p['f1'], p['recall'], p['threshold']))
                    m['posthoc_balanced_point'] = best
                    for fpr in (.01, .05):
                        valid = [p for p in candidates if p['specificity'] is not None and 1-p['specificity'] <= fpr+1e-12]
                        m[f'recall_at_fpr_{fpr}'] = max((p['recall'] for p in valid), default=0.)
                    block['metrics'][arm][method] = m
                    if show:
                        lines.append(f'| {arm} | {method} | {m["average_precision"]:.4f} | {m["pr_auc_trapezoid"]:.4f} | {m["roc_auc"]:.4f} | {100*best["precision"]:.2f} | {100*best["recall"]:.2f} | {100*best["f1"]:.2f} | {100*m["recall_at_fpr_0.01"]:.2f} |')
                    for low, high in ((1, 32), (33, 128), (129, 512), (513, 10**9)):
                        selected = [i for i, k in enumerate(keys) if low <= sum(a['action_tokens'] for a in output[k]['actions']
                                    if not tool_only or a['kind'] == 'tool_call') <= high]
                        if not selected:
                            continue
                        sl = evaluate([labels[i] for i in selected], [scores[i] for i in selected])
                        sl.pop('pr_curve')
                        block['length_slices'].setdefault(arm, {}).setdefault(method, {})[f'{low}:{high}'] = sl
            blocks[name] = block
            if show:
                lines.append('')
        result['suites'][suite] = blocks
    write(root / f'{group}_comparison.json', result)
    (root / f'{group}_comparison.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines), flush=True)


def group_run(root, group):
    arms = ARMS[:2] if group == 'frozen' else ARMS[2:]
    for arm in arms:
        score(root, arm)
    analyze(root, group)
    write(root / f'{group}_status.json', {'status': 'complete', 'arms': arms})


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=('prepare', 'group', 'analyze'))
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--parent', type=Path)
    p.add_argument('--group', choices=('frozen', 'lora'))
    args = p.parse_args()
    if args.phase == 'prepare':
        prepare(args.root, args.parent)
    elif args.phase == 'group':
        group_run(args.root, args.group)
    else:
        analyze(args.root, args.group)
