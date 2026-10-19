"""Paired component bootstrap of two locked event predictors; never fit thresholds."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import random

from graph_safety.metrics import evaluate


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(',', ':')).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_events(path):
    result = {}
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            raise ValueError(f'{path}:{line_number}: blank event row; no silent skipping')
        raw = json.loads(line)
        span = raw.get('span', raw)
        key, label, score = span['key'], span['label'], raw['score']
        if key in result:
            raise ValueError(f'{path}: duplicate event key {key}')
        if label not in (0, 1) or isinstance(score, bool) or not isinstance(score, (int, float)):
            raise ValueError(f'{path}: invalid binary label or score for {key}')
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError(f'{path}: score outside finite [0, 1] for {key}')
        if not span.get('state_sha256') or not raw.get('ids_sha256'):
            raise ValueError(f'{path}: state_sha256 and ids_sha256 required for {key}')
        result[key] = {'label': label, 'score': score, 'state_sha256': span['state_sha256'],
                       'ids_sha256': raw['ids_sha256'], 'dialog': raw.get('dialog')}
    if not result:
        raise ValueError(f'Empty event file: {path}')
    return result


def metrics(rows, threshold):
    result = evaluate([r['label'] for r in rows], [r['score'] for r in rows], threshold)
    return {name: result[name] for name in ('average_precision', 'pr_auc_trapezoid', 'roc_auc')} | {
        'f1': result['operating_point']['f1']}


def quantile(values, q):
    values = sorted(values)
    position = (len(values) - 1) * q
    lo, hi = math.floor(position), math.ceil(position)
    return values[lo] + (values[hi] - values[lo]) * (position - lo)


def summarize(values, repetitions):
    valid = [v for v in values if v is not None]
    return {'valid_replicates': len(valid), 'undefined_replicates': repetitions - len(valid),
            'mean': sum(valid) / len(valid) if valid else None,
            'percentile_95_ci': [quantile(valid, .025), quantile(valid, .975)] if valid else None}


def run(args):
    if args.repetitions < 500:
        raise ValueError('At least 500 bootstrap repetitions required')
    for threshold in (args.threshold_a, args.threshold_b):
        if not math.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError('Locked thresholds must be finite in [0, 1]')
    manifest = json.loads(args.manifest.read_text())
    event_meta, components = {}, defaultdict(list)
    component_attribution = defaultdict(lambda: {'sources': set(), 'domains': set(), 'dialogs': set()})
    for row in manifest['rows']:
        component = row['component']  # Explicit reconstructed family component, never guessed from IDs.
        if not component:
            raise ValueError(f'Missing component in dialog {row["key"]}')
        attribution = component_attribution[component]
        attribution['sources'].update(row['sources'])
        attribution['domains'].update(row['domains'])
        attribution['dialogs'].add(row['key'])
        for key in row['action_keys']:
            if key in event_meta:
                raise ValueError(f'Duplicate manifest action key: {key}')
            event_meta[key] = {'dialog': row['key'], 'ids_sha256': row['ids_sha256'],
                               'component': component}
            components[component].append(key)
    if not event_meta:
        raise ValueError('Manifest contains no actions')
    a, b = load_events(args.events_a), load_events(args.events_b)
    expected = set(event_meta)
    coverage = {'expected_events': len(expected), 'events_a': len(a), 'events_b': len(b),
        'common_event_keys': sorted(set(a) & set(b)),
        'missing_a': sorted(expected - set(a)), 'missing_b': sorted(expected - set(b)),
        'unexpected_a': sorted(set(a) - expected), 'unexpected_b': sorted(set(b) - expected)}
    if any(coverage[field] for field in ('missing_a', 'missing_b', 'unexpected_a', 'unexpected_b')):
        raise ValueError('Exact full manifest case set required; no intersection-only filtering: ' +
                         json.dumps(coverage, ensure_ascii=False))
    changed_pair, changed_a, changed_b = [], [], []
    keys = list(event_meta)  # Manifest order; input JSONL ordering need not match.
    for key in keys:
        left, right, meta = a[key], b[key], event_meta[key]
        if left['label'] != right['label'] or left['state_sha256'] != right['state_sha256']:
            raise ValueError(f'Paired label/state hash mismatch: {key}')
        for name, event in [('a', left), ('b', right)]:
            if event['dialog'] is not None and event['dialog'] != meta['dialog']:
                raise ValueError(f'Model {name} dialog ID mismatch: {key}')
        if left['ids_sha256'] != right['ids_sha256']:
            changed_pair.append(key)
        if left['ids_sha256'] != meta['ids_sha256']:
            changed_a.append(key)
        if right['ids_sha256'] != meta['ids_sha256']:
            changed_b.append(key)
    input_changes = {'pair_different_token_ids': changed_pair,
                     'a_different_from_manifest': changed_a, 'b_different_from_manifest': changed_b}
    if any(input_changes.values()) and not args.allow_input_change:
        raise ValueError('Token-ID changes require explicit --allow-input-change: ' + json.dumps(input_changes))

    observed_a = metrics([a[k] for k in keys], args.threshold_a)
    observed_b = metrics([b[k] for k in keys], args.threshold_b)
    values = {name: {'a': [], 'b': [], 'delta_b_minus_a': []} for name in observed_a}
    rng = random.Random(args.seed)
    group_ids = sorted(components)
    sampled_event_counts = []
    sample_digest = hashlib.sha256()
    for _ in range(args.repetitions):
        sampled = [group_ids[rng.randrange(len(group_ids))] for _ in group_ids]
        sample_digest.update((json.dumps(sampled, separators=(',', ':')) + '\n').encode())
        # Repeated components repeat ALL their events; both models receive the identical draw.
        sampled_keys = [key for group in sampled for key in components[group]]
        sampled_event_counts.append(len(sampled_keys))
        ma = metrics([a[k] for k in sampled_keys], args.threshold_a)
        mb = metrics([b[k] for k in sampled_keys], args.threshold_b)
        for name in values:
            values[name]['a'].append(ma[name]); values[name]['b'].append(mb[name])
            values[name]['delta_b_minus_a'].append(
                mb[name] - ma[name] if ma[name] is not None and mb[name] is not None else None)
    summaries = {}
    for name, distributions in values.items():
        summaries[name] = {side: summarize(v, args.repetitions) for side, v in distributions.items()}
        valid = [v for v in distributions['delta_b_minus_a'] if v is not None]
        summaries[name]['delta_b_minus_a'].update({
            'fraction_positive_descriptive': sum(v > 0 for v in valid) / len(valid) if valid else None,
            'fraction_zero_descriptive': sum(v == 0 for v in valid) / len(valid) if valid else None,
            'observed': observed_b[name] - observed_a[name]
                if observed_a[name] is not None and observed_b[name] is not None else None})
    result = {'report_version': 1, 'delta_direction': 'B minus A',
        'inputs': {name: {'path': str(path.resolve()), 'sha256': file_hash(path)} for name, path in
                   [('events_a', args.events_a), ('events_b', args.events_b), ('manifest', args.manifest)]},
        'locked_thresholds': {'a': args.threshold_a, 'b': args.threshold_b}, 'thresholds_refitted': False,
        'coverage': coverage, 'common_full_event_count': len(keys),
        'ordered_manifest_event_keys_sha256': digest(keys),
        'allow_input_change': args.allow_input_change, 'input_changes': input_changes,
        'observed': {'a': observed_a, 'b': observed_b}, 'bootstrap': summaries,
        'sampling': {'unit': 'explicit manifest family/component', 'paired': True,
            'with_replacement': True, 'seed': args.seed, 'repetitions': args.repetitions,
            'components_per_draw': len(group_ids), 'sampled_components_sha256': sample_digest.hexdigest(),
            'event_counts_min_max': [min(sampled_event_counts), max(sampled_event_counts)]},
        'components': {c: {'event_keys': components[c],
            **{field: sorted(value) for field, value in component_attribution[c].items()}}
            for c in group_ids},
        'paired_events': [{'key': k, **event_meta[k], 'label': a[k]['label'],
            'state_sha256': a[k]['state_sha256'], 'ids_sha256_a': a[k]['ids_sha256'],
            'ids_sha256_b': b[k]['ids_sha256'], 'score_a': a[k]['score'], 'score_b': b[k]['score']}
            for k in keys],
        'caveats': [
            'Run only after both finalists and evaluation protocol are locked. This program cannot verify that the caller did not inspect confirmation scores earlier.',
            'Percentile intervals and positive-delta fractions are descriptive conditional on this case set and finalist selection; positive fraction is not a formal p-value or posterior probability.',
            'Components preserve all within-family/dialog/event dependence captured by the manifest. Residual cross-component semantic or source dependence is not eliminated.',
            'Component source/domain attribution is overlapping, dialog-level provenance, not independent source/domain sampling units. This is not an uncertainty estimate for unseen domains or sources.',
            'Components are sampled globally, not source-stratified; source mix and event count can vary across draws. Metrics remain ordinary event-weighted AP, not component-weighted or source-balanced AP.',
            'All-safe draws leave AP and trapezoidal PR-AUC undefined; single-class draws leave ROC-AUC undefined. Undefined values are counted and omitted only from that metric interval, never imputed as zero.',
            'F1 uses each model\'s separately supplied locked threshold on every draw; no threshold search or calibration is performed.',
            'With --allow-input-change, comparison may mix an instruction/input change with a model change. Exact action keys, state hashes and labels must still match; token-ID differences are enumerated.'
        ]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'output': str(args.output), 'events': len(keys), 'components': len(group_ids),
                      'bootstrap': summaries}, indent=2, allow_nan=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--events-a', type=Path, required=True)
    parser.add_argument('--events-b', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--threshold-a', type=float, required=True)
    parser.add_argument('--threshold-b', type=float, required=True)
    parser.add_argument('--repetitions', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=20261018)
    parser.add_argument('--allow-input-change', action='store_true')
    parser.add_argument('--output', type=Path, required=True, help='New JSON path; refuses to overwrite')
    run(parser.parse_args())
