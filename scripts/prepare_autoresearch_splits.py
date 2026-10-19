"""Freeze CPU-only provenance/component splits over unchanged whole-dialog-v2 rows."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from build_safety_candidates import family, normalized


def serial(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def digest(value):
    return hashlib.sha256(serial(value).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def prepare(args):
    dialogs_path = args.dialogs / 'dialogs.json'
    records_path = args.bundle / 'training/records.json'
    dialogs = json.loads(dialogs_path.read_text())
    records = json.loads(records_path.read_text())
    parent_audit = json.loads((args.dialogs / 'data_audit.json').read_text())
    assert file_hash(dialogs_path) == parent_audit['dialogs_sha256']
    assert file_hash(records_path) == parent_audit['source_records_sha256']
    by_key = {r['key']: r for r in records}
    assert len(by_key) == len(records)
    parents = {}

    def find(node):
        parents.setdefault(node, node)
        while parents[node] != node:
            parents[node] = parents[parents[node]]
            node = parents[node]
        return node

    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parents[max(a, b)] = min(a, b)

    metadata, node_owner = {}, {}
    protected_requests = json.loads((args.bundle / 'benchmarks/request_blocklist.json').read_text())
    protected_families = {digest(family(r)) for r in protected_requests}
    protected_sources = {'atbench', 'atbench500', 'r-judge', 'rjudge', 'stepguard', 'toolsafe', 'ts-bench'}
    provenance_fields = Counter()
    for row in records:
        state = json.loads(row['state'])
        request = state['user_request']
        request_family = digest(family(request))
        assert request_family not in protected_families, row['key']
        sources, domains, families, trajectories = set(row['sources']), set(), set(), set()
        nodes = {'group:' + row['group'], 'request:' + digest(normalized(request)),
                 'request_family:' + request_family, 'input:' + row['input_sha256']}
        for origin in row['origins']:
            sources.add(origin['source'])
            pp = origin['provenance']
            pp = pp if isinstance(pp, list) else [pp]
            for p in pp:
                provenance_fields.update(p.keys())
                source = p.get('source', origin['source'])
                sources.add(source)
                if p.get('domain'):
                    domains.add(source + ':' + str(p['domain']))
                for field, target in [('family_id', families), ('trajectory', trajectories),
                                      ('audit_id', trajectories)]:
                    if p.get(field):
                        value = source + ':' + str(p[field])
                        target.add(value)
                        nodes.add(('family:' if field == 'family_id' else 'trajectory:') + value)
        assert not {s.lower() for s in sources} & protected_sources, row['key']
        for node in sorted(nodes):
            if node in node_owner:
                union(row['group'], node_owner[node])
            else:
                node_owner[node] = row['group']
        metadata[row['key']] = {'sources': sorted(sources), 'domains': sorted(domains),
            'families': sorted(families), 'trajectories': sorted(trajectories),
            'request_sha256': digest(normalized(request)), 'request_family_sha256': request_family,
            'original_group': row['group'], 'input_sha256': row['input_sha256'],
            'state_sha256': hashlib.sha256(row['state'].encode()).hexdigest()}

    # Include excluded actions in component reconstruction: they can bridge admitted rows.
    components = defaultdict(list)
    component_sources, component_domains = defaultdict(set), defaultdict(set)
    for row in records:
        c = find(row['group'])
        component_sources[c].update(metadata[row['key']]['sources'])
        component_domains[c].update(metadata[row['key']]['domains'])
    dialog_meta = {}
    for row in dialogs:
        assert row['ids_sha256'] == digest(row['ids'])
        assert len(row['ids']) == row['tokens']
        sources, domains, families, requests, trajectories = set(), set(), set(), set(), set()
        for span in row['spans']:
            original = by_key[span['key']]
            meta = metadata[span['key']]
            assert original['group'] == row['group'] and original['split'] == row['split']
            assert original['label'] == span['label']
            assert meta['state_sha256'] == span['state_sha256']
            sources.update(meta['sources']); domains.update(meta['domains'])
            families.update(meta['families']); trajectories.update(meta['trajectories'])
            requests.add(meta['request_family_sha256'])
        c = find(row['group'])
        components[c].append(row)
        dialog_meta[row['key']] = {'component': c, 'original_group': row['group'],
            'sources': sorted(sources), 'domains': sorted(domains), 'families': sorted(families),
            'trajectories': sorted(trajectories), 'request_families': sorted(requests)}

    held_sources = set(args.holdout_sources)
    held_domains = {'StepSafe:agentdog_safety'} if 'AgentDoG' in held_sources else set()
    original_validation = [r for r in dialogs if r['split'] == 'validation']
    validation_components = {find(r['group']) for r in original_validation}
    eligible, ood, quarantined = {}, {}, []
    for c, rows in components.items():
        train = [r for r in rows if r['split'] == 'train']
        if not train:
            continue
        if c in validation_components:
            quarantined.extend(train)
        elif component_sources[c] & held_sources or component_domains[c] & held_domains:
            ood[c] = train
        else:
            eligible[c] = train

    # Labels do not participate in component assignment, stratification or subsampling.
    strata = defaultdict(list)
    for c in ood:
        strata[tuple(sorted(component_sources[c] & held_sources))].append(c)
    dev_components = [[], []]
    for stratum, groups in sorted(strata.items()):
        loads = [0, 0]
        for c in sorted(groups, key=lambda g: digest([args.seed, 'dev', g])):
            side = 0 if loads[0] <= loads[1] else 1
            dev_components[side].append(c)
            loads[side] += len(ood[c])

    train_full = [r for c in sorted(eligible) for r in eligible[c]]
    # Proportional quotas preserve source and multi-action composition, whole components.
    train_strata = defaultdict(list)
    for c, rows in eligible.items():
        key = (tuple(sorted(component_sources[c])), any(len(r['spans']) > 1 for r in rows))
        train_strata[key].append(c)
    selected = set()
    for key, groups in sorted(train_strata.items()):
        target = args.small_dialogs * sum(len(eligible[c]) for c in groups) / len(train_full)
        count = 0
        for c in sorted(groups, key=lambda g: digest([args.seed, 'small', g])):
            if abs(count + len(eligible[c]) - target) <= abs(count - target):
                selected.add(c)
                count += len(eligible[c])
    splits = {'train_small': [r for r in train_full if find(r['group']) in selected],
        'train_full': train_full,
        'dev_select': [r for c in dev_components[0] for r in ood[c]],
        'dev_confirm': [r for c in dev_components[1] for r in ood[c]],
        'original_validation': original_validation}
    assert len(splits['train_small']) > 0 and len(splits['dev_select']) > 0 and len(splits['dev_confirm']) > 0

    def counts(rows):
        return {'dialogs': len(rows), 'events': sum(len(r['spans']) for r in rows),
            'positive_events': sum(s['label'] for r in rows for s in r['spans']),
            'unsafe_dialogs': sum(any(s['label'] for s in r['spans']) for r in rows),
            'multi_action_dialogs': sum(len(r['spans']) > 1 for r in rows),
            'multiple_unsafe_dialogs': sum(sum(s['label'] for s in r['spans']) > 1 for r in rows),
            'tokens': sum(r['tokens'] for r in rows), 'max_tokens': max((r['tokens'] for r in rows), default=0),
            'original_groups': len({r['group'] for r in rows}),
            'components': len({find(r['group']) for r in rows}),
            'source_dialogs': dict(Counter(s for r in rows for s in dialog_meta[r['key']]['sources'])),
            'domain_dialogs': dict(Counter(s for r in rows for s in dialog_meta[r['key']]['domains'])),
            'source_positive_events': dict(Counter(s for r in rows for z in r['spans'] if z['label']
                                                   for s in metadata[z['key']]['sources']))}

    leakage = []
    names = list(splits)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            if {left, right} == {'train_small', 'train_full'}:
                continue
            for field in ['component', 'original_group', 'families', 'trajectories', 'request_families']:
                def values(name):
                    result = set()
                    for row in splits[name]:
                        v = dialog_meta[row['key']][field]
                        result.update(v if isinstance(v, list) else [v])
                    return result
                overlap = values(left) & values(right)
                assert not overlap, (left, right, field, overlap)
                leakage.append({'left': left, 'right': right, 'field': field, 'overlap': 0})
    assert not set().union(*(component_sources[c] for c in eligible)) & held_sources
    assert not set().union(*(component_domains[c] for c in eligible)) & held_domains

    cache = args.cache_root / 'features'
    cache_markers = {p.name: {'sha256': file_hash(p), 'metadata': json.loads(p.read_text())}
                     for p in sorted(cache.glob('complete*.json'))}
    assert len(cache_markers) == 2
    assert all(v['metadata']['dialogs_sha256'] == parent_audit['dialogs_sha256'] for v in cache_markers.values())
    missing = [r['key'] for r in dialogs if not (cache / (r['key'] + '.pt')).is_file()]
    assert not missing, missing
    args.output.mkdir(parents=True, exist_ok=False)
    manifests = {}
    for name, rows in splits.items():
        # Exact original rows, including original split fields; never retokenize or insert metadata.
        path = args.output / (name + '.json')
        write(path, rows)
        manifest = {'name': name, 'rows_file': path.name, 'sha256': file_hash(path),
            'counts': counts(rows), 'dialog_keys': [r['key'] for r in rows],
            'component_ids': sorted({find(r['group']) for r in rows}),
            'rows': [{'key': r['key'], 'row_sha256': digest(r), 'ids_sha256': r['ids_sha256'],
                      'tokens': r['tokens'], 'original_split': r['split'],
                      'action_keys': [s['key'] for s in r['spans']], **dialog_meta[r['key']]}
                     for r in rows]}
        write(args.output / (name + '.manifest.json'), manifest)
        manifests[name] = {'sha256': manifest['sha256'], **manifest['counts']}
    write(args.output / 'action_metadata_audit.json', metadata)
    write(args.output / 'quarantined_train.json', {'counts': counts(quarantined),
        'dialog_keys': [r['key'] for r in quarantined],
        'reason': 'Reconstructed family/trajectory/request component intersects original validation.'})
    inputs = [dialogs_path, args.dialogs / 'data_audit.json', records_path,
              args.bundle / 'benchmarks/request_blocklist.json', args.bundle / 'benchmarks/overlap_audit.json',
              Path(__file__), Path(__file__).with_name('build_safety_candidates.py')]
    audit = {'version': 'autoresearch-sixhour-v1', 'seed': args.seed,
        'input_files': {str(p.resolve()): file_hash(p) for p in inputs},
        'heldout_sources_all_origin_provenance': sorted(held_sources),
        'heldout_domains': sorted(held_domains), 'splits': manifests,
        'quarantined_train': counts(quarantined), 'leakage_checks': leakage,
        'provenance_fields_observed': dict(provenance_fields),
        'cache_root_read_only': str(args.cache_root.resolve()), 'cache_markers': cache_markers,
        'cache_files_present': len(dialogs),
        'cache_verification': 'Existence and parent completion metadata only; tensors not loaded or hashed. Trainer must check per-row ids_sha256 and cache identity.',
        'sampling': 'Fixed salted SHA256 order; provenance/component strata; no labels or model outputs used for splitting or subsampling.',
        'labels': 'Original span labels unchanged, only for loss/evaluation and reported aggregate counts; no label/teacher/rationale metadata inserted into model inputs.',
        'limitations': [
            'StepSafe is also a repackaging of AgentDoG; shared StepSafe attribution remains. Held-out AgentDoG provenance and agentdog_safety domain are excluded from training, not every source name disjoint.',
            'Domain tags are incomplete; missing domain is not inferred from synthetic IDs. StepShield domain=train is a source partition, not a semantic domain.',
            'Original validation is previously viewed/easy, not a new independent holdout. Dev confirm must not guide selection.',
            'Family reconstruction uses existing components plus exact/entity-masked requests and explicit provenance family/trajectory/audit IDs across all source records; not proof of semantic independence.',
            'Protected benchmark exclusions inherited from immutable parent bundle, plus exact/entity-masked request blocklist and source-name checks. No benchmark rows added; no new semantic contamination audit.',
            'Source counts are multi-attribution, not mutually exclusive. AgentAlign may have no unsafe examples; source-balanced labels cannot be fabricated.',
            'All original dialog rows and split fields unchanged; caller selects explicit files/keys, not original split=train/validation fields. train_small is intentionally a subset of train_full.'
        ], 'no_truncation': True, 'no_gpu': True}
    write(args.output / 'audit.json', audit)
    print(json.dumps({'output': str(args.output), 'splits': manifests,
                      'quarantined_train': counts(quarantined)}, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    root = Path('/var/local/danefedov/experiments/agent-safety')
    p.add_argument('--dialogs', type=Path, default=root / 'datasets/whole-dialog-4b-v2')
    p.add_argument('--bundle', type=Path, default=root / 'datasets/data-bundle-whole-dialog-v1')
    p.add_argument('--cache-root', type=Path, default=root / 'runs/whole-dialog-4b-v3')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--small-dialogs', type=int, default=2000)
    p.add_argument('--seed', default='autoresearch-sixhour-v1-fixed-data')
    p.add_argument('--holdout-sources', nargs='+', default=['AgentDoG', 'AgentAlign'])
    prepare(p.parse_args())
