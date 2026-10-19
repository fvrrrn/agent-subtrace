"""Read-only CPU audit; never opens confirm, benchmark, or prediction files."""
import collections
import hashlib
import json
from pathlib import Path

ROOT = Path('/var/local/danefedov/experiments/agent-safety')
DATA = ROOT / 'runs/autoresearch-sixhour-v1/data'


def load(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


records_path = ROOT / 'datasets/data-bundle-whole-dialog-v1/training/records.json'
records = load(records_path)
by_key = {r['key']: r for r in records}
result = {'protocol': 'CPU read-only; sample SHA256(semantics-review-v1:+key), first 3 per source, no label/prediction selection',
          'records_sha256': sha(records_path), 'splits': {}, 'samples': []}
for name in ('train_small', 'train_full'):
    path = DATA / (name + '.json')
    rows = load(path)
    manifest = load(DATA / (name + '.manifest.json'))
    keys = [s['key'] for r in rows for s in r['spans']]
    admitted = [by_key[k] for k in keys]
    source = collections.defaultdict(list)
    for r in admitted:
        for s in r['sources']:
            source[s].append(r)
    groups = collections.Counter(r['group'] for r in rows)
    origins = collections.Counter(o['file'] for r in admitted for o in r['origins'])
    qualities = collections.Counter(str(p.get('label_quality', p.get('source_label_status'))) for r in admitted for o in r['origins'] for p in (o['provenance'] if isinstance(o['provenance'], list) else [o['provenance']]))
    input_labels = collections.defaultdict(set)
    for r in admitted:
        input_labels[r['input_sha256']].add(r['label'])
    action_results = [r['key'] for r in admitted if isinstance(json.loads(r['state'])['proposed_action'], dict) and 'tool_result' in json.loads(r['state'])['proposed_action']]
    evidence = collections.Counter()
    audit_methods = collections.Counter()
    for r in admitted:
        native = any('/injection-actions-v1/' in o['file'] for o in r['origins'])
        silver = any(o.get('teacher_labels') for o in r['origins'])
        evidence['native_and_silver' if native and silver else 'native_only' if native else 'silver_only' if silver else 'neither'] += 1
        for o in r['origins']:
            if '/injection-actions-v1/' in o['file']:
                audit_methods[str(o['provenance'].get('audit_method'))] += 1
    prefixes = collections.defaultdict(list)
    for r in rows:
        for s in r['spans']:
            h = hashlib.sha256(json.dumps(r['ids'][:s['end']], separators=(',', ':')).encode()).hexdigest()
            prefixes[h].append((s['key'], s['label']))
    collisions = []
    for h, values in prefixes.items():
        if len({label for _, label in values}) <= 1:
            continue
        states = [json.loads(by_key[k]['state']) for k, _ in values]
        different = [f for f in states[0] if any(s[f] != states[0][f] for s in states[1:])]
        collisions.append({'causal_prefix_sha256':h, 'keys_labels':values, 'different_state_fields':different,
            'differences':[{f:s[f] for f in different} for s in states],
            'shared_action':states[0]['proposed_action'],
            'history_last_event':states[0]['observed_history'][-1] if states[0]['observed_history'] else None,
            'origins':[[{'key':o['key'],'provenance':o['provenance']} for o in by_key[k]['origins']] for k,_ in values]})
    result['splits'][name] = {'sha256': sha(path), 'dialogs': len(rows), 'events': len(keys), 'unique_actions': len(set(keys)),
        'unsafe': sum(r['label'] for r in admitted), 'groups': len(groups), 'components': len(manifest['component_ids']),
        'max_dialogs_per_group': max(groups.values()), 'tokens': sum(r['tokens'] for r in rows),
        'action_tokens': sum(s['end']-s['start'] for r in rows for s in r['spans']),
        'max_tokens': max(r['tokens'] for r in rows), 'sources': {s: {'events': len(v), 'unsafe': sum(r['label'] for r in v), 'groups':len({r['group'] for r in v})} for s,v in source.items()},
        'origin_files': dict(origins), 'quality_fields': dict(qualities), 'duplicate_input_hashes':len(admitted)-len(input_labels),
        'contradictory_inputs':sum(len(v)>1 for v in input_labels.values()),
        'contradictory_origin_labels':sum(any(o['label']!=r['label'] for o in r['origins']) for r in admitted),
        'label_evidence_rows': dict(evidence), 'native_audit_methods':dict(audit_methods),
        'duplicate_serialized_causal_prefixes':sum(len(v)-1 for v in prefixes.values()),
        'contradictory_serialized_causal_prefixes':[v for v in prefixes.values() if len({label for _,label in v})>1],
        'collision_details':collisions,
        'current_action_tool_result_keys': action_results}
    if name == 'train_full':
        for s, values in sorted(source.items()):
            chosen = sorted(values, key=lambda r: hashlib.sha256(('semantics-review-v1:'+r['key']).encode()).hexdigest())[:3]
            for r in chosen:
                result['samples'].append({'source_stratum':s, **{k:r[k] for k in ('key','label','group','origins')}, 'state': json.loads(r['state'])})
print(json.dumps(result, ensure_ascii=False, indent=2))
