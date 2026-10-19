"""Freeze already labeled action data, protecting evaluation requests and scenario groups."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from build_safety_candidates import digest, family, normalized
from audit_candidate_overlap import shingles

ROOT = Path('/var/local/danefedov/experiments/agent-safety')
OUT = ROOT / 'datasets/available-action-training-v1'


def write(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')


def file_hash(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        while b := f.read(1024 * 1024):
            h.update(b)
    return h.hexdigest()


def main():
    from datasketch import MinHash, MinHashLSH
    OUT.mkdir(exist_ok=False)
    corpus = ROOT / 'datasets/corpus-v1'
    paths = [ROOT / f'datasets/injection-actions-v1/{source}/records.jsonl'
             for source in ('toolsafe', 'stepsafe')]
    paths += [corpus / f'annotations/{name}/silver.jsonl' for name in
              ('strong_pilot', 'strong_synthetic', 'strong_bulk_checkpoint_v1')]
    parent = {}

    def find(k):
        parent.setdefault(k, k)
        if parent[k] != k:
            parent[k] = find(parent[k])
        return parent[k]

    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[max(a, b)] = min(a, b)

    unique, requests = {}, {}
    input_counts = Counter()
    for path in paths:
        for line in path.open():
            row = json.loads(line)
            assert row['label'] in (0, 1)
            state = json.loads(row['state']) if isinstance(row['state'], str) else row['state']
            state = {k: state[k] for k in ('tool_interfaces', 'user_request', 'observed_history', 'proposed_action')}
            h = digest(state)
            request_id = digest(family(state['user_request']))
            node = 'request:' + request_id
            requests[request_id] = state['user_request']
            union(node, 'group:' + row['group'])
            provenance = row['provenance']
            pp = provenance if isinstance(provenance, list) else [provenance]
            for p in pp:
                if p.get('family_id'):
                    union(node, 'origin:' + p['source'] + ':' + p['family_id'])
            origin = {'key': row['key'], 'file': str(path.relative_to(ROOT)),
                      'source': row.get('source') or pp[0]['source'], 'label': row['label'],
                      'provenance': provenance, 'teacher_labels': row.get('teacher_labels')}
            input_counts[origin['source']] += 1
            if h in unique:
                unique[h]['origins'].append(origin)
            else:
                unique[h] = {'key': 'available:' + h, 'state': json.dumps(state, ensure_ascii=False),
                    'input_sha256': h, 'request_family': request_id, 'label': row['label'],
                    'origins': [origin], 'trajectory': row.get('trajectory') or row['key'],
                    'mechanism': 'published_or_strong_silver_action',
                    'previous_unsafe': None, 'previous_safe': None}
    evaluations = json.loads((corpus / 'evaluation_requests.json').read_text())
    known = json.loads((corpus / 'holdout_blocklist.json').read_text())
    blocked = set()
    index = MinHashLSH(threshold=.75, num_perm=64)
    stored = {}

    def signature(s):
        m = MinHash(num_perm=64, seed=20261018)
        m.update_batch([v.encode() for v in sorted(s)])
        return m

    for i, text in enumerate(evaluations):
        s = shingles(text)
        stored['eval:' + str(i)] = s
        index.insert('eval:' + str(i), signature(s))
    for req, text in sorted(requests.items()):
        node = 'request:' + req
        if digest(normalized(text)) in known['request_hashes'] or req in known['request_family_hashes']:
            blocked.add(node)
        s = shingles(text)
        m = signature(s)
        for other in index.query(m):
            t = stored[other]
            if len(s & t) / len(s | t) < .85:
                continue
            if other.startswith('eval:'):
                blocked.add(node)
            else:
                union(node, other)
        stored[node] = s
        index.insert(node, m)
    blocked = {find(k) for k in blocked}
    records, rejected = [], []
    for row in unique.values():
        group = find('request:' + row.pop('request_family'))
        reason = None
        if group in blocked:
            reason = 'evaluation_overlap_connected_family'
        if len({p['label'] for p in row['origins']}) > 1:
            reason = 'conflicting_labels'
        row['group'] = digest(group)
        row['sources'] = sorted({o['source'] for o in row['origins']})
        if reason:
            rejected.append(dict(row, reason=reason))
        else:
            bucket = int(hashlib.sha256(('available-v1:' + row['group']).encode()).hexdigest()[:8], 16) % 10
            row['split'] = 'test' if bucket == 0 else 'validation' if bucket == 1 else 'train'
            records.append(row)
    # Existing StepGuard action holdout stays external. No source labels/rationales enter state.
    reference = ROOT / 'archive/legacy-shm/danefedov-controlled-monotonic-verified-v1/results/records.json'
    external = [r for r in json.loads(reference.read_text()) if r['split'] == 'external_test']
    for r in external:
        r['sources'] = ['StepGuard']
    assert all(len({r['split'] for r in records if r['group'] == g}) == 1 for g in {r['group'] for r in records})
    assert all({r['label'] for r in records if r['split'] == s} == {0, 1} for s in ('train', 'validation', 'test'))
    write(OUT / 'records.json', records + external)
    with (OUT / 'quarantine.jsonl').open('w') as f:
        for r in rejected:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    counts = Counter(f'{r["split"]}:{r["label"]}' for r in records + external)
    source_counts = Counter((r['split'], '+'.join(r['sources'])) for r in records)
    summary = {'counts': dict(counts), 'group_count': len({r['group'] for r in records}),
        'source_inputs': dict(input_counts), 'unique_internal_actions': len(records),
        'source_slices': {str(k): v for k, v in source_counts.items()},
        'deduplicated_inputs': sum(input_counts.values()) - len(unique),
        'quarantine': len(rejected), 'quarantine_reasons': dict(Counter(r['reason'] for r in rejected)),
        'external_test': len(external), 'records_sha256': file_hash(OUT / 'records.json'),
        'source_files_sha256': {str(p.relative_to(ROOT)): file_hash(p) for p in paths + [reference]},
        'data_snapshot': 'Already available labels only; future judge outputs do not change this snapshot.',
        'label_scope': 'current_action_safety', 'split': '80/10/10 hash of connected request/source families; lexical Jaccard groups and heldout exclusion',
        'limits': ['Published benchmark annotations repurposed as exploratory training, not their official train split.',
            'No independent full-benchmark claims on ToolSafe/StepSafe/AgentDojo families after this training.',
            'Machine silver and published annotations are not new human gold.',
            'Partial teacher completion can bias source/class composition.',
            'StepGuard was evaluated previously; this is exploratory transfer, not untouched confirmation.',
            'No event/token injection labels inferred; no monotonic loss or artificial token curves.']}
    write(OUT / 'summary.json', summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
