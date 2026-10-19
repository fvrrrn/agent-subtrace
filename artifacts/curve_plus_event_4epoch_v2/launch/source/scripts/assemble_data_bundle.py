"""Separate protected benchmarks and freeze available labels into grouped 95/5 data."""
from collections import Counter, defaultdict
import argparse
import hashlib
import json
from pathlib import Path
import time

from assemble_available_corpus import file_hash, write
from build_safety_candidates import digest, family, normalized
from audit_candidate_overlap import shingles
from resume_corpus_annotation import aggregate

ROOT = Path('/var/local/danefedov/experiments/agent-safety')
OUT = ROOT / 'datasets/data-bundle-v1'


def jsonl(path):
    with path.open() as f:
        for line in f:
            yield json.loads(line)


def save_lines(path, rows):
    with path.open('w') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')


def main():
    from datasketch import MinHash, MinHashLSH
    OUT.mkdir(exist_ok=False)
    for name in ('training', 'benchmarks', 'unlabeled', 'quarantine', 'snapshots'):
        (OUT / name).mkdir()
    corpus = ROOT / 'datasets/corpus-v1'
    inputs = {}
    capture = []
    outputs = defaultdict(dict)
    # Fixed byte-prefix snapshots: judges can continue appending without changing this bundle.
    for name in ('strong_bulk', 'strong_bulk_2gpu'):
        path = corpus / 'annotations' / name / 'calls.jsonl'
        cutoff = path.stat().st_size
        with path.open('rb') as f:
            data = f.read(cutoff)
        end = data.rfind(b'\n') + 1
        complete = data[:end]
        target = OUT / 'snapshots' / (name + '_calls.jsonl')
        target.write_bytes(complete)
        capture.append({'source': str(path), 'captured_at_unix': time.time(),
                        'cutoff_bytes': cutoff, 'complete_bytes': end,
                        'trailing_incomplete_bytes_not_in_snapshot': len(data) - end,
                        'sha256': file_hash(target)})
        for r in jsonl(target):
            assert r['teacher'] not in outputs[r['key']], 'Duplicate completed checkpoint judgment'
            outputs[r['key']][r['teacher']] = r
    selections = corpus / 'annotations/strong_bulk_checkpoint_v1/selection_records.jsonl'
    selection_rows = list(jsonl(selections))
    inputs[str(selections)] = file_hash(selections)
    silver_summary = aggregate(selection_rows, outputs, OUT / 'snapshots')
    pending = (dict(r, split='unassigned', label=None, status='pending_teacher_pair')
               for r in selection_rows if len(outputs.get(r['key'], {})) != 2)
    save_lines(OUT / 'unlabeled/pending_selected_records.jsonl', pending)
    write(OUT / 'snapshots/annotation_snapshot.json', {'captures': capture, 'counts': silver_summary})
    del outputs, selection_rows, data, complete

    reference = ROOT / 'archive/legacy-shm/danefedov-controlled-monotonic-verified-v1/results/records.json'
    stepguard = [dict(r, original_split=r['split'], split='benchmark')
                 for r in json.loads(reference.read_text()) if r['split'].startswith('external_')]
    (OUT / 'benchmarks/stepguard').mkdir()
    save_lines(OUT / 'benchmarks/stepguard/records.jsonl', stepguard)
    # Include ALL TS-Bench records, not just its definite or previously admitted subset.
    toolsafe = [dict(r, split='benchmark') for r in jsonl(ROOT / 'datasets/injection-actions-v1/source_records.jsonl')
                if r['source'] == 'ToolSafe']
    (OUT / 'benchmarks/toolsafe').mkdir()
    save_lines(OUT / 'benchmarks/toolsafe/records.jsonl', toolsafe)
    for name, target in {
        'rjudge': ROOT / 'archive/legacy-home/laya-rjudge-baseline-20261013/data/rjudge',
        'atbench': corpus / 'raw/AI45Research--ATBench',
        'stepguard/raw': corpus / 'raw/ninty-seven--stepguard_data',
        'toolsafe/raw': corpus / 'raw/MurrayTom--ToolSafe/snapshot/TS-Bench',
    }.items():
        assert target.exists(), target
        (OUT / 'benchmarks' / name).symlink_to(target, target_is_directory=True)
    benchmark_requests = json.loads((corpus / 'evaluation_requests.json').read_text())
    benchmark_requests += [json.loads(r['state'])['user_request'] for r in stepguard + toolsafe]
    benchmark_requests = sorted(set(benchmark_requests))
    benchmark_states = {r['input_sha256'] for r in toolsafe}
    benchmark_states.update(digest(json.loads(r['state'])) for r in stepguard)
    write(OUT / 'benchmarks/registry.json', {
        'protected': ['R-Judge', 'ATBench (including legacy ATBench500)', 'StepGuard (dev AND test)', 'ToolSafe/TS-Bench'],
        'stepguard_normalized_actions': len(stepguard), 'toolsafe_source_steps': len(toolsafe),
        'request_blocklist_count': len(benchmark_requests),
        'rule': 'Never include these benchmark rows or matching request families in training/validation. Benchmark raw files retain their original labels and partitions.',
        'raw_links': 'Pointers to immutable original acquired files; not copies of training rows.'})
    write(OUT / 'benchmarks/request_blocklist.json', benchmark_requests)

    paths = [ROOT / 'datasets/injection-actions-v1/stepsafe/records.jsonl',
             corpus / 'annotations/strong_pilot/silver.jsonl',
             corpus / 'annotations/strong_synthetic/silver.jsonl', OUT / 'snapshots/silver.jsonl']
    parent = {}

    def find(k):
        parent.setdefault(k, k)
        root = k
        while parent[root] != root:
            root = parent[root]
        while k != root:
            k, parent[k] = parent[k], root
        return root

    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[max(a, b)] = min(a, b)

    unique, requests = {}, {}
    source_inputs = Counter()
    for path in paths:
        inputs[str(path)] = file_hash(path)
        for r in jsonl(path):
            assert r['label'] in (0, 1)
            state = json.loads(r['state']) if isinstance(r['state'], str) else r['state']
            state = {k: state[k] for k in ('tool_interfaces', 'user_request', 'observed_history', 'proposed_action')}
            h = digest(state)
            req = digest(family(state['user_request']))
            node = 'request:' + req
            requests[req] = state['user_request']
            union(node, 'group:' + r['group'])
            provenance = r['provenance'] if isinstance(r['provenance'], list) else [r['provenance']]
            for p in provenance:
                if p.get('family_id'):
                    union(node, 'origin:' + p['source'] + ':' + p['family_id'])
            source = r.get('source') or provenance[0]['source']
            assert source != 'ToolSafe'
            source_inputs[source] += 1
            origin = {'source': source, 'key': r['key'], 'file': str(path), 'label': r['label'],
                      'provenance': r['provenance'], 'teacher_labels': r.get('teacher_labels')}
            if h in unique:
                unique[h]['origins'].append(origin)
            else:
                unique[h] = {'key': 'bundle:' + h, 'state': json.dumps(state, ensure_ascii=False),
                             'label': r['label'], 'label_scope': 'action_safety', 'input_sha256': h,
                             'request_family': req, 'origins': [origin],
                             'trajectory': r.get('trajectory') or r['key'],
                             'mechanism': 'published_or_strong_silver_action',
                             'previous_unsafe': None, 'previous_safe': None}
    index = MinHashLSH(threshold=.75, num_perm=64)
    stored, blocked, matches = {}, set(), []
    protected = {digest(family(t)) for t in benchmark_requests}

    def signature(s):
        m = MinHash(num_perm=64, seed=20261018)
        m.update_batch([v.encode() for v in sorted(s)])
        return m

    for i, text in enumerate(benchmark_requests):
        s = shingles(text)
        stored['eval:' + str(i)] = s
        index.insert('eval:' + str(i), signature(s))
    for req, text in sorted(requests.items()):
        node = 'request:' + req
        if req in protected:
            blocked.add(node)
        s = shingles(text)
        m = signature(s)
        for other in index.query(m):
            t = stored[other]
            jaccard = len(s & t) / len(s | t)
            if jaccard < .85:
                continue
            if other.startswith('eval:'):
                blocked.add(node)
                matches.append({'request_family': req, 'evaluation_index': int(other.split(':')[1]), 'jaccard': jaccard})
            else:
                union(node, other)
        stored[node] = s
        index.insert(node, m)
    blocked = {find(k) for k in blocked}
    admitted, rejected = [], []
    for r in unique.values():
        group = find('request:' + r.pop('request_family'))
        reasons = []
        if group in blocked or r['input_sha256'] in benchmark_states:
            reasons.append('protected_benchmark_family_overlap')
        if len({p['label'] for p in r['origins']}) > 1:
            reasons.append('conflicting_labels')
        r['group'] = digest(group)
        r['sources'] = sorted({p['source'] for p in r['origins']})
        if reasons:
            rejected.append(dict(r, quarantine_reasons=reasons, split='quarantine'))
        else:
            admitted.append(r)
    groups = defaultdict(list)
    for r in admitted:
        groups[r['group']].append(r)
    target = round(.05 * len(admitted))
    val_groups, val_count = set(), 0
    # Deterministic whole-family selection approaching 5% of rows; labels/predictions not used.
    for g in sorted(groups, key=lambda x: hashlib.sha256(('bundle-v1-95-5:' + x).encode()).hexdigest()):
        n = len(groups[g])
        if abs(val_count + n - target) < abs(val_count - target):
            val_groups.add(g)
            val_count += n
    for r in admitted:
        r['split'] = 'validation' if r['group'] in val_groups else 'train'
    assert all({r['label'] for r in admitted if r['split'] == split} == {0, 1} for split in ('train', 'validation'))
    assert len({r['input_sha256'] for r in admitted}) == len(admitted)
    assert {r['group'] for r in admitted if r['split'] == 'train'}.isdisjoint(val_groups)
    assert not {r['input_sha256'] for r in admitted} & benchmark_states
    write(OUT / 'training/records.json', admitted)
    for split in ('train', 'validation'):
        save_lines(OUT / 'training' / (split + '.jsonl'), (r for r in admitted if r['split'] == split))
    save_lines(OUT / 'quarantine/benchmark_overlap_or_conflicts.jsonl', rejected)
    write(OUT / 'benchmarks/overlap_audit.json', {'matches': matches, 'blocked_components': len(blocked),
          'method': 'Exact/entity-masked requests + MinHash retrieval and exact five-word Jaccard >=0.85; exclude whole connected scenario families.',
          'limitation': 'Lexical audit, not proof of semantic independence.'})
    (OUT / 'unlabeled/master_candidate_pool').symlink_to(corpus / 'derived/candidates_v3', target_is_directory=True)
    (OUT / 'raw_sources').symlink_to(corpus / 'raw', target_is_directory=True)
    (OUT / 'quarantine/teacher_disagreements.jsonl').symlink_to(OUT / 'snapshots/quarantine.jsonl')
    write(OUT / 'source_inventory.json', {
        'acquired_sources': json.loads((corpus / 'data_inventory.json').read_text()),
        'raw_sources': str(corpus / 'raw'), 'master_candidates': str(corpus / 'derived/candidates_v3'),
        'rule': 'Raw sources and unannotated/trajectory-only labels are NOT automatically training examples.',
        'training_input_hashes': inputs})
    counts = Counter(f'{r["split"]}:{r["label"]}' for r in admitted)
    summary = {'unique_labeled_actions': len(admitted), 'train': len(admitted) - val_count, 'validation': val_count,
        'validation_percent': 100 * val_count / len(admitted), 'validation_target': target,
        'class_counts': dict(counts), 'source_inputs': dict(source_inputs),
        'source_slices': {str(k): v for k, v in Counter((r['split'], '+'.join(r['sources']), r['label']) for r in admitted).items()},
        'groups': len(groups), 'validation_groups': len(val_groups),
        'duplicate_inputs_merged': sum(source_inputs.values()) - len(unique),
        'quarantined_unique_actions': len(rejected),
        'quarantine_reasons': dict(Counter(reason for r in rejected for reason in r['quarantine_reasons'])),
        'protected_benchmarks': ['R-Judge', 'ATBench', 'StepGuard', 'ToolSafe/TS-Bench'],
        'stepguard_benchmark_actions': len(stepguard), 'toolsafe_benchmark_steps': len(toolsafe),
        'annotation_snapshot': silver_summary, 'training_started': False,
        'records_sha256': file_hash(OUT / 'training/records.json'),
        'limits': ['Machine silver and published annotations are not new human gold.',
                   'No truncation or token admission filtering; future training must report length rejections.',
                   'Source annotations may use hindsight; action gold is not injection-event or token gold.',
                   'No internal test split. Original 80/10/10 snapshot remains unchanged but is superseded.',
                   'New judge outputs require a new version; current data and split are fixed.']}
    write(OUT / 'summary.json', summary)
    write(OUT / 'manifest.json', {'script_sha256': file_hash(Path(__file__)),
        'files': {str(p.relative_to(OUT)): {'bytes': p.stat().st_size, 'sha256': file_hash(p)}
                  for p in sorted(OUT.rglob('*')) if p.is_file() and not p.is_symlink()},
        'symlinks': {str(p.relative_to(OUT)): str(p.resolve()) for p in sorted(OUT.rglob('*')) if p.is_symlink()}})
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUT)
    OUT = parser.parse_args().output
    main()
