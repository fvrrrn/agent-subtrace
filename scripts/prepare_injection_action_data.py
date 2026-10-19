"""Stage published action labels separately; do not train, relabel or infer injection gold."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from build_safety_candidates import digest, family, normalized

ROOT = Path('/var/local/danefedov/experiments/agent-safety')
VISIBLE = ('surface', 'agent_id', 'agent_role', 'tool_name', 'tool_args',
           'tool_result', 'content', 'target_agent', 'target_role')
INJECTION_SOURCES = {'untrusted_content', 'compromised_tool', 'poisoned_memory',
                     'peer_agent', 'handoff_corruption'}


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        while chunk := stream.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def write_line(stream, value):
    stream.write(json.dumps(value, ensure_ascii=False) + '\n')


def visible(event, current=False):
    return {k: event[k] for k in VISIBLE if event.get(k) is not None
            and not (current and k == 'tool_result')}


def record(source, key, tools, user, history, action, label, event_type,
           trajectory, segment, provenance, injection_context, reason):
    # Insertion order is the model-visible order; never alphabetically render storage keys.
    state = {'tool_interfaces': tools, 'user_request': user,
             'observed_history': history, 'proposed_action': action}
    return {'key': key, 'source': source, 'state': json.dumps(state, ensure_ascii=False),
            'input_sha256': digest(state), 'label': label, 'label_scope': 'action_safety',
            'event_type': event_type, 'trajectory': trajectory, 'segment': segment,
            'split': 'unassigned', 'mechanism': 'published_action_annotation',
            'previous_unsafe': None, 'previous_safe': None,
            'injection_context': injection_context, 'injection_context_basis': reason,
            'injection_event_label': None, 'provenance': provenance}


def toolsafe(corpus):
    raw = corpus / 'raw/MurrayTom--ToolSafe'
    revision = json.loads((raw / 'SOURCE_MANIFEST.json').read_text())['revision']
    base = raw / 'snapshot/TS-Bench'
    for path in sorted(base.rglob('*.json')):
        relative = str(path.relative_to(base))
        file_hash = sha256(path)
        for i, row in enumerate(json.loads(path.read_text())):
            score = row['score']
            label = int(score) if score in (0, 1) else None
            if score not in (0, .5, 1):
                raise ValueError(f'Unexpected ToolSafe score: {relative}:{i}: {score}')
            # This is suite metadata, not a new action/exposure/span annotation.
            context = relative.startswith(('agentdojo-traj/', 'asb-traj/'))
            action = row['current_action']
            event_type = 'tool_call' if ('Action Input:' in action and 'Action:' in action) else 'agent_action'
            r = record('ToolSafe', f'toolsafe:{relative}:{i}', row['env_info'],
                       row['instruction'], row['history'], action, label, event_type,
                       f'toolsafe:{relative}:{row["id-interaction"]}', row['id-segment'],
                       {'repository': 'MurrayTom/ToolSafe', 'revision': revision,
                        'file': relative, 'file_sha256': file_hash, 'row': i,
                        'original_score': score, 'id_interaction': row['id-interaction'],
                        'id_segment': row['id-segment'], 'source_partition': 'published_evaluation',
                        'label_quality': 'published_step_annotation'}, context,
                       'published_prompt_injection_benchmark_suite' if context else None)
            yield r, row['instruction'], [r['trajectory']]


def stepsafe(corpus):
    import pyarrow.parquet as pq
    raw = corpus / 'raw/carriegu0818--StepSafe'
    revision = json.loads((raw / 'SOURCE_MANIFEST.json').read_text())['revision']
    base = corpus / 'derived/stepsafe_rebuilt/rl'
    tasks = {r['env_kwargs']['audit_id']: r['env_kwargs']
             for r in pq.read_table(base / 'train.parquet').to_pylist()}
    annotations = base / 'trajectory_annotations.jsonl'
    file_hash = sha256(annotations)
    for line_index, line in enumerate(annotations.open()):
        row = json.loads(line)
        if row['audit_id'] not in tasks:
            continue  # Same canonical rebuilt RL-task admission as the corpus builder.
        task = tasks[row['audit_id']]
        events = row['trajectory']
        if isinstance(events, str):
            events = json.loads(events)
        sequences = [e['sequence_num'] for e in events]
        if len(set(sequences)) != len(sequences) or sequences != sorted(sequences):
            raise ValueError(f'Non-unique or unordered events: {row["audit_id"]}')
        labels = {a['sequence_num']: a for a in row['step_labels']}
        if len(labels) != len(row['step_labels']) or not set(labels) <= set(sequences):
            raise ValueError(f'Invalid annotation alignment: {row["audit_id"]}')
        user = task.get('goal') or next((e.get('content', '') for e in events
                                        if e.get('agent_role') == 'user'), '')
        tools = task.get('profile') or row.get('profile') or task.get('role_prompts') or ''
        links = ['stepsafe:' + str(row[k]) for k in ('audit_id', 'contrastive_pair_id',
                 'perturbation_pair_id', 'factorial_family_id', 'root_source_audit_id',
                 'source_audit_id') if row.get(k)]
        history = []
        for event in events:
            seq = event['sequence_num']
            if seq in labels:
                annotation = labels[seq]
                target, violation = annotation.get('current_action_target'), annotation.get('policy_violated')
                label = (1 if target == 'STOP' and violation is True else
                         0 if target == 'CONTINUE' and violation is False else None)
                context = row.get('risk_source') in INJECTION_SOURCES
                # Annotation presence defines reviewed actors, including named peer roles.
                if event.get('surface') not in ('tool_call', 'communication'):
                    raise ValueError(f'Unexpected reviewed surface: {row["audit_id"]}:{seq}')
                r = record('StepSafe', f'stepsafe:{row["audit_id"]}:{seq}', tools, user,
                           history, visible(event, current=True), label,
                           'tool_call' if event['surface'] == 'tool_call' else 'agent_message',
                           'stepsafe:' + row['audit_id'], seq,
                           {'repository': 'carriegu0818/StepSafe', 'revision': revision,
                            'file': 'rebuilt/rl/trajectory_annotations.jsonl', 'file_sha256': file_hash,
                            'row': line_index, 'audit_id': row['audit_id'],
                            'source_partition': row.get('split'),
                            'original_current_action_target': target, 'original_policy_violated': violation,
                            'risk_source': row.get('risk_source'),
                            'step_risk_source': annotation.get('risk_source'),
                            'audit_method': annotation.get('audit_method'),
                            'label_quality': 'published_current_action_annotation_hindsight_possible'},
                           context, 'source_trajectory_risk_category_not_verified_injection_span' if context else None)
                # Serialize now, before adding the reviewed event or its later observation.
                yield r, user, links
            history = history + [visible(event)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--output-name', default='injection-actions-v1')
    args = parser.parse_args()
    corpus = args.root / 'datasets/corpus-v1'
    out = args.root / 'datasets' / args.output_name
    out.mkdir(parents=True, exist_ok=False)
    parent = {}

    def find(key):
        parent.setdefault(key, key)
        root = key
        while parent[root] != root:
            root = parent[root]
        while key != root:
            key, parent[key] = parent[key], root
        return root

    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[max(a, b)] = min(a, b)

    known = json.loads((corpus / 'holdout_blocklist.json').read_text())
    requests, blocked, duplicates = {}, set(), defaultdict(list)
    source_counts = Counter()
    spool = out / 'source_records.jsonl'
    with spool.open('w') as stream:
        for generator in (toolsafe, stepsafe):
            for r, user, links in generator(corpus):
                request_id = digest(family(user))
                node = 'request:' + request_id
                for link in links:
                    union(node, 'family:' + link)
                r['request_family'] = request_id
                if (digest(normalized(user)) in known['request_hashes'] or
                        request_id in known['request_family_hashes']):
                    blocked.add(node)
                requests[request_id] = user
                duplicates[r['input_sha256']].append({'key': r['key'], 'label': r['label']})
                write_line(stream, r)
                source_counts[r['source']] += 1
    # Existing lexical near-overlap procedure, reused without changing labels.
    from audit_candidate_overlap import shingles
    from datasketch import MinHash, MinHashLSH
    evaluations = json.loads((corpus / 'evaluation_requests.json').read_text())
    index = MinHashLSH(threshold=.75, num_perm=64)
    stored, near_matches = {}, []
    for i, text in enumerate(evaluations):
        s = shingles(text)
        m = MinHash(num_perm=64, seed=20261018)
        m.update_batch([v.encode() for v in sorted(s)])
        stored[str(i)] = s
        index.insert(str(i), m)
    for request_id, text in requests.items():
        s = shingles(text)
        m = MinHash(num_perm=64, seed=20261018)
        m.update_batch([v.encode() for v in sorted(s)])
        for match in index.query(m):
            t = stored[match]
            jaccard = len(s & t) / len(s | t)
            if jaccard >= .85:
                blocked.add('request:' + request_id)
                near_matches.append({'request_family': request_id, 'evaluation_index': int(match),
                                     'jaccard': jaccard})
    blocked = {find(key) for key in blocked}
    conflicts = {h for h, rows in duplicates.items()
                 if len({r['label'] for r in rows if r['label'] is not None}) > 1}
    counts, source_stats = Counter(), defaultdict(Counter)
    handles = {}
    for source in source_counts:
        folder = out / source.lower()
        folder.mkdir()
        handles[source] = {name: (folder / (name + '.jsonl')).open('w')
                           for name in ('records', 'injection_context_records', 'quarantine')}
    seen = set()
    try:
        for line in spool.open():
            r = json.loads(line)
            r['group'] = digest(find('request:' + r.pop('request_family')))
            reasons = []
            if r['label'] is None:
                reasons.append('ambiguous_or_intervention_only_label')
            if find('request:' + digest(family(json.loads(r['state'])['user_request']))) in blocked:
                reasons.append('evaluation_request_overlap_family')
            if r['input_sha256'] in conflicts:
                reasons.append('conflicting_published_labels')
            duplicate = r['label'] is not None and r['input_sha256'] in seen
            if r['label'] is not None:
                seen.add(r['input_sha256'])
            if duplicate:
                reasons.append('exact_duplicate_input')
            r['status'] = 'quarantined' if reasons else 'formatted_not_released_for_training'
            r['quarantine_reasons'] = reasons
            stream_name = 'quarantine' if reasons else 'records'
            write_line(handles[r['source']][stream_name], r)
            counts[stream_name] += 1
            source_stats[r['source']][stream_name] += 1
            for reason in reasons:
                counts[reason] += 1
            if not reasons:
                source_stats[r['source']][f'label_{r["label"]}'] += 1
                if r['injection_context']:
                    write_line(handles[r['source']]['injection_context_records'], r)
                    source_stats[r['source']]['injection_context_records'] += 1
    finally:
        for streams in handles.values():
            for stream in streams.values():
                stream.close()
    write_json(out / 'overlap_audit.json', {'near_matches': near_matches,
               'blocked_components': len(blocked), 'evaluation_request_count': len(evaluations),
               'method': 'Exact normalized/masked request plus 64-permutation MinHash retrieval and exact 5-word Jaccard >=0.85; exclude whole connected families.',
               'limitation': 'Lexical audit, not proof of semantic independence.'})
    write_json(out / 'duplicates.json', {h: rows for h, rows in duplicates.items() if len(rows) > 1})
    # The previously recorded AgentDrift snapshot was not found; public endpoint is 401.
    write_json(out / 'agentdrift_status.json', {
        'status': 'unavailable_not_converted', 'repository': 'Asif-0209/AgentDrift',
        'historical_revision': '014a514fa998b4ac4519579fceb8a5884b379bda',
        'reason': 'No local/server snapshot found; HF metadata and README requests returned HTTP 401. No credentials sought.',
        'mapping_policy': 'Never map hijacked to unsafe or injection point to unsafe action automatically.'})
    write_json(out / 'summary.json', {'source_records': dict(source_counts), 'counts': dict(counts),
               'sources': {k: dict(v) for k, v in source_stats.items()},
               'training_started': False, 'training_release': False, 'split': 'unassigned',
               'label_scope': 'action_safety', 'agentdrift': 'unavailable_not_converted',
               'limitations': ['Published evaluation data repackaged, not an official training split.',
                   'Injection-context subset uses source suite/trajectory categories, not new exposure or injection-event gold.',
                   'No injection-event labels or token targets invented; all such labels remain null.',
                   'Partial action supervision does not label every event. No truncation, no unsafe monotonic pairs inferred.',
                   'StepSafe source annotations may use hindsight. Conversion is not semantic adjudication.',
                   'Named reviewed multi-agent roles preserved; current result omitted and history strictly precedes action.',
                   'Do not evaluate on the same benchmark families after training on these records.']})
    write_json(out / 'manifest.json', {'files': {str(p.relative_to(out)): {'sha256': sha256(p), 'bytes': p.stat().st_size}
               for p in sorted(out.rglob('*')) if p.is_file()}, 'script_sha256': sha256(Path(__file__))})
    print((out / 'summary.json').read_text(), flush=True)


if __name__ == '__main__':
    main()
