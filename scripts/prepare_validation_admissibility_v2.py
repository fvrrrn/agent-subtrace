"""CPU-only prospective validation admission; never reads predictions or confirm labels."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import shutil

from prepare_whole_dialogs import message, serial, digest, BASE, REVISION


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def counts(rows):
    return {'dialogs': len(rows), 'actions': sum(len(d['spans']) for d in rows),
            'labels': dict(Counter(str(s['label']) for d in rows for s in d['spans']))}


def main(args):
    from transformers import AutoTokenizer
    args.output.mkdir(parents=True, exist_ok=False)
    tok = AutoTokenizer.from_pretrained(BASE, revision=REVISION, local_files_only=True)
    splits = json.loads((args.source / 'splits.json').read_text())
    names = ('dev_select', 'original_validation')
    datasets = {n: json.loads((args.source / (n + '.json')).read_text()) for n in names}
    keys = {s['key'] for rows in datasets.values() for d in rows for s in d['spans']}
    # Only selected records are inspected. Other records' labels are never accessed.
    records = {r['key']: r for r in json.loads(args.records.read_text()) if r['key'] in keys}
    prefixes = defaultdict(list)
    issues = []
    audited = []
    def issue(n, d, s, reason, detail=None):
        issues.append({'split': n, 'dialog_key': d['key'], 'action_key': s['key'] if s else None,
                       'reason': reason, 'detail': detail})
    for n, rows in datasets.items():
        assert [d['key'] for d in rows] == splits[n]
        for d in rows:
            if digest(d['ids']) != d['ids_sha256'] or len(d['ids']) != d['tokens']:
                issue(n, d, None, 'stored_token_integrity_failure')
            full = tok.apply_chat_template(d['messages'], tools=d['tools'], tokenize=False,
                                           add_generation_prompt=False)
            if tok(full, add_special_tokens=False)['input_ids'] != d['ids']:
                issue(n, d, None, 'native_serialization_ids_mismatch')
            for s in d['spans']:
                r = records.get(s['key'])
                if r is None:
                    issue(n, d, s, 'missing_source_record')
                    continue
                evidence = {'split': n, 'dialog_key': d['key'], 'action_key': s['key'],
                            'label': s['label'], 'start': s['start'], 'end': s['end'],
                            'source_record_sha256': digest(r), 'origins': r['origins']}
                audited.append(evidence)
                if not 0 <= s['start'] < s['end'] <= len(d['ids']):
                    issue(n, d, s, 'invalid_action_span')
                    continue
                prefix = d['ids'][:s['end']]
                evidence['causal_ids_sha256'] = digest(prefix)
                prefixes[digest(prefix)].append((evidence, prefix))
                if hashlib.sha256(r['state'].encode()).hexdigest() != s['state_sha256'] or r['label'] != s['label']:
                    issue(n, d, s, 'source_state_or_label_mismatch')
                state = json.loads(r['state'])
                try:
                    history = [message(e) for e in state['observed_history']]
                    # The original trie stores serial(event) and reloads it: key order
                    # matters to native tool-argument rendering despite dict equality.
                    expected = [json.loads(serial(e)) for e in history + [message(state['proposed_action'], proposed=True)]]
                    tools = state['tool_interfaces']
                    system = []
                    native_tools = tools if isinstance(tools, list) else None
                    if native_tools is None and tools:
                        system.append({'role': 'system', 'content': tools if isinstance(tools, str) else serial(tools)})
                    if (not history or history[0]['role'] != 'user') and not state['user_request'].startswith('Authored counterfactual scenario:'):
                        system.append({'role': 'user', 'content': state['user_request']})
                    expected = system + expected
                    if d['messages'][:len(expected)] != expected or d['tools'] != native_tools:
                        issue(n, d, s, 'source_history_action_or_tools_not_preserved')
                    rendered = tok.apply_chat_template(expected, tools=native_tools, tokenize=False,
                                                       add_generation_prompt=False)
                    end = rendered.rfind('<|im_end|>')
                    if end < 0 or tok(rendered[:end], add_special_tokens=False)['input_ids'] != prefix:
                        issue(n, d, s, 'causal_source_prefix_not_exact')
                    if 'tool_result' in state['proposed_action']:
                        issue(n, d, s, 'current_action_result_in_input')
                except (ValueError, TypeError, KeyError) as error:
                    issue(n, d, s, 'source_serialization_failure', str(error))
    contradictions = []
    for h, members in prefixes.items():
        if len({e['label'] for e, _ in members}) > 1:
            assert all(ids == members[0][1] for _, ids in members)
            contradictions.append({'causal_ids_sha256': h, 'ids': members[0][1],
                                   'members': [e for e, _ in members]})
            for e, _ in members:
                issues.append({'split': e['split'], 'dialog_key': e['dialog_key'],
                               'action_key': e['action_key'], 'reason': 'identical_causal_ids_opposing_labels',
                               'detail': h})
    excluded = {n: {i['dialog_key'] for i in issues if i['split'] == n} for n in names}
    summary = {}
    for n, rows in datasets.items():
        keep = [d for d in rows if d['key'] not in excluded[n]]
        removed = [d for d in rows if d['key'] in excluded[n]]
        splits[n] = [d['key'] for d in keep]
        write(args.output / (n + '.json'), keep)
        write(args.output / (n + '.excluded.json'), removed)
        summary[n] = {'before': counts(rows), 'after': counts(keep), 'excluded': counts(removed)}
    for n in ('train_small', 'train_full', 'dev_confirm'):
        # Opaque byte copy, particularly dev_confirm: no parsing or label access.
        shutil.copyfile(args.source / (n + '.json'), args.output / (n + '.json'))
    write(args.output / 'splits.json', splits)
    write(args.output / 'issues.json', issues)
    write(args.output / 'contradictions.json', contradictions)
    write(args.output / 'audited_actions.json', audited)
    write(args.output / 'excluded_source_records.json', [records[k] for k in sorted({s['key'] for n, rows in datasets.items() for d in rows if d['key'] in excluded[n] for s in d['spans']})])
    protocol = {'version': 'validation_admissibility_v2', 'deadline_unix': 1792348785,
                'selection': ['maximum dev_select AP', 'minimum event BCE', 'minimum compute cost'],
                'admission': 'Whole dialog excluded if any supervised action has objective defect; both labels of identical causal-input contradictions removed across both audited splits.',
                'train': 'unchanged; known prior full-train 39 conflicting actions/19 groups and small-train 6/3 retained',
                'confirm': 'unchanged opaque copy; labels not inspected',
                'limitations': ['Previously inspected dev is not independent held-out evaluation.',
                               'No semantic relabeling; ambiguities are not automatic exclusions.',
                               'Source-to-message equality and native causal IDs are checked; this does not establish that source context itself is complete.',
                               'Same admission required across new models; additional tokenizer failures require explicit common-case admission.',
                               'If exclusions occur, prior comparisons are not directly matched.'],
                'oldval_trainer_extension_required': bool(excluded['original_validation']),
                'no_predictions_or_scores_read': True, 'no_new_annotation': True}
    write(args.output / 'protocol.json', protocol)
    manifest = {'summary': summary, 'contradiction_groups': len(contradictions),
                'issue_counts': dict(Counter(i['reason'] for i in issues)),
                'source_hashes': {str(p): sha(p) for p in [args.records, args.original_dialogs,
                    Path(__file__), Path(__file__).with_name('prepare_whole_dialogs.py'),
                    *[args.source / (n + '.json') for n in splits], args.source / 'splits.json']},
                'output_hashes': {p.name: sha(p) for p in sorted(args.output.glob('*.json'))},
                'template_sha256': hashlib.sha256(tok.chat_template.encode()).hexdigest(),
                'tokenizer': {'base': BASE, 'revision': REVISION}}
    write(args.output / 'manifest.json', manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--records', type=Path, required=True)
    p.add_argument('--original-dialogs', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    main(p.parse_args())
