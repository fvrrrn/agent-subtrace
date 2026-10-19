"""CPU-only StepGuard inventory and quarantined candidates; never grants admission."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re

from build_safety_candidates import family, normalized


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(',', ':')).encode()).hexdigest()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def main(args):
    source = args.root / 'datasets/corpus-v1/raw/ninty-seven--stepguard_data'
    manifest = json.loads((source / 'SOURCE_MANIFEST.json').read_text())
    for entry in manifest['files']:
        assert sha(source / entry['path']) == entry['sha256'], entry['path']
    args.output.mkdir(parents=True, exist_ok=False)
    metadata_dir = args.root / 'runs/autoresearch-sixhour-v1/data'
    action_meta = json.loads((metadata_dir / 'action_metadata_audit.json').read_text())
    holdouts = {}
    inputs = {str(source / 'SOURCE_MANIFEST.json'): sha(source / 'SOURCE_MANIFEST.json'),
              str(metadata_dir / 'action_metadata_audit.json'): sha(metadata_dir / 'action_metadata_audit.json')}
    for split in ('dev_select', 'dev_confirm'):
        path = metadata_dir / (split + '.manifest.json')
        # Only rows metadata; never access counts, labels, or dev split data files.
        rows = json.loads(path.read_text())['rows']
        keys = {key for row in rows for key in row['action_keys']}
        holdouts[split] = {
            'ids': keys | {row['key'] for row in rows},
            'requests': {action_meta[key]['request_sha256'] for key in keys},
            'request_families': {v for row in rows for v in row['request_families']},
            'groups': {row['original_group'] for row in rows},
            'inputs': {action_meta[key]['input_sha256'] for key in keys}}
        inputs[str(path)] = sha(path)
    toolsafe_path = args.root / 'datasets/data-bundle-whole-dialog-v1/benchmarks/toolsafe/records.jsonl'
    protected_requests, protected_families = set(), set()
    for line in toolsafe_path.open():
        state = json.loads(json.loads(line)['state'])
        request = state['user_request']
        protected_requests.add(digest(normalized(request)))
        protected_families.add(digest(family(request)))
    inputs[str(toolsafe_path)] = sha(toolsafe_path)
    old_path = args.root / 'archive/legacy-shm/danefedov-controlled-monotonic-verified-v1/results/records.json'
    old = {r['key']: r for r in json.loads(old_path.read_text()) if r['split'].startswith('external_')}
    inputs[str(old_path)] = sha(old_path)
    inventory, candidates = [], []
    for relative in ('sft3k/agentguard_sft3k_sharegpt.jsonl', 'rl4k/agentguard_rl4k_grpo.jsonl'):
        for index, line in enumerate((source / relative).open()):
            raw = json.loads(line)
            if 'conversations' in raw:
                prompt = raw['conversations'][0]['value']
                response = raw['conversations'][1]['value']
                match = re.findall(r'<Judgment>\s*(safe|unsafe)\s*</Judgment>', response)
                label = match[0] if len(match) == 1 else None
                level = 'action' if '<<Action Under Review>>' in prompt else 'trajectory'
                unsafe_step = re.findall(r'<UnsafeStep>(.*?)</UnsafeStep>', response)
            else:
                prompt, label, level = raw['instruction'], raw['label'], raw['level']
                assert raw['output'] == label
                unsafe_step = []
            key = f'stepguard:{relative}:{index}'
            row = {'key': key, 'source_id': raw.get('id'), 'file': relative, 'line_1based': index + 1,
                   'official_split': 'train', 'level': level, 'original_label': label,
                   'raw_row_sha256': digest(raw), 'prompt_sha256': digest(prompt),
                   'unsafe_step_annotation': unsafe_step,
                   'top_level_fields': sorted(raw), 'original_family_id': None,
                   'reasons': ['missing_row_provenance_and_original_family_ids'],
                   'admitted': False}
            if level != 'action':
                row['reasons'].append('trajectory_label_not_current_action_gold')
            else:
                match = re.fullmatch(r'.*?### Input\s*<<User Request>>\s*(.*?)\s*'
                    r'<<Interaction History>>\s*(.*?)\s*<<Action Under Review>>\s*(.*?)\s*'
                    r'<<Available Tools>>\s*(.*)', prompt, re.S)
                if not match:
                    row['reasons'].append('unrecognized_action_serialization')
                else:
                    request, history, action, tools = match.groups()
                    state = dict(tool_interfaces=tools, user_request=request,
                                 observed_history=history, proposed_action=action)
                    row.update(input_sha256=digest(state), request_sha256=digest(normalized(request)),
                               request_family_sha256=digest(family(request)))
                    if '[ENVIRONMENT]' in action or '[Observation' in action:
                        row['reasons'].append('possible_current_action_result_in_candidate')
                    if row['request_family_sha256'] in protected_families:
                        row['reasons'].append('protected_toolsafe_request_family')
                    if row['request_sha256'] in protected_requests:
                        row['reasons'].append('protected_toolsafe_exact_normalized_request')
                    for split, meta in holdouts.items():
                        if (key in meta['ids'] or row['request_sha256'] in meta['requests']
                                or row['request_family_sha256'] in meta['request_families']
                                or row['input_sha256'] in meta['inputs']):
                            row['reasons'].append('overlap_' + split)
                    candidates.append({'key': key, 'state': state, 'label': {'safe': 0, 'unsafe': 1}.get(label),
                        'request_family_sha256': row['request_family_sha256'], 'input_sha256': row['input_sha256'],
                        'original_family_id': None, 'admitted': False})
            inventory.append(row)
    by_input = defaultdict(list)
    for row in inventory:
        if 'input_sha256' in row:
            by_input[row['input_sha256']].append(row)
    contradictions = []
    for input_hash, rows in by_input.items():
        if len({row['original_label'] for row in rows}) > 1:
            contradictions.append({'input_sha256': input_hash, 'keys': [r['key'] for r in rows]})
            for row in rows:
                row['reasons'].append('contradictory_identical_causal_input_both_sides')
    current = {row['key']: row for row in inventory}
    old_coverage = []
    for key, row in old.items():
        candidate = current.get(key)
        old_coverage.append({'key': key, 'old_split': row['split'], 'found_in_official_train': candidate is not None,
            'state_matches': candidate is not None and digest(json.loads(row['state'])) == candidate.get('input_sha256'),
            'label_matches': candidate is not None and row['label'] == {'safe': 0, 'unsafe': 1}.get(candidate['original_label'])})
    for name, rows in [('inventory', inventory), ('candidates_NOT_ADMITTED', candidates)]:
        with (args.output / (name + '.jsonl')).open('w') as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    write(args.output / 'contradictions.json', contradictions)
    write(args.output / 'old_evaluation_coverage.json', old_coverage)
    summary = {'status': 'BLOCKED_PROVENANCE_NOT_TRAINING_RELEASE', 'revision': manifest['revision'],
        'official_splits': manifest['card_metadata']['configs'], 'license': manifest['card_metadata']['license'],
        'rows': len(inventory), 'levels': dict(Counter(r['level'] for r in inventory)),
        'labels': dict(Counter(r['original_label'] for r in inventory)),
        'candidate_actions': len(candidates), 'admitted_actions': 0,
        'reasons': dict(Counter(reason for row in inventory for reason in row['reasons'])),
        'contradictory_input_groups': len(contradictions),
        'old_evaluation': dict(Counter(r['old_split'] for r in old_coverage)),
        'old_coverage_all_exact': all(r['state_matches'] and r['label_matches'] for r in old_coverage),
        'input_files': inputs, 'source_files': manifest['files'], 'script_sha256': sha(Path(__file__)),
        'holdout_check': 'IDs, normalized requests, request families and causal input hashes; manifests metadata only.',
        'limits': ['Original source-family IDs absent; request families are proxies, not exact original groups.',
                   'ToolSafe overlap checked, ATBench/RJudge row-level provenance not cleared; all rows blocked.',
                   'No token offsets or evidence spans in released schemas. Analysis prose is not token gold.',
                   'Trajectory UnsafeStep annotation is an action index, not token evidence or causal action gold.',
                   'No native tokenization or whole-dialog admission performed while provenance is unresolved.']}
    write(args.output / 'inventory_summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/var/local/danefedov/experiments/agent-safety'))
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args())
