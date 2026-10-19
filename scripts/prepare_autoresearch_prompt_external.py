"""CPU-only fixed-prompt replay preparation; no inference, cache, or scoring."""
import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import evaluate_whole_dialog_benchmarks as original
from prepare_autoresearch_prompt import INSTRUCTION, TEMPLATE_SHA256, action_span, render, with_instruction
from prepare_whole_dialogs import BASE, REVISION, digest, write

LIMIT = 65536
SUITES = ('atbench', 'rjudge', 'dojo')


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def prepare_case(tok, parent_case):
    case = deepcopy(parent_case)
    require(bool(case['actions']), 'no_actions')
    parent_prefixes = {}
    for seq in parent_case['sequences']:
        require(seq['tokens'] == len(seq['ids']) <= LIMIT, 'invalid_parent_length')
        require(digest(seq['ids']) == seq['ids_sha256'], 'parent_sequence_hash_mismatch')
        require(len(seq['endpoints']) == len(seq['action_indices']), 'parent_endpoint_count_mismatch')
        for index, endpoint in zip(seq['action_indices'], seq['endpoints']):
            require(0 <= index < len(case['actions']) and index not in parent_prefixes,
                    'duplicate_or_invalid_parent_action_index')
            require(0 <= endpoint < seq['tokens'], 'invalid_parent_endpoint')
            parent_prefixes[index] = {'sequence_key': seq['key'],
                'sequence_ids_sha256': seq['ids_sha256'],
                'prefix_ids_sha256': digest(seq['ids'][:endpoint + 1]), 'endpoint': endpoint}
    require(set(parent_prefixes) == set(range(len(case['actions']))), 'missing_parent_action')
    require(parent_case['ids_sha256'] == digest([s['ids_sha256'] for s in parent_case['sequences']]),
            'parent_case_hash_mismatch')
    rendered = []
    for index, action in enumerate(case['actions']):
        state_hash = hashlib.sha256(action['state'].encode()).hexdigest()
        require(state_hash == action['state_sha256'], 'parent_state_hash_mismatch')
        state = json.loads(action['state'])
        # Reproduce the original input before modifying it. Only the explicit state
        # input fields accessed by state_messages enter the template, never labels.
        old_text, old_prefix_ids, messages, tools = original.render(tok, state)
        old_prefix = old_text[:old_text.rfind('<|im_end|>')]
        require(old_prefix == action['native_text'], 'parent_native_text_mismatch')
        require(digest(old_prefix_ids) == parent_prefixes[index]['prefix_ids_sha256'],
                'parent_causal_prefix_mismatch')
        # Each state's history ends BEFORE its proposed action result. A fallback
        # sequence is built from this state alone, never by slicing a later state.
        messages, _ = with_instruction(messages)
        text, ids, offsets = render(tok, messages, tools)
        start, end, prefix_hash = action_span(tok, text, text, ids, offsets)
        require(end < len(ids), 'missing_native_action_closing_tokens')
        action['native_text'] = text[:text.rfind('<|im_end|>')]
        action['parent_native_text_sha256'] = hashlib.sha256(old_prefix.encode()).hexdigest()
        action['parent_prefix_ids_sha256'] = parent_prefixes[index]['prefix_ids_sha256']
        rendered.append({'text': text, 'ids': ids, 'offsets': offsets,
                         'prefix': action['native_text'], 'start': start, 'end': end,
                         'prefix_ids_sha256': prefix_hash})
    full = rendered[-1]
    stable = all(full['text'].startswith(r['prefix']) and
                 full['ids'][:r['end']] == r['ids'][:r['end']] for r in rendered)
    if stable:
        sequences = [{'key': case['key'], 'ids': full['ids'], 'tokens': len(full['ids']),
                      'endpoints': [r['end'] - 1 for r in rendered],
                      'action_indices': list(range(len(rendered)))}]
    else:
        sequences = [{'key': action['key'], 'ids': r['ids'], 'tokens': len(r['ids']),
                      'endpoints': [r['end'] - 1], 'action_indices': [index]}
                     for index, (action, r) in enumerate(zip(case['actions'], rendered))]
    require(max(s['tokens'] for s in sequences) <= LIMIT,
            f'new_whole_case_context_overflow:{max(s["tokens"] for s in sequences)}')
    spans = [None] * len(case['actions'])
    for seq in sequences:
        seq['ids_sha256'] = digest(seq['ids'])
        seq['parent_sequence_ids_sha256'] = {
            parent_prefixes[i]['sequence_key']: parent_prefixes[i]['sequence_ids_sha256']
            for i in seq['action_indices']}
        # Independent full-sequence offsets are used to verify EVERY action span.
        sequence_render = full if stable else rendered[seq['action_indices'][0]]
        for index, endpoint in zip(seq['action_indices'], seq['endpoints']):
            r, action = rendered[index], case['actions'][index]
            start, end, prefix_hash = action_span(tok, r['text'], sequence_render['text'],
                                                 seq['ids'], sequence_render['offsets'])
            require((start, end) == (r['start'], r['end']) and end == endpoint + 1,
                    'full_vs_causal_action_span_mismatch')
            require(prefix_hash == r['prefix_ids_sha256'], 'causal_prefix_hash_mismatch')
            require(tok.convert_tokens_to_ids('<|im_end|>') not in seq['ids'][start:end],
                    'action_body_contains_role_closing_token')
            spans[index] = {'key': action['key'], 'state_sha256': action['state_sha256'],
                'start': start, 'end': end, 'action_tokens': end - start,
                'prefix_ids_sha256': prefix_hash,
                'parent_prefix_ids_sha256': parent_prefixes[index]['prefix_ids_sha256'],
                'parent_sequence_ids_sha256': parent_prefixes[index]['sequence_ids_sha256'],
                'parent_sequence_key': parent_prefixes[index]['sequence_key'],
                'parent_endpoint': parent_prefixes[index]['endpoint']}
    case.update(parent_ids_sha256=parent_case['ids_sha256'],
                parent_evaluation_mode=parent_case['evaluation_mode'],
                parent_tokens=parent_case['tokens'], sequences=sequences,
                ids_sha256=digest([s['ids_sha256'] for s in sequences]),
                tokens=max(s['tokens'] for s in sequences),
                evaluation_mode='whole_dialog' if stable else 'causal_prefix_template_fallback')
    return case, spans


def prepare(args):
    from transformers import AutoTokenizer
    parent = args.parent / args.suite
    folder = args.root / args.suite
    require(not folder.exists(), f'output_already_exists:{folder}')
    parent_audit = original.read(parent / 'audit.json')
    paths = {name: parent / name for name in ('cases.json', 'audit.json', 'shared_rejections.json')}
    hashes = {name: original.sha(path) for name, path in paths.items()}
    require(hashes['cases.json'] == parent_audit['cases_sha256'], 'parent_cases_hash_mismatch')
    require(parent_audit['template_sha256'] == TEMPLATE_SHA256 and parent_audit['no_truncation'],
            'parent_template_or_truncation_policy_mismatch')
    parents = original.read(paths['cases.json'])
    rejected = original.read(paths['shared_rejections.json'])
    require(len(parents) == parent_audit['admitted'] and len(rejected) == parent_audit['rejected'],
            'parent_admission_counts_mismatch')
    keys = [c['key'] for c in parents]
    require(bool(keys) and len(keys) == len(set(keys)), 'duplicate_or_empty_parent_case_keys')
    require(not set(keys).intersection(r['key'] for r in rejected), 'parent_admission_rejection_overlap')
    tok = AutoTokenizer.from_pretrained(BASE, revision=REVISION, local_files_only=True)
    require(tok.is_fast and hashlib.sha256(tok.chat_template.encode()).hexdigest() == TEMPLATE_SHA256,
            'pinned_native_template_mismatch')
    cases, spans = [], {}
    # Strict policy: validate the entire suite before creating output. Any NEW
    # error aborts with its case key; no silently reduced protected evaluation set.
    for case in parents:
        try:
            prepared, action_spans = prepare_case(tok, case)
        except (ValueError, TypeError, KeyError, IndexError) as error:
            raise ValueError(f'PROMPT_EXTERNAL_PREPARATION_ABORTED case={case["key"]}: {error}') from error
        cases.append(prepared)
        spans[case['key']] = action_spans
    require([c['key'] for c in cases] == keys, 'protected_case_set_changed')
    for case, old in zip(cases, parents):
        require([(a['key'], a['state_sha256']) for a in case['actions']] ==
                [(a['key'], a['state_sha256']) for a in old['actions']], 'protected_action_set_changed')
    folder.mkdir(parents=True, exist_ok=False)
    write(folder / 'cases.json', cases)
    write(folder / 'spans.json', spans)
    # Copy inherited exclusions byte-for-byte; ATBench's 19 remain unchanged.
    (folder / 'shared_rejections.json').write_bytes(paths['shared_rejections.json'].read_bytes())
    audit = dict(parent_audit, admitted=len(cases), rejected=len(rejected),
        actions=sum(len(c['actions']) for c in cases),
        cases_sha256=original.sha(folder / 'cases.json'),
        spans_sha256=original.sha(folder / 'spans.json'),
        shared_rejections_sha256=original.sha(folder / 'shared_rejections.json'),
        base=BASE, revision=REVISION, template_sha256=TEMPLATE_SHA256,
        prompt=INSTRUCTION, prompt_sha256=hashlib.sha256(INSTRUCTION.encode()).hexdigest(),
        tokenizer_vocab_sha256=digest(tok.get_vocab()), parent_root=str(args.parent.resolve()),
        parent_files_sha256=hashes, matched_case_ids=keys, matched_case_ids_sha256=digest(keys),
        new_exclusions=[], new_exclusion_policy='strict_abort_before_output',
        evaluation_modes=dict(Counter(c['evaluation_mode'] for c in cases)),
        parent_evaluation_modes=dict(Counter(c['evaluation_mode'] for c in parents)),
        mode_changes=[c['key'] for c in cases if c['evaluation_mode'] != c['parent_evaluation_mode']],
        max_tokens=max(c['tokens'] for c in cases), limit=LIMIT, no_truncation=True,
        input_fields=['state.user_request', 'state.observed_history', 'state.proposed_action',
                      'state.tool_interfaces'], labels_used_only_as_output_metadata=True,
        causal_prefix_gate='independent native state render; character/ID equality; sequence offsets; '
                          'fallback from each original state only',
        preparation_only=True, inference_performed=False, feature_cache_created=False)
    write(folder / 'audit.json', audit)
    print(json.dumps({'suite': args.suite, 'admitted': len(cases), 'rejected': len(rejected),
                      'new_exclusions': 0, 'audit_sha256': original.sha(folder / 'audit.json'),
                      'evaluation_modes': audit['evaluation_modes']}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--suite', choices=SUITES, required=True)
    parser.add_argument('--root', type=Path, required=True)
    prepare(parser.parse_args())
