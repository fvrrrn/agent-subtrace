"""Build maximal verified dialogs, native chat templates and action-token masks."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

BASE = 'Qwen/Qwen3.5-4B-Base'
REVISION = '1001bb4d826a52d1f399e183466143f4da7b741b'


def serial(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def digest(value):
    return hashlib.sha256(serial(value).encode()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def message(event, proposed=False):
    if not isinstance(event, dict):
        raise ValueError('unstructured_event')
    role = event.get('role', event.get('agent_role'))
    if 'agent_id' in event:
        role = {'user': 'user', 'environment': 'tool'}.get(event['agent_id'], 'assistant')
        # Multi-agent identity and routing remain visible, without collapsing their content.
        return {'role': role, 'content': serial(event)}
    role = {'agent': 'assistant', 'environment': 'tool'}.get(role, role)
    if role is None and proposed:
        role = 'assistant'
    if role not in ('system', 'user', 'assistant', 'tool'):
        raise ValueError('unknown_role')
    if 'action' in event or ('name' in event and 'arguments' in event):
        action = event.get('action', event)
        if isinstance(action, str) and 'arguments' in event:
            action = {'name': action, 'arguments': event['arguments']}
        if isinstance(action, dict) and 'name' in action and 'arguments' in action:
            return {'role': role, 'content': event.get('thought', ''),
                    'tool_calls': [{'type': 'function', 'function': action}]}
        return {'role': role, 'content': serial({k: v for k, v in event.items() if k != 'role'})}
    out = {'role': role, 'content': event.get('content', '')}
    if 'tool_calls' in event:
        out['tool_calls'] = json.loads(json.dumps(event['tool_calls']))
        for call in out['tool_calls']:
            fn = call.get('function', call)
            if isinstance(fn.get('arguments'), str):
                fn['arguments'] = json.loads(fn['arguments'])
    if 'tool_call_id' in event:
        out['tool_call_id'] = event['tool_call_id']
    return out


def auxiliary_only(row):
    for origin in row['origins']:
        prov = origin['provenance']
        prov = prov if isinstance(prov, list) else [prov]
        if any(p.get('source', origin['source']) != 'StepShield' or
               p.get('domain') != 'generated_benign' for p in prov):
            return False
    return True


def prepare(args):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(BASE, revision=REVISION, local_files_only=True)
    assert tok.chat_template and tok.is_fast
    args.root.mkdir(parents=True, exist_ok=False)
    rows = json.loads(args.records.read_text())
    groups, rejected = defaultdict(list), []
    for row in rows:
        if auxiliary_only(row):
            rejected.append({'key': row['key'], 'reason': 'auxiliary_generated_benign_only'})
            continue
        state = json.loads(row['state'])
        if not isinstance(state['observed_history'], list):
            rejected.append({'key': row['key'], 'reason': 'opaque_history'})
            continue
        try:
            request = state['user_request']
            user_texts = [e.get('content', '') for e in state['observed_history']
                          if isinstance(e, dict) and
                          (e.get('role') == 'user' or e.get('agent_id') == 'user')]
            authored_metadata = request.startswith('Authored counterfactual scenario:')
            if authored_metadata and not user_texts:
                raise ValueError('no_actual_user_request_after_metadata_removal')
            if user_texts and not authored_metadata and request and not any(request in t for t in user_texts):
                raise ValueError('user_request_history_mismatch')
            history = [message(e) for e in state['observed_history']]
            action = message(state['proposed_action'], proposed=True)
            if action['role'] != 'assistant':
                raise ValueError('proposed_not_assistant')
        except (ValueError, TypeError) as error:
            rejected.append({'key': row['key'], 'reason': str(error)})
            continue
        # Never join on task family alone: every event prefix must match exactly.
        bucket = digest([row['group'], row['split'], state['tool_interfaces'], state['user_request']])
        groups[bucket].append((row, state, history + [action]))
    dialogs, assigned = [], set()
    for bucket, entries in groups.items():
        # A trie of actual event sequences distinguishes alternative executions.
        trie = {'children': {}, 'labels': []}
        for row, state, messages in entries:
            node = trie
            for event in messages:
                node = node['children'].setdefault(serial(event), {'children': {}, 'labels': []})
            node['labels'].append(row)
        leaves = []
        def visit(node, path, annotations):
            annotations = annotations + [(len(path)-1, r) for r in node['labels']]
            if not node['children']:
                leaves.append((path, annotations))
            for event, child in sorted(node['children'].items()):
                visit(child, path + [json.loads(event)], annotations)
        visit(trie, [], [])
        # Shared prefix labels occur only once in the loss, even with branched traces.
        for history, annotations in sorted(leaves, key=lambda x: (-len(x[0]), digest(x[0]))):
            annotations = [(i, r) for i, r in annotations if r['key'] not in assigned]
            if not annotations:
                continue
            state = entries[0][1]
            tools = state['tool_interfaces']
            system = []
            if isinstance(tools, list):
                native_tools = tools
            else:
                native_tools = None
                if tools:
                    system = [{'role': 'system', 'content': tools if isinstance(tools, str) else serial(tools)}]
            # Respect an actual initial user event; no synthetic user repeats when it exists.
            if (not history or history[0]['role'] != 'user') and not state['user_request'].startswith('Authored counterfactual scenario:'):
                system.append({'role': 'user', 'content': state['user_request']})
            messages = system + history
            try:
                rendered = tok.apply_chat_template(messages, tools=native_tools, tokenize=False,
                                                   add_generation_prompt=False)
                tokenized = tok(rendered, add_special_tokens=False, return_offsets_mapping=True)
                ids, offsets = tokenized['input_ids'], tokenized['offset_mapping']
                assert ids == tok.apply_chat_template(messages, tools=native_tools, tokenize=True,
                                                     add_generation_prompt=False, return_dict=False)
                spans = []
                for index, row in annotations:
                    stop = len(system) + index + 1
                    prefix = tok.apply_chat_template(messages[:stop], tools=native_tools, tokenize=False,
                                                     add_generation_prompt=False)
                    # Native templates may rewrite old reasoning on later user turns.
                    # Reject instead of silently leaking future-turn structure or editing template.
                    marker = '<|im_start|>assistant\n'
                    start_char = prefix.rfind(marker) + len(marker)
                    end_char = prefix.rfind('<|im_end|>')
                    if start_char < len(marker) or end_char <= start_char:
                        raise ValueError('missing_action_span')
                    if not rendered.startswith(prefix[:end_char]):
                        raise ValueError('native_template_rewrites_causal_prefix')
                    body = rendered[start_char:end_char]
                    empty_think = '<think>\n\n</think>\n\n'
                    if body.startswith(empty_think):
                        start_char += len(empty_think)
                    indices = [j for j, (a, b) in enumerate(offsets) if b > a and a >= start_char and b <= end_char]
                    if not indices or indices != list(range(indices[0], indices[-1]+1)):
                        raise ValueError('noncontiguous_action_tokens')
                    prefix_ids = tok(prefix[:end_char], add_special_tokens=False)['input_ids']
                    if ids[:indices[-1]+1] != prefix_ids:
                        raise ValueError('prefix_token_boundary_changed')
                    spans.append({'start': indices[0], 'end': indices[-1]+1, 'label': row['label'],
                                  'key': row['key'], 'state_sha256': hashlib.sha256(row['state'].encode()).hexdigest()})
                spans.sort(key=lambda x: x['start'])
                if any(a['end'] > b['start'] for a,b in zip(spans, spans[1:])):
                    raise ValueError('duplicate_action_annotation')
                if len(ids) > args.limit:
                    raise ValueError('whole_dialog_context_overflow')
            except (ValueError, TypeError) as error:
                rejected.extend({'key': row['key'], 'reason': str(error), 'dialog_group': bucket}
                                for _, row in annotations)
                continue
            keys = [r['key'] for _, r in annotations]
            assigned.update(keys)
            dialogs.append({'key': digest([bucket, messages]), 'group': entries[0][0]['group'],
                'split': entries[0][0]['split'], 'messages': messages, 'tools': native_tools,
                'ids': ids, 'ids_sha256': digest(ids), 'spans': spans, 'tokens': len(ids),
                'sources': sorted({s for _, r in annotations for s in r['sources']})})
    assert len(assigned) == sum(len(d['spans']) for d in dialogs)
    assert not ({d['group'] for d in dialogs if d['split']=='train'} &
                {d['group'] for d in dialogs if d['split']=='validation'})
    write(args.root / 'dialogs.json', dialogs)
    write(args.root / 'rejections.json', rejected)
    summary = {'dialogs': len(dialogs), 'annotated_actions': len(assigned),
        'input_actions': len(rows), 'rejected_unique_actions': len(set(r['key'] for r in rows)-assigned),
        'counts': dict(Counter(d['split'] for d in dialogs)),
        'action_counts': dict(Counter(f'{d["split"]}:{s["label"]}' for d in dialogs for s in d['spans'])),
        'multi_action_dialogs': sum(len(d['spans'])>1 for d in dialogs),
        'multiple_unsafe_dialogs': sum(sum(s['label'] for s in d['spans'])>1 for d in dialogs),
        'max_tokens': max(d['tokens'] for d in dialogs),
        'reasons': dict(Counter(r['reason'] for r in rejected)),
        'base': BASE, 'revision': REVISION, 'template_sha256': hashlib.sha256(tok.chat_template.encode()).hexdigest(),
        'source_records_sha256': hashlib.sha256(args.records.read_bytes()).hexdigest(),
        'dialogs_sha256': hashlib.sha256((args.root/'dialogs.json').read_bytes()).hexdigest(),
        'limit': args.limit, 'causal_prefix_audit': 'all admitted supervised endpoints; native template and token IDs',
        'loss': 'linear curve per labeled action, anchor previous observed label or zero initially; unknown context unlabelled',
        'mono': 'weight0.1 margin0.2 raw endpoint logodds on adjacent labeled unsafe actions in a dialog; one forward',
        'no_prefix_unroll': True, 'no_truncation': True}
    write(args.root / 'data_audit.json', summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--records', type=Path, required=True)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--limit', type=int, default=16384)
    prepare(p.parse_args())
