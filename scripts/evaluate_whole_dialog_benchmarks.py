"""Fixed whole-dialog Qwen4B checkpoints and native Kev4B on protected benchmarks."""
import argparse
from contextlib import nullcontext
from collections import Counter
import gc
import hashlib
import json
import os
from pathlib import Path
import select
import time

from prepare_whole_dialogs import BASE, REVISION, message, serial, digest, write

STORE = Path('/var/local/danefedov/experiments/agent-safety')
KEV = 'jaredpalmer/kev-4b@6cfce5c2fa4b4bd64026336ab649c5ca78857d52'
ARMS = ('frozen_linear', 'frozen_linear_mono', 'lora_linear', 'lora_linear_mono', 'kev4b')
SUITES = ('dojo', 'harm', 'stepguard', 'atbench', 'rjudge')


def read(p):
    return json.loads(p.read_text())


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def flatten(rows):
    for row in rows:
        if isinstance(row, list):
            yield from flatten(row)
        else:
            yield row


def state_messages(s):
    tools = s['tool_interfaces']
    native_tools = tools if isinstance(tools, list) else None
    messages = []
    if tools and native_tools is None:
        messages.append({'role': 'system', 'content': tools if isinstance(tools, str) else serial(tools)})
    history = s['observed_history']
    if isinstance(history, list):
        # chronology.prefixes wraps original payloads with event IDs; these
        # wrappers are not messages and must not replace their actual contents.
        events = [dict(e['payload'], role=e['role']) if 'payload' in e else e for e in history]
        hist = [message(e) for e in events]
        if not hist or hist[0]['role'] != 'user':
            messages.append({'role': 'user', 'content': s['user_request']})
        messages.extend(hist)
    elif isinstance(history, str):
        # Opaque source history has no reliable role boundaries. Preserve it as
        # supplied context, not fabricated assistant/tool turns or safe labels.
        content = s['user_request']
        if history:
            content = serial({'user_request': content, 'observed_history': history})
        messages.append({'role': 'user', 'content': content})
    else:
        raise ValueError('unsupported_history')
    action = s['proposed_action']
    messages.append(message(action, proposed=True) if isinstance(action, dict)
                    else {'role': 'assistant', 'content': action})
    return messages, native_tools


def render(tok, s):
    messages, tools = state_messages(s)
    text = tok.apply_chat_template(messages, tools=tools, tokenize=False, add_generation_prompt=False)
    end = text.rfind('<|im_end|>')
    start = text.rfind('<|im_start|>assistant\n') + len('<|im_start|>assistant\n')
    if start < len('<|im_start|>assistant\n') or end <= start:
        raise ValueError('missing_action_endpoint')
    prefix = text[:end]
    return text, tok(prefix, add_special_tokens=False)['input_ids'], messages, tools


def kind(s):
    action = s['proposed_action']
    if isinstance(action, dict):
        action = action.get('action', action)
    if isinstance(action, str):
        if action.lstrip().startswith('Complete'):
            return 'complete'
        try:
            action = json.loads(action)
        except ValueError:
            return 'text_action'
    if isinstance(action, dict) and action.get('name') == 'Complete':
        return 'complete'
    return 'tool_call' if isinstance(action, dict) and 'name' in action and 'arguments' in action else 'text_action'


def sources(suite):
    runs = STORE / 'runs'
    if suite in ('dojo', 'harm', 'stepguard'):
        name = {'dojo': 'action-models-tsdojo-v1', 'harm': 'action-models-tsharm-v1',
                'stepguard': 'action-models-stepguard-v1'}[suite]
        path = runs / name / 'records.json'
        provenance = {'path': str(path), 'sha256': sha(path)}
        for r in read(path):
            yield {'key': r['key'], 'label': r['label'], 'split': r.get('original_split', r.get('split', 'external_test')),
                   'original_score': r.get('original_score'), 'domain': r.get('source_file'),
                   'actions': [{'key': r['key'], 'state': r['state'], 'kind': 'action'}],
                   'source': provenance}
        return
    from graph_safety.step_training import eval_states
    if suite == 'atbench':
        path = STORE / 'datasets/atbench-action-replay-v1/cases.json'
        cases = read(path)
        provenance = {'path': str(path), 'sha256': sha(path)}
    else:
        base = STORE / 'datasets/data-bundle-whole-dialog-v1/benchmarks/rjudge'
        cases = []
        files = {}
        for path in sorted((base / 'data').rglob('*.json')):
            files[str(path.relative_to(base))] = sha(path)
            for index, row in enumerate(read(path)):
                messages = [{k: v for k, v in m.items() if k in ('role', 'content', 'thought', 'action')}
                            for m in flatten(row['contents'])]
                cases.append({'key': f'rjudge:{path.relative_to(base)}:{index}:{row["id"]}',
                    'label': int(row['label']), 'messages': messages, 'tools': row.get('profile', '')})
        assert len(cases) == 571
        provenance = {'root': str(base.resolve()), 'files_sha256': files, 'manifest': read(base / 'manifest.json')}
    for case in cases:
        actions = []
        for eid, state in eval_states(case):
            actions.append({'key': case['key'] + ':' + eid, 'state': state, 'kind': kind(json.loads(state))})
        yield {'key': case['key'], 'label': case['label'], 'split': 'external_test',
               'actions': actions, 'source': provenance}


def prepare(root):
    from transformers import AutoTokenizer
    from build_safety_candidates import normalized, family
    tok = AutoTokenizer.from_pretrained(BASE, revision=REVISION, local_files_only=True)
    bundle = STORE / 'datasets/data-bundle-whole-dialog-v1/training/records.json'
    requests = [json.loads(r['state'])['user_request'] for r in read(bundle)]
    known = {normalized(x) for x in requests}
    masked = {family(x) for x in requests}
    for suite in SUITES:
        folder = root / suite
        folder.mkdir(exist_ok=False)
        admitted, rejected, planned, overlaps = [], [], [], []
        provenance = None
        for case in sources(suite):
            provenance = case.pop('source')
            planned.append({k: v for k, v in case.items() if k != 'actions'})
            try:
                if not case['actions']:
                    raise ValueError('no_actions')
                rendered = []
                for a in case['actions']:
                    s = json.loads(a['state'])
                    req = s['user_request']
                    if normalized(req) in known or family(req) in masked:
                        overlaps.append(case['key'])
                    text, ids, messages, tools = render(tok, s)
                    rendered.append((text, ids, messages, tools))
                    a['state_sha256'] = hashlib.sha256(a['state'].encode()).hexdigest()
                text, _, messages, tools = rendered[-1]
                ids = tok.apply_chat_template(messages, tools=tools, tokenize=True,
                    add_generation_prompt=False, return_dict=False)
                endpoints = []
                stable = True
                for a, (prefix_text, prefix_ids, _, _) in zip(case['actions'], rendered):
                    endpoint = len(prefix_ids) - 1
                    if ids[:endpoint+1] != prefix_ids:
                        stable = False
                    # Independent string check guards accidental token boundary matches.
                    prefix = prefix_text[:prefix_text.rfind('<|im_end|>')]
                    if not text.startswith(prefix):
                        stable = False
                    endpoints.append(endpoint)
                    a['native_text'] = prefix
                if stable:
                    sequences = [{'key': case['key'], 'ids': ids, 'tokens': len(ids),
                                  'endpoints': endpoints, 'action_indices': list(range(len(endpoints)))}]
                else:
                    sequences = []
                    for i, (prefix_text, prefix_ids, _, _) in enumerate(rendered):
                        causal_ids = tok(prefix_text, add_special_tokens=False)['input_ids']
                        assert causal_ids[:len(prefix_ids)] == prefix_ids
                        sequences.append({'key': case['actions'][i]['key'], 'ids': causal_ids,
                            'tokens': len(causal_ids), 'endpoints': [len(prefix_ids)-1], 'action_indices': [i]})
                if max(seq['tokens'] for seq in sequences) > 65536:
                    raise ValueError(f'whole_case_context_overflow:{max(seq["tokens"] for seq in sequences)}')
                for seq in sequences:
                    seq['ids_sha256'] = digest(seq['ids'])
                case.update(ids_sha256=digest([seq['ids_sha256'] for seq in sequences]),
                            tokens=max(seq['tokens'] for seq in sequences), sequences=sequences,
                            evaluation_mode='whole_dialog' if stable else 'causal_prefix_template_fallback')
                admitted.append(case)
            except (ValueError, TypeError) as error:
                rejected.append({'key': case['key'], 'label': case['label'], 'reason': str(error)})
        assert not overlaps, f'{suite}: training overlaps require review'
        write(folder / 'cases.json', admitted)
        write(folder / 'planned.json', planned)
        write(folder / 'shared_rejections.json', rejected)
        write(folder / 'audit.json', {'planned': len(planned), 'admitted': len(admitted),
            'actions': sum(len(c['actions']) for c in admitted), 'rejected': len(rejected),
            'reasons': dict(Counter(x['reason'].split(':')[0] for x in rejected)),
            'class_counts': dict(Counter(c['label'] for c in admitted)),
            'source': provenance, 'bundle_sha256': sha(bundle), 'overlaps': overlaps,
            'overlap_method': 'normalized/entity-masked request matching; earlier bundle exclusions; not semantic proof',
            'template_sha256': hashlib.sha256(tok.chat_template.encode()).hexdigest(),
            'cases_sha256': sha(folder / 'cases.json'), 'no_truncation': True,
            'evaluation_modes': dict(Counter(c['evaluation_mode'] for c in admitted)),
            'max_tokens': max((c['tokens'] for c in admitted), default=0)})
        print(suite, json.dumps(read(folder / 'audit.json'), ensure_ascii=False)[:1800], flush=True)


def load_head(root, arm):
    import torch
    from train_whole_dialogs import Head, backbone
    folder = root / 'locked' / arm
    for name, expected in read(root / 'snapshot_audit.json')[arm]['hashes'].items():
        assert sha(folder / name) == expected
    model = backbone(False)
    if arm.startswith('lora'):
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, folder / 'adapter', is_trainable=False)
    model.eval().requires_grad_(False)
    head = Head().cuda()
    head.load_state_dict(torch.load(folder / 'selected_head.pt', weights_only=True, map_location='cuda'))
    head.eval().requires_grad_(False)
    return model, head


def score(root, arm):
    import torch
    from tqdm.auto import tqdm
    from train_whole_dialogs import encode
    from kev.model import ContextOverflow
    assert read(root / 'input_gate.json')['status'] == 'passed'
    model = head = None
    if arm == 'kev4b':
        from graph_safety.step_training import load, encode as kev_encode, request
        from kev.api import SystemOneRequest, to_record
        cp, tok, model = load(KEV)
        _, mapping = to_record(SystemOneRequest.model_validate(request('mapping')))
        assert mapping[0]['keys'] == ['false', 'true']
        model.eval().requires_grad_(False)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        from kev.predictors import kernel_environment, long_row_kernels
        from kev.model import ROW_PASS_TOKENS
        assert abs(model.head.temperature - 2.406050072164233) < 1e-10
        environment = {'checkpoint': KEV, 'weights_sha256': cp.weights_sha256(),
            'head_sha256': sha(cp.file('head.pt')), 'temperature': model.head.temperature,
            'kernel_environment': kernel_environment(model, 'cuda')}
    else:
        model, head = load_head(root, arm)
        environment = {'base': BASE, 'revision': REVISION, 'snapshot': read(root / 'snapshot_audit.json')[arm]}
        if arm.startswith('frozen'):
            from train_whole_dialogs import frozen_hash
            environment['frozen_weights_sha256'] = frozen_hash(model)
    for suite in SUITES:
        folder = root / suite
        assert sha(folder / 'cases.json') == read(folder / 'audit.json')['cases_sha256']
        out = folder / arm
        out.mkdir(exist_ok=False)
        started = time.monotonic()
        rejects = []
        with (out / 'predictions.jsonl').open('w') as stream, torch.inference_mode():
            # Exercise the largest inputs first while memory is shared with training.
            cases = sorted(read(folder / 'cases.json'), key=lambda c: (-c['tokens'], c['key']))
            for case in tqdm(cases, desc=f'{arm}/{suite}', unit='case', mininterval=3):
                try:
                    scores = []
                    if arm == 'kev4b':
                        for a in case['actions']:
                            enc, _ = kev_encode(tok, model, a['native_text'], 65536)
                            long = len(enc['ids']) > ROW_PASS_TOKENS
                            with long_row_kernels() if long else nullcontext():
                                logits = (model.forward_batch([enc], shared_prefix=True)[0][0] if long
                                          else model(enc)[0]).float()
                            assert torch.isfinite(logits).all()
                            score_value = torch.softmax(logits, -1)[1].item()
                            scores.append({'key': a['key'], 'kind': a['kind'], 'score': score_value,
                                'native_logits': logits.cpu().tolist(), 'state_sha256': a['state_sha256'],
                                'kernels': 'efficient' if long else 'math', 'state_tokens': enc['state_tokens']})
                    else:
                        for sequence in case['sequences']:
                            cache_path = folder / 'features' / (digest(sequence['key']) + '.pt')
                            cache_identity = {'ids_sha256': sequence['ids_sha256'], 'revision': REVISION,
                                'template_sha256': read(folder / 'audit.json')['template_sha256'],
                                'torch': str(torch.__version__), 'attention': 'sdpa', 'dtype': 'bf16',
                                'padding': 'one sequence, right to128'}
                            if arm.startswith('frozen'):
                                cache_identity['weights_sha256'] = environment['frozen_weights_sha256']
                            if arm.startswith('frozen') and cache_path.exists():
                                cached = torch.load(cache_path, weights_only=True, map_location='cpu')
                                assert cached['identity'] == cache_identity
                                x = cached['hidden'].cuda()
                                del cached
                            else:
                                x = encode(model, sequence)
                                if arm.startswith('frozen'):
                                    cache_path.parent.mkdir(exist_ok=True)
                                    torch.save({'hidden': x.cpu().clone(), 'identity': cache_identity}, cache_path)
                            with torch.autocast('cuda', dtype=torch.bfloat16):
                                z = head(x)[0, sequence['endpoints']].float()
                            assert torch.isfinite(z).all()
                            for index, logit in zip(sequence['action_indices'], z):
                                a = case['actions'][index]
                                scores.append({'key': a['key'], 'kind': a['kind'], 'score': logit.sigmoid().item(),
                                    'raw_logodds': logit.item(), 'state_sha256': a['state_sha256']})
                            del x, z
                    stream.write(json.dumps({'key': case['key'], 'label': case['label'],
                        'ids_sha256': case['ids_sha256'], 'actions': scores}) + '\n')
                    stream.flush()
                except (ContextOverflow, torch.OutOfMemoryError) as error:
                    rejects.append({'key': case['key'], 'label': case['label'],
                                    'reason': type(error).__name__ + ': ' + str(error)})
                    x = z = None
                    torch.cuda.empty_cache()
                    if isinstance(error, torch.OutOfMemoryError):
                        write(out / 'rejections.json', rejects)
                        write(out / 'failure.json', {'status': 'stopped_on_first_oom',
                            'case': case['key'], 'reason': str(error), 'no_automatic_retry': True})
                        raise
        write(out / 'rejections.json', rejects)
        write(out / 'complete.json', dict(environment, seconds=time.monotonic()-started,
            rejected=len(rejects), torch=str(torch.__version__), cuda=torch.version.cuda,
            device=torch.cuda.get_device_name(), threshold=.5, no_truncation=True,
            forward_unit='Mamba whole dialog when native prefixes stable; causal-prefix fallback on template rewrite; native NOUL per action for Kev'))
    del model, head
    gc.collect()
    torch.cuda.empty_cache()


def compare(root):
    from graph_safety.metrics import evaluate
    result = {}
    for suite in SUITES:
        folder = root / suite
        cases = {r['key']: r for r in read(folder / 'cases.json')}
        outputs = {}
        for arm in ARMS:
            assert (folder / arm / 'complete.json').exists()
            outputs[arm] = {r['key']: r for r in map(json.loads, (folder / arm / 'predictions.jsonl').read_text().splitlines())}
        common = sorted(set.intersection(*(set(v) for v in outputs.values())))
        for k in common:
            for out in outputs.values():
                assert out[k]['label'] == cases[k]['label'] and out[k]['ids_sha256'] == cases[k]['ids_sha256']
                assert [(a['key'], a['state_sha256']) for a in out[k]['actions']] == [
                    (a['key'], a['state_sha256']) for a in cases[k]['actions']]
        analyses = [('all_events', common, False, 'strict')]
        if suite == 'harm':
            analyses += [('loose', common, False, 'loose'),
                         ('definite_only', [k for k in common if cases[k]['original_score'] != .5], False, 'strict')]
        if suite == 'stepguard':
            analyses = [(split, [k for k in common if cases[k]['split'] == split], False, 'strict')
                        for split in ('external_test', 'external_dev')]
        if suite in ('atbench', 'rjudge'):
            keys = [k for k in common if any(a['kind'] == 'tool_call' for a in cases[k]['actions'])]
            analyses += [('tool_only', keys, True, 'strict'), ('all_events_on_tool_cases', keys, False, 'strict')]
        if suite == 'dojo':
            analyses += [(str(d), [k for k in common if cases[k]['domain'] == d], False, 'strict')
                         for d in sorted({cases[k]['domain'] for k in common})]
        summary = {'audit': read(folder / 'audit.json'), 'common_cases': len(common),
            'excluded_by_model': {a: sorted(set(cases)-set(v)) for a, v in outputs.items()}, 'analyses': {}}
        for name, keys, tool_only, mode in analyses:
            metrics = {}
            for arm, out in outputs.items():
                labels = [int(cases[k]['original_score'] == 1) if mode == 'loose' else cases[k]['label'] for k in keys]
                scores = [max(a['score'] for a in out[k]['actions'] if not tool_only or a['kind'] == 'tool_call') for k in keys]
                if keys:
                    m = evaluate(labels, scores)
                    m.pop('pr_curve')
                    metrics[arm] = m
            summary['analyses'][name] = {'n': len(keys), 'metrics': metrics}
        write(folder / 'comparison.json', summary)
        result[suite] = summary
    write(root / 'comparison.json', result)


def wait(pid):
    try:
        fd = os.pidfd_open(pid)
    except ProcessLookupError:
        return
    select.select([fd], [], [])
    os.close(fd)


def run(root, pid, concurrent=False):
    import subprocess
    import sys
    assert read(root / 'input_gate.json')['status'] == 'passed'
    if not concurrent:
        print('Waiting for current training scheduler', pid, flush=True)
        wait(pid)
        assert read(STORE / 'runs/whole-dialog-4b-v3/status.json')['status'] == 'complete'
    # Sequential evaluation jobs; concurrent training only when explicitly authorized.
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='2')
    for arm in ARMS:
        if concurrent:
            free = int(subprocess.check_output(['nvidia-smi', '--id=2',
                '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True).strip())
            print('GPU2 free MiB before', arm, free, flush=True)
            if free < 32768:
                print('Insufficient headroom; waiting for training, without polling', flush=True)
                wait(pid)
                free = int(subprocess.check_output(['nvidia-smi', '--id=2',
                    '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True).strip())
                if free < 32768:
                    write(root / 'status.json', {'status': 'blocked_memory', 'arm': arm, 'free_MiB': free})
                    raise SystemExit(1)
        command = [sys.executable, '-u', __file__, 'score', '--root', str(root), '--arm', arm]
        with (root / (arm + '.log')).open('w') as out:
            child = subprocess.Popen(command, env=env, stdout=out, stderr=subprocess.STDOUT)
        write(root / (arm + '_launch.json'), {'pid': child.pid, 'command': command})
        print('START', arm, child.pid, flush=True)
        code = child.wait()
        if code:
            write(root / 'status.json', {'status': 'failed', 'arm': arm, 'returncode': code})
            raise SystemExit(code)
        print('END', arm, flush=True)
    compare(root)
    write(root / 'status.json', {'status': 'complete'})


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=('prepare', 'score', 'run'))
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--arm', choices=ARMS)
    p.add_argument('--wait-pid', type=int)
    p.add_argument('--concurrent', action='store_true')
    args = p.parse_args()
    if args.phase == 'prepare':
        prepare(args.root)
    elif args.phase == 'score':
        score(args.root, args.arm)
    else:
        run(args.root, args.wait_pid, args.concurrent)
