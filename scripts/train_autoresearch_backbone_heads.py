"""Small-only native 0.8B heads; isolated reuse of the frozen trainer's math/output protocol."""
import argparse
import importlib.util
import json
import math
from pathlib import Path
import time
import traceback

import torch

import prepare_autoresearch_backbone as prep
import train_autoresearch_heads as shared
from prepare_whole_dialogs import digest, write

BASE, REVISION, DIM = prep.BACKBONES['0.8b']
TEMPLATE = '273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80'
FINAL_DEADLINE = 1792348785.
TRAIN_DEADLINE = FINAL_DEADLINE - 3600


def check_deadline(args):
    if time.time() >= args.deadline_unix:
        raise RuntimeError('backbone phase deadline reached; preserve final reserve')


def make_head(args):
    shared.seed(args.seed)
    return prep.make_head(DIM, 'curve_null' if args.readout == 'null' else 'curve_linear_softmax',
                          args.attention).cuda()


def load_data(args):
    data = args.data.parent if args.data.is_file() else args.data
    audit = prep.parent.read(data / 'data_audit.json')
    mapping, lookup = prep.manifest(args.splits)
    assert (audit['base'], audit['revision'], audit['hidden_size'], audit['template_sha256']) == (BASE, REVISION, DIM, TEMPLATE)
    assert audit['no_truncation'] and audit['limit'] == 16384 and not audit['exclusion_reasons']
    assert audit.get('launch_allowed', True), 'Exclusions block launch'
    assert audit['splits_sha256'] == prep.parent.file_hash(args.splits)
    assert audit['dialogs_sha256'] == prep.parent.file_hash(data / 'dialogs.json')
    assert prep.parent.read(data / 'splits.json') == mapping
    assert prep.parent.read(data / 'rejections.json') == []
    rows = prep.parent.read(data / 'dialogs.json')
    index = {row['key']: row for row in rows}
    assert len(index) == len(rows) == len(lookup) and set(index) == set(lookup)
    actual_actions = []
    for row in rows:
        assert row['split'] == lookup[row['key']]
        assert row['tokens'] == len(row['ids']) <= 16384 and row['tokens'] > 0
        assert digest(row['ids']) == row['ids_sha256']
        assert digest([row['messages'], row['tools']]) == row['messages_tools_sha256']
        assert row['parent_ids_sha256'] and row['spans']
        end = 0
        for span in row['spans']:
            assert end <= span['start'] < span['end'] <= row['tokens']
            assert span['label'] in (0, 1)
            assert digest(row['ids'][:span['end']]) == span['prefix_ids_sha256']
            assert span['parent_prefix_ids_sha256'] and span['parent_start'] < span['parent_end']
            assert 0 < span['message_stop'] <= len(row['messages'])
            assert row['messages'][span['message_stop']-1]['role'] == 'assistant'
            actual_actions.append(span['key'])
            end = span['end']
    assert len(actual_actions) == len(set(actual_actions)) == audit['actions']
    assert max(r['tokens'] for r in rows) == audit['max_tokens']
    cache = args.cache / 'features' if (args.cache / 'features').is_dir() else args.cache
    completions = sorted(cache.glob('complete*.json'))
    assert completions, 'No completed cache shards'
    identity, shard_numbers = None, set()
    for path in completions:
        meta = prep.parent.read(path)
        assert meta['dialogs_sha256'] == audit['dialogs_sha256']
        current = {k: v for k, v in meta.items() if k not in ('dialogs_sha256', 'shard', 'count')}
        assert identity is None or identity == current, 'Different cache shard identities'
        identity = current
        shard = meta['shard']
        assert shard not in shard_numbers and path.name == f'complete{shard}.json'
        shard_numbers.add(shard)
        assert meta['count'] == len(rows[shard::meta['shard_count']])
    assert shard_numbers == set(range(identity['shard_count'])), 'Incomplete cache'
    assert (identity['base'], identity['revision'], identity['hidden_size'], identity['template_sha256']) == (BASE, REVISION, DIM, TEMPLATE)
    assert identity['dtype'] == 'bf16' and identity['attention'] == 'sdpa'
    assert len(identity['weights_sha256']) == 64
    for name in ('tokenizer_vocab_sha256', 'source_dialogs_sha256', 'splits_sha256'):
        assert identity[name] == audit[name]
    assert identity.get('prepared_dialogs_sha256', audit['dialogs_sha256']) == audit['dialogs_sha256']
    if 'model_files' in audit:  # Other explicitly owned native-small preparer.
        assert identity['model_files_sha256'] == digest(audit['model_files'])
    else:  # prepare_autoresearch_backbone.py schema.
        for name, value in audit.items():
            if name != 'dialogs_sha256':
                assert identity[name] == value, name
    assert {p.stem for p in cache.glob('*.pt')} == set(index), 'Missing/extra cached dialogs'
    args.cache, args.cache_identity, args.audit, args.hidden_size = cache, identity, audit, DIM
    splits = {name: [index[key] for key in keys] for name, keys in mapping.items()}
    assert {name: len(values) for name, values in splits.items()} == prep.COUNTS
    # Compatibility alias only: NEVER the original 6058-dialog full-training set.
    splits['train_full'] = splits['train_small']
    return rows, splits


def features(args, row):
    assert row['split'] in ('train_small', 'dev_select'), 'No confirmation/external cache reads here'
    saved = torch.load(args.cache / (row['key']+'.pt'), map_location='cpu', weights_only=True)
    expected = dict(args.cache_identity)
    # Our preparer embeds dialogs_sha256 in tensor identity; completion strips it.
    if 'dialogs_sha256' in saved['identity']:
        expected['dialogs_sha256'] = args.audit['dialogs_sha256']
    assert saved['identity'] == expected
    assert saved['ids_sha256'] == row['ids_sha256']
    assert saved['parent_ids_sha256'] == row['parent_ids_sha256']
    if 'parent_row_sha256' in row:
        assert saved['parent_row_sha256'] == row['parent_row_sha256']
    else:
        assert saved['messages_tools_sha256'] == row['messages_tools_sha256']
    hidden = saved['hidden']
    assert hidden.shape == (row['tokens'], DIM) and hidden.dtype == torch.bfloat16
    assert torch.isfinite(hidden).all()
    x = torch.zeros((1, math.ceil(row['tokens']/128)*128, DIM), device='cuda', dtype=torch.bfloat16)
    x[0, :row['tokens']] = hidden.to('cuda')
    return x


def config(args):
    sources = [Path(__file__), Path(prep.__file__), Path(shared.__file__),
               Path(shared.original.__file__), Path(shared.curve_heads.__file__),
               Path(prep.parent.__file__), Path(__file__).with_name('prepare_whole_dialogs.py')]
    return {'readout': args.readout, 'curve': args.curve, 'event_weight': args.event_weight,
            'attention_before_mamba': args.attention, 'seed': args.seed,
            'base': BASE, 'revision': REVISION, 'hidden_size': DIM,
            'data_sha256': args.audit['dialogs_sha256'], 'template_sha256': TEMPLATE,
            'cache_identity': args.cache_identity, 'splits_sha256': prep.parent.file_hash(args.splits),
            'train_split': 'train_small', 'train_full_is_small_alias': True, 'train_dialogs': 2009,
            'original_full_train_used': False,
            'sources': {p.name: prep.parent.file_hash(p) for p in sources},
            'reuse': 'isolated module function bindings; original modules unchanged',
            'head': 'LN1024 -> Linear1024x256; unchanged two residual Mamba2 blocks/readouts',
            'selection': 'dev_select AP; ties hard event BCE, earliest epoch'}


def gate(args):
    rows, splits = load_data(args)
    assert not (args.root / 'gate.json').exists()
    probe = min((r for r in splits['train_small'] if r['spans'][0]['end'] < r['tokens']), key=lambda r: r['tokens'])
    head = make_head(args).eval()
    init = shared.curve_heads.initialization(head)
    x = features(args, probe)
    cut = probe['spans'][0]['end']
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        before, nb = head(x)
        changed = x.clone()
        changed[:, cut:] = 0
        after, na = head(changed)
        causal_error = (before[:, :cut]-after[:, :cut]).abs().max().item()
        null_error = (nb[:, :cut]-na[:, :cut]).abs().max().item() if nb is not None else 0.
        assert causal_error == null_error == 0
        torch.save(head.state_dict(), args.root / 'gate_reload.pt')
        head.load_state_dict(torch.load(args.root / 'gate_reload.pt', weights_only=True))
        loaded, nl = head(x)
        reload_error = (loaded-before).abs().max().item()
        assert reload_error == 0 and (nb is None or torch.equal(nb, nl))
    del x, changed, before, after, nb, na, loaded, nl
    head.train()
    z, null = args.core.forward(head, probe, args)
    z.retain_grad()
    if null is not None:
        null.retain_grad()
    token, event = shared.loss_parts(z, null, probe['spans'], args)
    watched = [(n, p) for n, p in head.named_parameters() if n == 'readout.weight' or
               n.startswith('blocks.') or n in ('projection.weight', 'attention_qkv.weight')]
    contributions = {}
    for name, loss in (('token', token), ('event', event)):
        gradients = torch.autograd.grad(loss, [p for _, p in watched], retain_graph=True)
        norms = {n: g.float().norm().item() for (n, _), g in zip(watched, gradients)}
        assert all(math.isfinite(v) for v in norms.values()) and norms['readout.weight'] > 0
        assert norms['projection.weight'] > 0
        for i in (0, 1):
            assert any(v > 0 for n, v in norms.items() if n.startswith(f'blocks.{i}.'))
        if args.attention:
            assert norms['attention_qkv.weight'] > 0
        contributions[name] = norms
    (token+args.event_weight*event).backward()
    mask = torch.zeros_like(z, dtype=torch.bool)
    ends = torch.zeros_like(mask)
    for span in probe['spans']:
        mask[span['start']:span['end']] = True
        ends[span['end']-1] = True
    assert torch.count_nonzero(z.grad[~mask]) == 0
    if null is not None:
        assert torch.count_nonzero(null.grad[~ends]) == 0
        assert head.null_readout.weight.grad.norm().item() > 0
    if args.readout == 'linear_softmax':
        for span in probe['spans']:
            p = z[span['start']:span['end']].detach().sigmoid()
            assert abs((p.square().sum()/p.sum()).item()-shared.event_logit(z, null, span, args.readout).sigmoid().item()) < 1e-6
    del z, null, token, event, gradients
    head.zero_grad(set_to_none=True)
    longest_tokens = max(row['tokens'] for row in rows)
    torch.cuda.reset_peak_memory_stats()
    x = torch.zeros((1, math.ceil(longest_tokens/128)*128, DIM), device='cuda', dtype=torch.bfloat16)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        z, null = head(x)
    z = z[0, :longest_tokens].float()
    null = null[0, :longest_tokens].float() if null is not None else None
    token, event = shared.loss_parts(z, null, [{'start': 0, 'end': longest_tokens, 'label': 1}], args)
    loss = token+args.event_weight*event
    assert torch.isfinite(loss)
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in head.parameters() if p.grad is not None)
    check_deadline(args)
    write(args.root / 'gate.json', {'status': 'passed', 'config': config(args), **init,
          'environment': shared.environment(), 'gradient_contributions': contributions,
          'causal_error': causal_error, 'null_causal_error': null_error, 'reload_error': reload_error,
          'loss_support_passed': True, 'probe': probe['key'], 'probe_tokens': probe['tokens'],
          'longest_tokens': longest_tokens, 'longest_backward_input': 'synthetic zeros; actual native maximum length; no held-out features/labels',
          'longest_backward_peak_bytes': torch.cuda.max_memory_allocated(),
          'optimizer_steps': 0, 'qwen_loaded': False, 'audit_gradients_discarded': True})


def bind_core(args):
    # Isolated module globals: never monkeypatch the imported shared trainer.
    spec = importlib.util.spec_from_file_location('_native_small_backbone_core', shared.__file__)
    core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(core)
    original_forward = core.forward

    def forward(head, row, bound_args):
        check_deadline(bound_args)
        return original_forward(head, row, bound_args)

    core.make_head, core.load_data, core.features, core.config = make_head, load_data, features, config
    core.forward, core.gate = forward, gate
    args.core = core
    return core


def evaluate_checkpoint(args):
    args.core.evaluate_checkpoint(args)
    selection = prep.parent.read(args.root / 'fit' / 'selection.json')
    reference = args.root / 'fit' / f'dev_select_epoch{selection["epoch"]}'
    replay = args.root / 'evaluations' / 'dev_select'
    files = {path.name for path in (reference / 'tokens').glob('*.pt')}
    assert files == {path.name for path in (replay / 'tokens').glob('*.pt')}
    for name in sorted(files):
        check_deadline(args)
        old = torch.load(reference / 'tokens' / name, map_location='cpu', weights_only=True)
        new = torch.load(replay / 'tokens' / name, map_location='cpu', weights_only=True)
        for field in ('dialog', 'ids', 'ids_sha256', 'tokens', 'spans'):
            assert old[field] == new[field], (name, field)
        assert torch.equal(old['raw_token_logits'], new['raw_token_logits']), name
        if old['null_logits'] is None:
            assert new['null_logits'] is None
        else:
            assert torch.equal(old['null_logits'], new['null_logits']), name
    old_events = [json.loads(line) for line in (reference / 'events.jsonl').read_text().splitlines()]
    new_events = [json.loads(line) for line in (replay / 'events.jsonl').read_text().splitlines()]
    assert len(old_events) == len(new_events)
    event_error = 0.
    for old, new in zip(old_events, new_events):
        for field in ('dialog', 'span', 'ids_sha256'):
            assert old[field] == new[field]
        event_error = max(event_error, abs(old['event_logit']-new['event_logit']))
    assert event_error <= 1e-6
    write(replay / 'raw_replay.json', {'status': 'passed', 'raw_token_logit_max_error': 0.,
          'event_logit_max_error': event_error, 'dialogs': len(files), 'events': len(old_events),
          'config': config(args), 'checkpoint_sha256': selection['head_sha256'],
          'reference_events_sha256': prep.parent.file_hash(reference / 'events.jsonl'),
          'replay_events_sha256': prep.parent.file_hash(replay / 'events.jsonl')})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('gate', 'train', 'eval'))
    for name in ('data', 'cache', 'splits', 'root'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--deadline-unix', type=float, required=True)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--readout', choices=shared.READOUTS, default='endpoint')
    parser.add_argument('--curve', choices=('broad', 'linear', 'step'), default='broad')
    parser.add_argument('--event-weight', type=float, default=.1)
    parser.add_argument('--attention', action='store_true')
    parser.add_argument('--seed', type=int, default=shared.original.SEED)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--eval-split', choices=('dev_select',), default='dev_select')
    args = parser.parse_args()
    allowed_deadline = FINAL_DEADLINE if args.phase == 'eval' else TRAIN_DEADLINE
    if not math.isfinite(args.deadline_unix) or args.deadline_unix > allowed_deadline:
        parser.error(f'deadline must be <= {allowed_deadline}; gate/train preserve one-hour final reserve')
    if args.event_weight != .1 or args.epochs < 1:
        parser.error('require simultaneous token BCE +0.1event BCE and positive epochs')
    args.train_split = 'train_small'
    args.root.mkdir(parents=True, exist_ok=True)
    core = bind_core(args)
    try:
        check_deadline(args)
        {'gate': core.gate, 'train': core.train, 'eval': evaluate_checkpoint}[args.phase](args)
    except Exception:
        write(args.root / f'failure_{args.phase}_{time.time_ns()}.json',
              {'phase': args.phase, 'traceback': traceback.format_exc(), 'status': 'failed'})
        raise


if __name__ == '__main__':
    main()
