"""Five authorized frozen-Instruct heads, reusing the original training loop."""
import argparse
import hashlib
import math
from pathlib import Path
import time
import traceback

import torch
from tqdm import tqdm

import train_autoresearch_heads as base
import train_sound_anomaly_heads as sound
from prepare_whole_dialogs import digest, write

ORIGINAL_CONFIG = base.config
ORIGINAL_FORWARD = base.forward


def load_data(args):
    audit = base.original.read(args.data / 'data_audit.json')
    assert audit['launch_allowed'] and not audit['exclusion_reasons']
    assert base.original.file_hash(args.data / 'dialogs.json') == audit['dialogs_sha256']
    assert base.original.file_hash(args.splits) == audit['splits_sha256']
    rows = base.original.read(args.data / 'dialogs.json')
    mapping = base.original.read(args.splits)
    assert {k: len(v) for k, v in mapping.items()} == {'train_small': 2009, 'dev_select': 301}
    assert not set(mapping['train_small']) & set(mapping['dev_select'])
    index = {r['key']: r for r in rows}
    assert len(index) == len(rows) == 2310
    assert set(index) == set(mapping['train_small']) | set(mapping['dev_select'])
    args.cache = args.cache / 'features' if (args.cache / 'features').is_dir() else args.cache
    complete = base.original.read(args.cache / 'complete0.json')
    assert complete['count'] == 2310 and complete['shard'] == 0
    args.cache_identity = {k: v for k, v in complete.items() if k not in ('count', 'shard')}
    # torch.cuda.get_device_capability returns a tuple; JSON stores it as a list.
    capability = args.cache_identity['environment']['capability']
    assert isinstance(capability, list) and len(capability) == 2
    assert all(type(value) is int for value in capability)
    args.cache_identity['environment']['capability'] = tuple(capability)
    for field in ('base', 'revision', 'template_sha256', 'dialogs_sha256'):
        assert args.cache_identity[field] == audit[field]
    args.audit = audit
    for row in rows:
        assert row['tokens'] == len(row['ids']) <= 16384
        assert row['ids_sha256'] == digest(row['ids'])
        previous = 0
        for span in row['spans']:
            assert previous <= span['start'] < span['end'] <= row['tokens']
            assert span['label'] in (0, 1)
            previous = span['end']
    return rows, {k: [index[key] for key in keys] for k, keys in mapping.items()}


def config(args):
    result = ORIGINAL_CONFIG(args)
    # Config is persisted as JSON, whereas feature identity compares Torch types.
    identity = dict(result['cache_identity'])
    identity['environment'] = dict(identity['environment'])
    identity['environment']['capability'] = list(identity['environment']['capability'])
    result['cache_identity'] = identity
    result.update(arm=args.arm, epochs=2, attention_before_mamba=False,
                  experiment_source_sha256=base.original.file_hash(Path(__file__)),
                  sound_source_sha256=base.original.file_hash(Path(sound.__file__)),
                  control_gate_sha256=base.original.file_hash(args.control_gate),
                  token_target='hard action Y' if args.arm in ('B', 'C') else args.curve,
                  deadline_unix=args.deadline_unix)
    return result


def forward(head, row, args):
    sound.guard(args)
    return ORIGINAL_FORWARD(head, row, args)


def gate(args):
    progress = tqdm(total=5, desc=args.arm + ' CUDA gates', unit='check')
    sound.guard(args)
    rows, splits = load_data(args)
    head = base.make_head(args).eval()
    init = base.curve_heads.initialization(head)
    core = {n: v for n, v in head.state_dict().items()
            if not n.startswith(('evidence.', 'window.', 'event.')) and n != 'alpha_raw'}
    core_hash = digest({n: hashlib.sha256(v.detach().cpu().numpy().tobytes()).hexdigest()
                        for n, v in core.items()})
    reference = base.original.read(args.control_gate)
    assert reference['status'] == 'passed'
    assert core_hash == reference['shared_head_sha256'], 'Initialization differs from frozen Base control'
    progress.update()
    probe = min((r for r in splits['train_small'] if r['spans'][0]['end'] < r['tokens']),
                key=lambda r: r['tokens'])
    x = base.features(args, probe)
    cut = probe['spans'][0]['end']
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        z, h = head(x)
        changed = x.clone()
        changed[:, cut:] = 0
        za, ha = head(changed)
        assert torch.equal(z[:, :cut], za[:, :cut])
        assert h is None or torch.equal(h[:, :cut], ha[:, :cut])
        torch.save(head.state_dict(), args.root / 'gate_reload.pt')
        head.load_state_dict(torch.load(args.root / 'gate_reload.pt', weights_only=True))
        zr, hr = head(x)
        assert torch.equal(z, zr) and (h is None or torch.equal(h, hr))
    span = probe['spans'][0]
    with torch.no_grad():
        before = base.event_logit(z[0].float(), h[0].float() if h is not None else None, span, args.readout)
        after = base.event_logit(za[0].float(), ha[0].float() if ha is not None else None, span, args.readout)
        assert torch.equal(before, after)
    del x, changed, z, h, za, ha, zr, hr
    progress.update()
    head.train()
    z, h = base.forward(head, probe, args)
    token, event = base.loss_parts(z, h, probe['spans'], args)
    contributions = {}
    for kind, loss in [('token', token), ('event', event)]:
        prefix = 'readout.' if kind == 'token' or args.arm.startswith('A_') else ('evidence.' if args.arm == 'B' else 'window.')
        watched = [(n, p) for n, p in head.named_parameters()
                   if n.startswith(('blocks.', prefix)) or (kind == 'event' and n.startswith('event.'))
                   or (kind == 'event' and n == 'alpha_raw')]
        gradients = torch.autograd.grad(loss, [p for _, p in watched], retain_graph=True)
        norms = {n: g.float().norm().item() for (n, _), g in zip(watched, gradients)}
        assert all(math.isfinite(v) for v in norms.values())
        for required in ('blocks.0.', 'blocks.1.', prefix):
            assert any(v > 0 for n, v in norms.items() if n.startswith(required))
        if kind == 'event' and args.arm in ('B', 'C'):
            assert norms['alpha_raw' if args.arm == 'B' else 'event.weight'] > 0
        contributions[kind] = norms
    progress.update()
    zd = z.detach().requires_grad_()
    hd = h.detach().requires_grad_() if h is not None else None
    td, ed = base.loss_parts(zd, hd, probe['spans'], args)
    gradients = torch.autograd.grad(td + .1 * ed, (zd, hd) if hd is not None else (zd,))
    mask = torch.zeros_like(z, dtype=torch.bool)
    for span in probe['spans']:
        mask[span['start']:span['end']] = True
    assert all(torch.count_nonzero(g[~mask]) == 0 for g in gradients)
    (token + .1 * event).backward()
    del z, h, zd, hd, token, event, td, ed, gradients
    head.zero_grad(set_to_none=True)
    progress.update()
    longest = max(rows, key=lambda r: r['tokens'])
    torch.cuda.reset_peak_memory_stats()
    z, h = base.forward(head, longest, args)
    token, event = base.loss_parts(z, h, longest['spans'], args)
    loss = token + .1 * event
    assert torch.isfinite(loss)
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in head.parameters() if p.grad is not None)
    write(args.root / 'gate.json', {'status': 'passed', 'config': config(args), **init,
          'shared_control_core_sha256': core_hash, 'environment': base.environment(),
          'gradient_contributions': contributions, 'causal_error': 0, 'reload_error': 0,
          'loss_support_passed': True, 'longest_tokens': longest['tokens'], 'longest_key': longest['key'],
          'longest_backward_input': 'real authorized train_small/dev_select cached features',
          'longest_backward_peak_bytes': torch.cuda.max_memory_allocated(), 'qwen_loaded': False})
    progress.update()
    progress.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('gate', 'train'))
    for name in ('root', 'data', 'cache', 'splits', 'control-gate'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--arm', choices=('A_broad', 'A_linear', 'A_step', 'B', 'C'), required=True)
    parser.add_argument('--deadline-unix', type=float, required=True)
    args = parser.parse_args()
    args.seed, args.epochs, args.event_weight = 20261017, 2, .1
    args.train_split, args.attention = 'train_small', False
    args.curve = args.arm.split('_')[1] if args.arm.startswith('A_') else 'step'
    args.readout = 'endpoint' if args.arm.startswith('A_') else args.arm
    args.root.mkdir(parents=True, exist_ok=True)
    base.load_data, base.config, base.environment = load_data, config, sound.environment
    base.forward = forward
    if args.arm in ('B', 'C'):
        base.make_head, base.forward = sound.make_head, sound.forward
        base.event_logit, base.loss_parts, base.validate = sound.event_logit, sound.loss_parts, sound.validate
    try:
        (gate if args.phase == 'gate' else base.train)(args)
    except Exception:
        write(args.root / f'failure_{args.phase}_{time.time_ns()}.json',
              {'status': 'failed', 'traceback': traceback.format_exc()})
        raise


if __name__ == '__main__':
    main()
