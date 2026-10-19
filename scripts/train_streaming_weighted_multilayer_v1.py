"""Registered six-arm adaptation: frozen features, weighted BCE + endpoint BCE."""
import argparse
import io
import json
import math
from pathlib import Path
import time
import traceback
from types import SimpleNamespace
import torch
import torch.nn.functional as F
from tqdm import tqdm
import train_autoresearch_heads as base
import train_frozen_instruct_4b_v1 as instruct
from prepare_whole_dialogs import write, digest

RUN = Path('/var/local/danefedov/experiments/agent-safety/runs/streaming-weighted-multilayer-v1')
OLD = RUN.parent / 'frozen-instruct-4b-v1'
DEADLINE = 1792345185
ORIGINAL_VALIDATE = base.validate
ORIGINAL_FORWARD = base.forward


def canonical(value):
    return json.loads(json.dumps(value))


def arguments(feature_set, curve, phase):
    return SimpleNamespace(root=RUN / (feature_set + '_' + curve), data=OLD / 'data',
        cache=(OLD if feature_set == 'last' else RUN) / 'cache',
        splits=OLD / 'data/splits.json', feature_set=feature_set, curve=curve,
        phase=phase, seed=20261017, epochs=2, event_weight=.1, train_split='train_small',
        attention=False, readout='endpoint', deadline_unix=DEADLINE)


def load_data(args):
    return instruct.load_data(args)


def config(args):
    result = canonical(instruct.ORIGINAL_CONFIG(args))
    result.update(feature_set=args.feature_set, epochs=2, deadline_unix=DEADLINE,
        weighting={'temperature': 1., 'tau': .35, 'detach': True,
                   'dispersion': 'population std of all labeled action tokens in current dialog',
                   'normalization': 'softmax within each action; mean actions; mean four dialogs'},
        source_sha256=base.original.file_hash(Path(__file__)),
        hidden_indices=[32] if args.feature_set == 'last' else [11, 22, 32])
    return result


def features(args, row):
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        saved = torch.load(args.cache / (row['key'] + '.pt'), map_location='cpu', weights_only=True)
    assert saved['ids_sha256'] == row['ids_sha256'] == digest(row['ids'])
    assert canonical(saved['identity']) == canonical(args.cache_identity)
    width = 2560 if args.feature_set == 'last' else 7680
    hidden = saved['hidden']
    assert hidden.shape == (row['tokens'], width) and hidden.dtype == torch.bfloat16
    assert torch.isfinite(hidden).all()
    x = torch.zeros((1, math.ceil(row['tokens']/128)*128, width), device='cuda', dtype=torch.bfloat16)
    x[0, :row['tokens']] = hidden.cuda()
    return x


def make_head(args):
    base.seed(args.seed)
    head = base.curve_heads.Head('curve_linear_softmax', False)
    if args.feature_set == 'three':
        # Preserve initialized Mamba/readout; only input normalization/projection differ.
        head.input_norm = torch.nn.LayerNorm(7680)
        head.projection = torch.nn.Linear(7680, 256)
    return head.cuda()


def loss_parts(z, null, spans, args):
    valid = torch.cat([z[s['start']:s['end']].float().sigmoid() for s in spans])
    beta = (valid.std(unbiased=False)/.35).clamp(0, 1) if len(valid) >= 2 else valid.new_zeros(())
    tokens, events = [], []
    for span in spans:
        values = z[span['start']:span['end']].float()
        alpha = torch.softmax(beta * values.sigmoid(), dim=0).detach()
        target = float(span['label']) * base.curve_heads.profile(len(values), z.device, args.curve)
        tokens.append((alpha * F.binary_cross_entropy_with_logits(values, target, reduction='none')).sum())
        events.append(F.binary_cross_entropy_with_logits(values[-1], values.new_tensor(float(span['label']))))
    return torch.stack(tokens).mean(), torch.stack(events).mean()


def forward(head, row, args):
    assert time.time() < DEADLINE, 'Original training deadline reached'
    return ORIGINAL_FORWARD(head, row, args)


def validate(head, rows, args, folder):
    metrics = ORIGINAL_VALIDATE(head, rows, args, folder)
    events = [json.loads(line) for line in (folder / 'events.jsonl').read_text().splitlines()]
    index = {row['key']: row for row in rows}
    groups = {}
    for event in events:
        row, span = index[event['dialog']], event['span']
        source = '|'.join(sorted(row['sources']))
        length = span['end'] - span['start']
        bucket = '<=32' if length <= 32 else ('33-128' if length <= 128 else '>128')
        for key in ('source:' + source, 'action_length:' + bucket):
            groups.setdefault(key, []).append(event)
    report = {}
    for name, group in groups.items():
        result = base.evaluate([e['span']['label'] for e in group], [e['score'] for e in group])
        result.pop('pr_curve', None)
        result['n'] = len(group)
        report[name] = result
    write(folder / 'fixed_threshold_slices.json', report)
    return metrics


def gate(args):
    rows, splits = load_data(args)
    with tqdm(total=5, desc=args.root.name + ' CUDA gates', unit='check') as progress:
        head = make_head(args).eval()
        init = base.curve_heads.initialization(head)
        reference = base.original.read(RUN.parent / 'frozen-instruct-4b-v3/A_broad/gate.json')
        if args.feature_set == 'last':
            assert init['full_head_sha256'] == reference['full_head_sha256']
        progress.update()
        probe = min((r for r in splits['train_small'] if r['spans'][0]['end'] < r['tokens']), key=lambda r: r['tokens'])
        x = features(args, probe)
        cut = probe['spans'][0]['end']
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            z, _ = head(x)
            changed = x.clone()
            changed[:, cut:] = 0
            za, _ = head(changed)
            assert torch.equal(z[:, :cut], za[:, :cut])
            torch.save(head.state_dict(), args.root / 'gate_reload.pt')
            head.load_state_dict(torch.load(args.root / 'gate_reload.pt', weights_only=True))
            zr, _ = head(x)
            assert torch.equal(z, zr)
        del x, changed, z, za, zr
        progress.update()
        head.train()
        z, h = forward(head, probe, args)
        token, event = loss_parts(z, h, probe['spans'], args)
        watched = [(n, p) for n, p in head.named_parameters() if n.startswith(('blocks.', 'readout.', 'projection.'))]
        contributions = {}
        for kind, loss in (('token', token), ('event', event)):
            gradients = torch.autograd.grad(loss, [p for _, p in watched], retain_graph=True)
            norms = {n: g.float().norm().item() for (n, _), g in zip(watched, gradients)}
            assert all(math.isfinite(v) for v in norms.values())
            for required in ('blocks.0.', 'blocks.1.', 'readout.', 'projection.'):
                assert any(v > 0 for n, v in norms.items() if n.startswith(required))
            contributions[kind] = norms
        progress.update()
        zd = z.detach().requires_grad_()
        td, ed = loss_parts(zd, None, probe['spans'], args)
        gradient = torch.autograd.grad(td + .1 * ed, zd)[0]
        mask = torch.zeros_like(z, dtype=torch.bool)
        for span in probe['spans']:
            mask[span['start']:span['end']] = True
        assert torch.count_nonzero(gradient[~mask]) == 0
        (token + .1 * event).backward()
        del z, h, token, event, zd, td, ed, gradients
        head.zero_grad(set_to_none=True)
        progress.update()
        longest = max(rows, key=lambda r: r['tokens'])
        torch.cuda.reset_peak_memory_stats()
        z, h = forward(head, longest, args)
        token, event = loss_parts(z, h, longest['spans'], args)
        loss = token + .1 * event
        assert torch.isfinite(loss)
        loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in head.parameters() if p.grad is not None)
        write(args.root / 'gate.json', {'status': 'passed', 'config': config(args), **init,
            'environment': base.environment(), 'gradient_contributions': contributions,
            'causal_error': 0, 'reload_error': 0, 'loss_support_passed': True,
            'longest_tokens': longest['tokens'], 'longest_key': longest['key'],
            'longest_backward_peak_bytes': torch.cuda.max_memory_allocated(), 'qwen_loaded': False})
        progress.update()


def cpu_admission(feature_set):
    configs = {}
    for curve in tqdm(('step', 'linear', 'broad'), desc=feature_set + ' CPU config boundaries'):
        args = arguments(feature_set, curve, 'gate')
        rows, _ = load_data(args)
        cfg = config(args)
        assert cfg == canonical(cfg)
        buffer = io.BytesIO()
        torch.save(cfg, buffer)
        buffer.seek(0)
        assert torch.load(buffer, weights_only=True) == cfg
        configs[curve] = cfg
    width = 2560 if feature_set == 'last' else 7680
    hashes = {}
    for row in tqdm(rows, desc=feature_set + ' CPU cache identity/IDs/shapes/hash', unit='dialog'):
        assert time.time() < DEADLINE
        path = args.cache / (row['key'] + '.pt')
        with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
            saved = torch.load(path, map_location='cpu', weights_only=True)
        assert canonical(saved['identity']) == canonical(args.cache_identity)
        assert saved['ids_sha256'] == row['ids_sha256'] == digest(row['ids'])
        assert saved['hidden'].shape == (row['tokens'], width)
        assert saved['hidden'].dtype == torch.bfloat16 and torch.isfinite(saved['hidden']).all()
        hashes[row['key']] = base.original.file_hash(path)
    if feature_set == 'last':
        reference = base.original.read(RUN.parent / 'frozen-instruct-4b-v3/v1_hashes.json')
        assert all(reference['cache/features/' + k + '.pt'] == v for k, v in hashes.items())
    write(RUN / (feature_set + '_admission.json'), {'status': 'passed', 'count': len(rows), 'configs': configs, 'hashes': hashes})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('cpu', 'gate', 'train'))
    parser.add_argument('--feature-set', choices=('last', 'three'), required=True)
    parser.add_argument('--curve', choices=('step', 'linear', 'broad'), default='step')
    cli = parser.parse_args()
    base.load_data, base.config, base.features = load_data, config, features
    base.make_head, base.forward, base.loss_parts = make_head, forward, loss_parts
    base.validate, base.environment = validate, instruct.sound.environment
    torch.set_num_threads(4)
    if cli.phase == 'cpu':
        cpu_admission(cli.feature_set)
        return
    args = arguments(cli.feature_set, cli.curve, cli.phase)
    args.root.mkdir(parents=True, exist_ok=True)
    try:
        (gate if cli.phase == 'gate' else base.train)(args)
        if cli.phase == 'train':
            path = args.root / 'fit/run.json'
            metadata = base.original.read(path)
            metadata['normalization'] = 'detached adaptive-weighted sum tokens per action, mean actions per dialog, mean four dialogs per optimizer step'
            write(path, metadata)
    except Exception:
        write(args.root / f'failure_{cli.phase}_{time.time_ns()}.json', {'traceback': traceback.format_exc()})
        raise


if __name__ == '__main__':
    main()
