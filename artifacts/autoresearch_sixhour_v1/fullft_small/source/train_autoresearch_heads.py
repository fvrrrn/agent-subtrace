"""Frozen whole-dialog heads; immutable source/cache, explicit selection splits."""
import argparse
import json
import math
import os
from pathlib import Path
import platform
import random
import shutil
import sys
import time
import traceback
from importlib.metadata import version, PackageNotFoundError

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

import train_curve_event_heads as curve_heads
import train_whole_dialogs as original
from prepare_whole_dialogs import write
from graph_safety.metrics import evaluate

READOUTS = ('endpoint', 'linear_softmax', 'null', 'multiscale', 'leaky')
SPLITS = ('train_small', 'train_full', 'dev_select', 'dev_confirm', 'original_validation')


def seed(value):
    random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False


def make_head(args):
    seed(args.seed)
    return curve_heads.Head('curve_null' if args.readout == 'null' else 'curve_linear_softmax',
                            args.attention).cuda()


def event_logit(z, null, span, readout):
    values = z[span['start']:span['end']].float()
    if readout == 'endpoint':
        return values[-1]
    if readout == 'null':
        return torch.logsumexp(values, 0) - math.log(len(values)) - null[span['end']-1]
    if readout == 'linear_softmax':
        positive, negative = F.logsigmoid(values), F.logsigmoid(-values)
        return torch.logsumexp(2*positive, 0) - torch.logsumexp(positive+negative, 0)
    if readout == 'leaky':
        # Endpoint of a normalized causal leaky integrator, e-folding32tokens.
        # Signed logits permit counterevidence; normalized weights avoid count growth.
        ages = torch.arange(len(values)-1, -1, -1, device=values.device, dtype=torch.float32)
        weights = torch.softmax(-ages/32., dim=0)
        return (weights*values).sum()
    # Fixed unit-temperature scan over means, not sums: no length reward.
    # Each valid window and each distinct scale gets equal prior mass.
    scales = sorted({min(len(values), width) for width in (1, 8, 32, 128)})
    pooled = []
    for width in scales:
        means = F.avg_pool1d(values[None, None, :], width, stride=1).flatten()
        pooled.append(torch.logsumexp(means, 0) - math.log(len(means)))
    return torch.logsumexp(torch.stack(pooled), 0) - math.log(len(pooled))


def loss_parts(z, null, spans, args):
    token, event = [], []
    for span in spans:
        values = z[span['start']:span['end']]
        target = float(span['label']) * curve_heads.profile(len(values), z.device, args.curve)
        token.append(F.binary_cross_entropy_with_logits(values, target))
        logit = event_logit(z, null, span, args.readout)
        event.append(F.binary_cross_entropy_with_logits(logit, logit.new_tensor(float(span['label']))))
    return torch.stack(token).mean(), torch.stack(event).mean()


def load_data(args):
    data = args.data.parent if args.data.is_file() else args.data
    rows, audit = original.load_data(data)
    cache = args.cache / 'features' if (args.cache / 'features').is_dir() else args.cache
    identity = None
    completion_files = sorted(cache.glob('complete*.json'))
    assert completion_files, 'No completed cache shards'
    for completion in completion_files:
        meta = original.read(completion)
        assert meta['dialogs_sha256'] == audit['dialogs_sha256'], 'Source/cache data mismatch'
        assert meta['template_sha256'] == audit['template_sha256']
        assert meta['base'] == audit['base'] and meta['revision'] == audit['revision']
        current = {k: v for k, v in meta.items() if k not in ('dialogs_sha256', 'shard', 'count')}
        assert identity is None or current == identity
        identity = current
    mapping = original.read(args.splits)
    assert set(mapping) == set(SPLITS), f'Expected top-level split mapping: {SPLITS}'
    index = {r['key']: r for r in rows}
    assert len(index) == len(rows)
    sets = {}
    for name, keys in mapping.items():
        assert isinstance(keys, list) and all(isinstance(k, str) for k in keys)
        assert len(set(keys)) == len(keys) and set(keys) <= index.keys(), name
        sets[name] = set(keys)
    assert sets['train_small'] <= sets['train_full']
    independent = ('train_full', 'dev_select', 'dev_confirm', 'original_validation')
    for i, name in enumerate(independent):
        for other in independent[i+1:]:
            assert not sets[name] & sets[other], f'Overlapping splits: {name}/{other}'
    assert sets['original_validation'] == {r['key'] for r in rows if r.get('parent_split', r['split']) == 'validation'}
    assert sets['train_full'] | sets['dev_select'] | sets['dev_confirm'] <= {
        r['key'] for r in rows if r.get('parent_split', r['split']) == 'train'}
    for row in rows:
        assert row['tokens'] == len(row['ids']) and row['spans']
        previous = 0
        for span in row['spans']:
            assert previous <= span['start'] < span['end'] <= row['tokens']
            assert span['label'] in (0, 1)
            previous = span['end']
    args.cache = cache
    args.cache_identity = identity
    args.audit = audit
    return rows, {name: [index[key] for key in keys] for name, keys in mapping.items()}


def features(args, row):
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        saved = torch.load(args.cache / (row['key']+'.pt'), map_location='cpu', weights_only=True)
    assert saved['ids_sha256'] == row['ids_sha256']
    assert saved['identity'] == args.cache_identity
    hidden = saved['hidden']
    assert hidden.shape == (row['tokens'], 2560) and hidden.dtype == torch.bfloat16
    assert torch.isfinite(hidden).all()
    n = math.ceil(row['tokens']/128)*128
    x = torch.zeros((1, n, 2560), device='cuda', dtype=torch.bfloat16)
    x[0, :row['tokens']] = hidden.to('cuda')
    return x


def forward(head, row, args):
    with torch.autocast('cuda', dtype=torch.bfloat16):
        z, null = head(features(args, row))
    z = z[0, :row['tokens']].float()
    null = null[0, :row['tokens']].float() if null is not None else None
    assert torch.isfinite(z).all() and (null is None or torch.isfinite(null).all())
    return z, null


def config(args):
    sources = [Path(__file__), Path(curve_heads.__file__), Path(original.__file__)]
    return {'readout': args.readout, 'curve': args.curve, 'event_weight': args.event_weight,
            'attention_before_mamba': args.attention, 'seed': args.seed,
            'data_sha256': args.audit['dialogs_sha256'],
            'template_sha256': args.audit['template_sha256'], 'cache_identity': args.cache_identity,
            'splits_sha256': original.file_hash(args.splits), 'train_split': args.train_split,
            'sources': {p.name: original.file_hash(p) for p in sources}}


def environment():
    packages = {}
    for package in ('torch', 'mamba-ssm', 'causal-conv1d', 'triton', 'transformers'):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None
    return {'packages': packages, 'python': sys.version, 'platform': platform.platform(),
            'cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(),
            'gpu': torch.cuda.get_device_name(), 'capability': torch.cuda.get_device_capability(),
            'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES'),
            'CUBLAS_WORKSPACE_CONFIG': os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
            'tf32': torch.backends.cuda.matmul.allow_tf32,
            'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
            'mamba': 'Mamba2 native use_mem_eff_path=True, chunk128',
            'attention': 'preLN residual causal SDPA, 4x64, dropout0, no FFN/extra position',
            'sdpa_flash': torch.backends.cuda.flash_sdp_enabled(),
            'sdpa_mem_efficient': torch.backends.cuda.mem_efficient_sdp_enabled(),
            'sdpa_math': torch.backends.cuda.math_sdp_enabled()}


def gate(args):
    rows, splits = load_data(args)
    assert not (args.root / 'gate.json').exists(), 'Gate already exists; use a fresh root'
    train = splits[args.train_split]
    assert train and splits['dev_select']
    probe = min((r for r in train if r['spans'][0]['end'] < r['tokens']), key=lambda r: r['tokens'])
    head = make_head(args).eval()
    init = curve_heads.initialization(head)
    x = features(args, probe)
    cut = probe['spans'][0]['end']
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        before, nb = head(x)
        changed = x.clone()
        changed[:, cut:] = 0
        after, na = head(changed)
        causal_error = (before[:, :cut]-after[:, :cut]).abs().max().item()
        null_error = (nb[:, :cut]-na[:, :cut]).abs().max().item() if nb is not None else 0.
        assert causal_error == null_error == 0, 'Noncausal head'
        reload_path = args.root / 'gate_reload.pt'
        torch.save(head.state_dict(), reload_path)
        head.load_state_dict(torch.load(reload_path, weights_only=True))
        loaded, nl = head(x)
        reload_error = (loaded-before).abs().max().item()
        assert reload_error == 0 and (nb is None or torch.equal(nb, nl))
    del x, changed, before, after, nb, na, loaded, nl
    head.train()
    z, null = forward(head, probe, args)
    z.retain_grad()
    if null is not None:
        null.retain_grad()
    token, event = loss_parts(z, null, probe['spans'], args)
    watched = [(n, p) for n, p in head.named_parameters() if
               n == 'readout.weight' or n.startswith('blocks.') or n == 'attention_qkv.weight']
    contributions = {}
    for name, loss in (('token', token), ('event', event)):
        gradients = torch.autograd.grad(loss, [p for _, p in watched], retain_graph=True)
        norms = {n: g.float().norm().item() for (n, _), g in zip(watched, gradients)}
        assert all(math.isfinite(v) for v in norms.values()) and norms['readout.weight'] > 0
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
            assert abs((p.square().sum()/p.sum()).item()-event_logit(z, null, span, args.readout).sigmoid().item()) < 1e-6
    del z, null, token, event, gradients
    head.zero_grad(set_to_none=True)
    longest_tokens = max(r['tokens'] for r in rows)
    torch.cuda.reset_peak_memory_stats()
    # Stress the original maximum length without consuming confirmation inputs
    # or labels. Real cached training features were checked by the probe above.
    x = torch.zeros((1, math.ceil(longest_tokens/128)*128, 2560),
                    device='cuda', dtype=torch.bfloat16)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        z, null = head(x)
    z = z[0, :longest_tokens].float()
    null = null[0, :longest_tokens].float() if null is not None else None
    stress_spans = [{'start': 0, 'end': longest_tokens, 'label': 1}]
    token, event = loss_parts(z, null, stress_spans, args)
    loss = token+args.event_weight*event
    assert torch.isfinite(loss)
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in head.parameters() if p.grad is not None)
    write(args.root / 'gate.json', {'status': 'passed', 'config': config(args), **init,
          'environment': environment(), 'gradient_contributions': contributions,
          'causal_error': causal_error, 'null_causal_error': null_error, 'reload_error': reload_error,
          'loss_support_passed': True, 'probe': probe['key'], 'probe_tokens': probe['tokens'],
          'longest_tokens': longest_tokens, 'longest_backward_input': 'synthetic zeros, original maximum length; no held-out labels or features',
          'longest_backward_peak_bytes': torch.cuda.max_memory_allocated(),
          'optimizer_steps': 0, 'qwen_loaded': False, 'audit_gradients_discarded': True})


def validate(head, rows, args, folder):
    assert rows, 'Empty evaluation split'
    folder.mkdir(parents=True, exist_ok=False)
    tensors = folder / 'tokens'
    tensors.mkdir()
    labels, scores, ces, curves = [], [], [], []
    head.eval()
    with torch.no_grad(), (folder / 'events.jsonl').open('w') as stream:
        for row in tqdm(rows, desc=folder.name, unit='dialog'):
            z, null = forward(head, row, args)
            token, _ = loss_parts(z, null, row['spans'], args)
            curves.append(token.item())
            torch.save({'dialog': row['key'], 'ids': row['ids'], 'ids_sha256': row['ids_sha256'],
                        'tokens': row['tokens'], 'spans': row['spans'], 'raw_token_logits': z.cpu(),
                        'null_logits': null.cpu() if null is not None else None}, tensors / (row['key']+'.pt'))
            for span in row['spans']:
                logit = event_logit(z, null, span, args.readout)
                score = logit.sigmoid().item()
                labels.append(span['label']); scores.append(score)
                ces.append(F.binary_cross_entropy_with_logits(logit, logit.new_tensor(float(span['label']))).item())
                stream.write(json.dumps({'dialog': row['key'], 'span': span, 'event_logit': logit.item(),
                                         'score': score, 'ids_sha256': row['ids_sha256']})+'\n')
    metrics = evaluate(labels, scores)
    pr = metrics.pop('pr_curve')
    for fpr in (.01, .05):
        metrics[f'recall_at_fpr_{fpr}'] = max((p['recall'] for p in pr if p['recall'] is not None
            and p.get('specificity') is not None and 1-p['specificity'] <= fpr+1e-12), default=0.)
    metrics.update(event_bce=sum(ces)/len(ces), token_bce_mean_dialog=sum(curves)/len(curves),
                   dialogs=len(rows), actual_lengths=[r['tokens'] for r in rows])
    quantiles = [0., .05, .25, .5, .75, .95, 1.]
    metrics['class_scores'] = {}
    for y in (0, 1):
        values = torch.tensor([s for s, label in zip(scores, labels) if label == y], dtype=torch.float64)
        metrics['class_scores'][str(y)] = {'count': len(values), 'quantile_levels': quantiles,
            'quantiles': torch.quantile(values, torch.tensor(quantiles, dtype=torch.float64)).tolist() if len(values) else None,
            'fraction_ge_half': (values >= .5).double().mean().item() if len(values) else None}
    write(folder / 'metrics.json', metrics)
    return metrics


def train(args):
    _, splits = load_data(args)
    gate_result = original.read(args.root / 'gate.json')
    assert gate_result['status'] == 'passed' and gate_result['config'] == config(args)
    folder = args.root / 'fit'
    folder.mkdir(parents=True, exist_ok=False)
    head = make_head(args)
    init = curve_heads.initialization(head)
    assert init['full_head_sha256'] == gate_result['full_head_sha256']
    optimizer = torch.optim.AdamW(head.parameters(), lr=3e-4, weight_decay=.01, fused=True)
    train_rows = splits[args.train_split]
    assert train_rows and splits['dev_select']
    write(folder / 'run.json', {'config': config(args), **init, 'environment': environment(),
          'epochs': args.epochs, 'train_dialogs': len(train_rows),
          'train_actual_lengths': {r['key']: r['tokens'] for r in train_rows},
          'normalization': 'mean tokens per action, mean actions per dialog, mean 4 dialogs per optimizer step; final partial window uses actual size',
          'lr': 3e-4, 'weight_decay': .01, 'clip': 1., 'accumulation': 4, 'qwen_loaded': False})
    best = None
    history = []
    started = time.monotonic()
    for epoch in range(1, args.epochs+1):
        if time.time() >= args.deadline_unix:
            write(folder / 'budget_stop.json', {'status': 'budget_stopped', 'epochs_done': len(history)})
            return
        order = list(train_rows)
        random.Random(args.seed+epoch).shuffle(order)
        head.train()
        optimizer.zero_grad(set_to_none=True)
        sum_token = sum_event = 0.
        for i, row in enumerate(tqdm(order, desc=f'epoch{epoch}', unit='dialog')):
            z, null = forward(head, row, args)
            token, event = loss_parts(z, null, row['spans'], args)
            loss = token+args.event_weight*event
            assert torch.isfinite(loss), row['key']
            window = min(4, len(order)-(i//4)*4)
            (loss/window).backward()
            sum_token += token.detach().item(); sum_event += event.detach().item()
            if (i+1) % 4 == 0 or i+1 == len(order):
                torch.nn.utils.clip_grad_norm_(head.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            del z, null, token, event, loss
            if time.time() >= args.deadline_unix:
                torch.save({'head':head.state_dict(),'optimizer':optimizer.state_dict(),
                    'epoch':epoch,'dialogs_done':i+1},folder/'deadline_partial.pt')
                write(folder/'budget_stop.json',{'status':'budget_stopped','partial_epoch':epoch,'dialogs_done':i+1})
                return
        checkpoint = folder / f'head_epoch{epoch}.pt'
        torch.save(head.state_dict(), checkpoint)
        torch.save({'epoch': epoch, 'optimizer': optimizer.state_dict(), 'python_rng': random.getstate(),
                    'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all(),
                    'config': config(args)}, folder / f'training_state_epoch{epoch}.pt')
        checkpoint_hash = original.file_hash(checkpoint)
        metrics = validate(head, splits['dev_select'], args, folder / f'dev_select_epoch{epoch}')
        write(folder / f'checkpoint_epoch{epoch}.json', {'sha256': checkpoint_hash, 'config': config(args)})
        history.append({'epoch': epoch, 'train_token_bce': sum_token/len(order),
                        'train_event_bce': sum_event/len(order), 'dev_select': metrics,
                        'checkpoint_sha256': checkpoint_hash, 'seconds': time.monotonic()-started})
        write(folder / 'history.json', history)
        criterion = (metrics['average_precision'], -metrics['event_bce'])
        if best is None or criterion > best:
            best = criterion
            shutil.copy2(checkpoint, folder / 'selected_head.pt')
            write(folder / 'selection.json', {'epoch': epoch, 'dev_select_event_bce': metrics['event_bce'],
                  'dev_select_ap': metrics['average_precision'],
                  'head_sha256': checkpoint_hash, 'config': config(args),
                  'criterion': 'maximum dev_select AP; ties lower hard event BCE, earliest epoch'})
    write(folder / 'complete.json', {'status': 'complete', 'epochs': args.epochs,
          'seconds': time.monotonic()-started, 'qwen_loaded': False, 'cache_access': 'read only'})


def evaluate_checkpoint(args):
    _, splits = load_data(args)
    assert args.checkpoint is not None, '--checkpoint required for eval'
    selection = original.read(args.root / 'fit' / 'selection.json')
    assert selection['config'] == config(args)
    checkpoint_hash = original.file_hash(args.checkpoint)
    # Even dev_select eval uses the locked selected checkpoint. Other epochs'
    # dev_select outputs are already retained by train.
    assert checkpoint_hash == selection['head_sha256'], 'Eval requires selected checkpoint'
    assert original.read(args.root / 'fit' / 'complete.json')['status'] == 'complete'
    head = make_head(args)
    head.load_state_dict(torch.load(args.checkpoint, map_location='cpu', weights_only=True))
    folder = args.root / 'evaluations' / args.eval_split
    validate(head, splits[args.eval_split], args, folder)
    write(folder / 'protocol.json', {'config': config(args), 'selection': selection,
          'checkpoint_sha256': checkpoint_hash, 'split': args.eval_split,
          'selection_allowed': args.eval_split == 'dev_select', 'threshold': .5,
          'threshold_fit': False, 'environment': environment()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('gate', 'train', 'eval'))
    for name in ('root', 'data', 'cache', 'splits'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--train-split', choices=('train_small', 'train_full'), default='train_small')
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--curve', choices=('broad', 'linear', 'step'), default='broad')
    parser.add_argument('--readout', choices=READOUTS, required=True)
    parser.add_argument('--event-weight', type=float, default=.1)
    parser.add_argument('--deadline-unix', type=float, default=float('inf'))
    parser.add_argument('--attention', action='store_true')
    parser.add_argument('--seed', type=int, default=original.SEED)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--eval-split', choices=('dev_select', 'dev_confirm', 'original_validation'), default='dev_confirm')
    args = parser.parse_args()
    assert math.isfinite(args.event_weight) and args.event_weight > 0 and args.epochs > 0
    args.root.mkdir(parents=True, exist_ok=True)
    try:
        {'gate': gate, 'train': train, 'eval': evaluate_checkpoint}[args.phase](args)
    except Exception:
        write(args.root / f'failure_{args.phase}_{time.time_ns()}.json',
              {'phase': args.phase, 'traceback': traceback.format_exc(), 'status': 'failed'})
        raise


if __name__ == '__main__':
    main()
