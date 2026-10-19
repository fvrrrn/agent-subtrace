"""Frozen cached Qwen, two residual Mamba layers: selected curve BCE + event BCE."""
import argparse
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path
import random
import shutil
import time

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

import train_whole_dialogs as original
from prepare_whole_dialogs import BASE, REVISION, digest, write
from graph_safety.metrics import evaluate

ARMS = ('curve_null', 'curve_linear_softmax')
EVENT_WEIGHT = .1


class Head(original.Head):
    def __init__(self, arm):
        super().__init__()
        self.null_readout = torch.nn.Linear(256, 1) if arm == 'curve_null' else None
        if self.null_readout is not None:
            torch.nn.init.zeros_(self.null_readout.weight)
            torch.nn.init.zeros_(self.null_readout.bias)

    def forward(self, x):
        x = self.projection(self.input_norm(x))
        for norm, block in zip(self.norms, self.blocks):
            x = x + block(norm(x))
        x = self.final_norm(x)
        return self.readout(x).squeeze(-1), (self.null_readout(x).squeeze(-1)
                                           if self.null_readout is not None else None)


def features(root, row):
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        saved = torch.load(root / 'features' / (row['key']+'.pt'), map_location='cpu', weights_only=True)
    assert saved['ids_sha256'] == row['ids_sha256']
    assert saved['identity'] == original.cache_identity(str(root))
    n = math.ceil(row['tokens']/128)*128
    x = torch.zeros((1, n, 2560), device='cuda', dtype=torch.bfloat16)
    x[0, :row['tokens']] = saved['hidden'].cuda()
    return x


def forward(head, row, root):
    with torch.autocast('cuda', dtype=torch.bfloat16):
        z, null = head(features(root, row))
    return z[0, :row['tokens']].float(), null[0, :row['tokens']].float() if null is not None else None


def profile(n, device):
    u = torch.arange(1, n+1, device=device, dtype=torch.float32)/n
    return (.9+1.2*u-1.5*u.square())/1.14


def event_logit(z, null, span, arm):
    values = z[span['start']:span['end']]
    if arm == 'curve_null':
        return torch.logsumexp(values, 0)-math.log(len(values))-null[span['end']-1]
    # Exact logit of sum(p²)/sum(p), without clipping or probability underflow:
    # s/(1-s) = sum(p²)/sum(p*(1-p)).
    positive, negative = F.logsigmoid(values), F.logsigmoid(-values)
    return torch.logsumexp(2*positive, 0)-torch.logsumexp(positive+negative, 0)


def loss_parts(z, null, spans, arm):
    curves, events = [], []
    for span in spans:
        values = z[span['start']:span['end']]
        target = float(span['label'])*profile(len(values), z.device)
        curves.append(F.binary_cross_entropy_with_logits(values, target))
        score_logit = event_logit(z, null, span, arm)
        events.append(F.binary_cross_entropy_with_logits(score_logit, score_logit.new_tensor(float(span['label']))))
    return torch.stack(curves).mean(), torch.stack(events).mean()


def initialization(head):
    def hashes(include_null):
        return {name: hashlib.sha256(value.detach().cpu().numpy().tobytes()).hexdigest()
                for name, value in head.state_dict().items() if include_null or not name.startswith('null_readout.')}
    return {'shared_head_sha256': digest(hashes(False)), 'full_head_sha256': digest(hashes(True))}


def data(args):
    rows, audit = original.load_data(args.data)
    assert audit['dialogs_sha256'] == original.read(args.root / 'protocol.json')['data_sha256']
    for shard in (0, 1):
        complete = original.read(args.root / 'features' / f'complete{shard}.json')
        assert complete['dialogs_sha256'] == audit['dialogs_sha256']
        assert complete['revision'] == REVISION and complete['template_sha256'] == audit['template_sha256']
    return rows


def gate(args):
    rows = data(args)
    probe = next(r for r in rows if r['split'] == 'train' and len(r['spans']) > 1 and r['tokens'] < 4096)
    torch.manual_seed(original.SEED)
    head = Head(args.arm).cuda().eval()
    init = initialization(head)
    x = features(args.root, probe)
    cut = probe['spans'][0]['end']
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        before, nb = head(x)
        changed = x.clone()
        changed[:, cut:] = 0
        after, na = head(changed)
        error = (before[:, :cut]-after[:, :cut]).abs().max().item()
        null_error = (nb[:, :cut]-na[:, :cut]).abs().max().item() if nb is not None else 0.
        assert error == null_error == 0
        state = {k: v.detach().clone() for k, v in head.state_dict().items()}
        head.load_state_dict(state)
        loaded, nl = head(x)
        reload_error = (loaded-before).abs().max().item()
        assert reload_error == 0
        if nb is not None:
            assert (nb-nl).abs().max().item() == 0
    del x, changed, before, after, nb, na, loaded, nl, state
    head.train()
    z, null = forward(head, probe, args.root)
    z.retain_grad()
    if null is not None:
        null.retain_grad()
    curve, event = loss_parts(z, null, probe['spans'], args.arm)
    assert torch.isfinite(curve+event)
    curve_gradient = torch.autograd.grad(curve, head.readout.weight, retain_graph=True)[0].float().norm().item()
    event_gradient = torch.autograd.grad(event, head.readout.weight, retain_graph=True)[0].float().norm().item()
    assert curve_gradient > 0 and event_gradient > 0
    (curve+EVENT_WEIGHT*event).backward()
    mask = torch.zeros_like(z, dtype=torch.bool)
    endpoints = torch.zeros_like(z, dtype=torch.bool)
    for s in probe['spans']:
        mask[s['start']:s['end']] = True
        endpoints[s['end']-1] = True
    assert torch.count_nonzero(z.grad[~mask]) == 0
    if null is not None:
        assert torch.count_nonzero(null.grad[~endpoints]) == 0
    gradients = {n: p.grad.float().norm().item() for n, p in head.named_parameters() if p.grad is not None}
    assert all(math.isfinite(v) for v in gradients.values())
    for i in (0, 1):
        assert any(v > 0 for n, v in gradients.items() if n.startswith(f'blocks.{i}.'))
    assert gradients['readout.weight'] > 0
    if null is not None:
        assert gradients['null_readout.weight'] > 0
    else:
        for s in probe['spans']:
            p = z[s['start']:s['end']].detach().sigmoid()
            assert abs((p.square().sum()/p.sum()).item()-event_logit(z, null, s, args.arm).sigmoid().item()) < 1e-6
    longest = max(rows, key=lambda r: r['tokens'])
    head.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats()
    zl, nl = forward(head, longest, args.root)
    cl, el = loss_parts(zl, nl, longest['spans'], args.arm)
    (cl+EVENT_WEIGHT*el).backward()
    assert all(torch.isfinite(p.grad).all() for p in head.parameters() if p.grad is not None)
    write(args.root / (args.arm+'_gate.json'), {'status': 'passed', **init, 'gradients': gradients,
        'suffix_error': error, 'null_suffix_error': null_error, 'reload_error': reload_error,
        'loss_support_passed': True, 'probe_curve_bce': curve.item(), 'probe_event_bce': event.item(),
        'curve_readout_gradient_norm': curve_gradient, 'event_readout_gradient_norm': event_gradient,
        'longest_tokens': longest['tokens'], 'longest_backward_peak_bytes': torch.cuda.max_memory_allocated(),
        'qwen_loaded': False, 'optimizer_steps': 0, 'audit_gradients_discarded': True})


def validate(head, rows, args, folder, epoch):
    head.eval()
    labels, scores, ces, curve_losses = [], [], [], []
    tensors = folder / f'validation_tokens_epoch{epoch}'
    tensors.mkdir()
    with (folder / f'validation_epoch{epoch}.jsonl').open('w') as stream, torch.no_grad():
        for row in tqdm(rows, desc=f'{args.arm} validation{epoch}', unit='dialog'):
            z, null = forward(head, row, args.root)
            c, _ = loss_parts(z, null, row['spans'], args.arm)
            curve_losses.append(c.item())
            torch.save({'dialog': row['key'], 'ids_sha256': row['ids_sha256'],
                        'raw_token_logits': z.cpu(), 'null_logits': null.cpu() if null is not None else None,
                        'spans': row['spans']}, tensors / (row['key']+'.pt'))
            for s in row['spans']:
                logit = event_logit(z, null, s, args.arm)
                score = logit.sigmoid().item()
                ce = F.binary_cross_entropy_with_logits(logit, logit.new_tensor(float(s['label']))).item()
                labels.append(s['label']); scores.append(score); ces.append(ce)
                stream.write(json.dumps({'key': s['key'], 'dialog': row['key'], 'label': s['label'],
                    'score': score, 'event_logit': logit.item(), 'token_endpoint': z[s['end']-1].sigmoid().item(),
                    'state_sha256': s['state_sha256'], 'ids_sha256': row['ids_sha256']})+'\n')
    metrics = evaluate(labels, scores)
    metrics.pop('pr_curve')
    metrics['event_bce'] = sum(ces)/len(ces)
    metrics['curve_bce_mean_dialog'] = sum(curve_losses)/len(curve_losses)
    return metrics


def train(args):
    rows = data(args)
    assert original.read(args.root / (args.arm+'_gate.json'))['status'] == 'passed'
    folder = args.root / 'arms' / args.arm
    folder.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(original.SEED)
    random.seed(original.SEED)
    head = Head(args.arm).cuda()
    init = initialization(head)
    assert init['full_head_sha256'] == original.read(args.root / (args.arm+'_gate.json'))['full_head_sha256']
    opt = torch.optim.AdamW(head.parameters(), lr=3e-4, weight_decay=.01, fused=True)
    write(folder / 'run.json', {**init, 'arm': args.arm, 'event_weight': EVENT_WEIGHT,
        'seed': original.SEED, 'epochs': 2, 'qwen_loaded': False, 'lora': False,
        'parameters': sum(p.numel() for p in head.parameters()), 'source_sha256': original.file_hash(Path(__file__)),
        'torch': str(torch.__version__), 'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(),
        'mamba_ssm': version('mamba-ssm'), 'triton': version('triton'),
        'tf32': torch.backends.cuda.matmul.allow_tf32})
    train_rows = [r for r in rows if r['split'] == 'train']
    validation = [r for r in rows if r['split'] == 'validation']
    best = float('inf'); history = []; started = time.monotonic()
    for epoch in (1, 2):
        order = list(train_rows)
        random.Random(original.SEED+epoch).shuffle(order)
        head.train(); opt.zero_grad(set_to_none=True)
        sum_curve = sum_event = 0.
        for i, row in enumerate(tqdm(order, desc=f'{args.arm} epoch{epoch}/2', unit='dialog')):
            z, null = forward(head, row, args.root)
            curve, event = loss_parts(z, null, row['spans'], args.arm)
            window = min(4, len(order)-(i//4)*4)
            ((curve+EVENT_WEIGHT*event)/window).backward()
            sum_curve += curve.detach().item(); sum_event += event.detach().item()
            if (i+1)%4 == 0 or i+1 == len(order):
                norm = torch.nn.utils.clip_grad_norm_(head.parameters(), 1., error_if_nonfinite=True)
                if epoch == 1 and i < 4:
                    write(folder / 'first_gradient.json', {'norm': norm.item(), 'finite': True})
                opt.step(); opt.zero_grad(set_to_none=True)
            del z, null, curve, event
        metrics = validate(head, validation, args, folder, epoch)
        path = folder / f'head_epoch{epoch}.pt'
        torch.save(head.state_dict(), path)
        history.append({'epoch': epoch, 'train_curve_bce': sum_curve/len(order),
            'train_event_bce': sum_event/len(order), 'validation': metrics, 'seconds': time.monotonic()-started})
        write(folder / 'history.json', history)
        if metrics['event_bce'] < best:
            best = metrics['event_bce']
            shutil.copy2(path, folder / 'selected_head.pt')
            write(folder / 'selection.json', {'epoch': epoch, 'validation_event_bce': best,
                'head_sha256': original.file_hash(folder / 'selected_head.pt'), 'criterion': 'hard event validation BCE'})
    write(folder / 'complete.json', {'status': 'complete', 'epochs': 2, 'seconds': time.monotonic()-started,
                                   'frozen_cache_unchanged': True, 'qwen_loaded': False})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('gate', 'train'))
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--arm', choices=ARMS, required=True)
    args = parser.parse_args()
    {'gate': gate, 'train': train}[args.phase](args)
