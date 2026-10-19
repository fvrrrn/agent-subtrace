"""Frozen Qwen features: forecast + auto-pool/subsequence event classification."""
import argparse
import hashlib
import math
from pathlib import Path
import time
import traceback

import torch
import torch.nn.functional as F

import train_autoresearch_heads as base
from prepare_whole_dialogs import digest, write


class Head(base.curve_heads.Head):
    def __init__(self, arm):
        super().__init__('curve_linear_softmax', False)
        self.arm = arm
        if arm == 'B':
            self.evidence = torch.nn.Linear(256, 1)
            self.alpha_raw = torch.nn.Parameter(torch.tensor(math.log(math.expm1(1.))))
        else:
            self.window = torch.nn.Linear(768, 32)
            self.event = torch.nn.Linear(352, 1)

    def forward(self, x):
        x = self.projection(self.input_norm(x))
        for norm, block in zip(self.norms, self.blocks):
            x = x + block(norm(x))
        h = self.final_norm(x)
        return self.readout(h).squeeze(-1), h

    def event_logit(self, h, span):
        h = h[span['start']:span['end']].float()
        if self.arm == 'B':
            r = self.evidence(h).squeeze(-1)
            weights = F.softplus(self.alpha_raw) * r.sigmoid()
            return torch.logsumexp(weights + F.logsigmoid(r), 0) - torch.logsumexp(weights + F.logsigmoid(-r), 0)
        cumulative = F.pad(h.cumsum(0), (0, 0, 1, 0))
        end = torch.arange(len(h), device=h.device)
        pooled = []
        for width in (8, 32, 128):
            left = (end - width + 1).clamp_min(0)
            mean = (cumulative[end + 1] - cumulative[left]) / (end - left + 1)[:, None]
            value = torch.cat((mean, h - h[left], h), -1)
            pooled.append(F.relu(self.window(value)).amax(0))
        return self.event(torch.cat((h[-1], *pooled))).squeeze(-1)


ACTIVE = None
BASE_CONFIG = base.config
BASE_VALIDATE = base.validate
BASE_ENVIRONMENT = base.environment


def environment():
    result = BASE_ENVIRONMENT()
    result['attention'] = 'none'
    return result


def guard(args):
    if time.time() >= args.deadline_unix:
        raise TimeoutError('Training/validation deadline reached')


def make_head(args):
    global ACTIVE
    base.seed(args.seed)
    ACTIVE = Head(args.arm).cuda()
    return ACTIVE


def forward(head, row, args):
    guard(args)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        z, h = head(base.features(args, row))
    z, h = z[0, :row['tokens']].float(), h[0, :row['tokens']].float()
    assert torch.isfinite(z).all() and torch.isfinite(h).all()
    return z, h


def event_logit(z, h, span, readout):
    return ACTIVE.event_logit(h, span)


def loss_parts(z, h, spans, args):
    tokens, events = [], []
    for span in spans:
        values = z[span['start']:span['end']]
        tokens.append(F.binary_cross_entropy_with_logits(values, torch.full_like(values, float(span['label']))))
        logit = event_logit(z, h, span, args.readout)
        events.append(F.binary_cross_entropy_with_logits(logit, logit.new_tensor(float(span['label']))))
    return torch.stack(tokens).mean(), torch.stack(events).mean()


def config(args):
    result = BASE_CONFIG(args)
    result.update(arm=args.arm, token_target='forecast action Y', windows=[8, 32, 128] if args.arm == 'C' else None,
                  source_sha256=base.original.file_hash(Path(__file__)), attention_before_mamba=False,
                  control_gate_sha256=base.original.file_hash(args.control_gate))
    return result


def validate(head, rows, args, folder):
    # Parent evaluator retains every raw forecast logit, full IDs/spans and raw
    # event logits. Its legacy null_logits field contains h here, not NULL logits.
    guard(args)
    result = BASE_VALIDATE(head, rows, args, folder)
    write(folder / 'tensor_schema.json', {'raw_token_logits': 'forecast logits',
          'null_logits': 'legacy field name: normalized Mamba states, not NULL logits',
          'events.jsonl': 'raw event logits, scores and action spans'})
    return result


def gate(args):
    guard(args)
    _, splits = base.load_data(args)
    train = splits['train_small']
    assert len(train) == 2009 and len(splits['dev_select']) == 301
    head = make_head(args).eval()
    core = {n: v for n, v in head.state_dict().items()
            if not n.startswith(('evidence.', 'window.', 'event.')) and n != 'alpha_raw'}
    core_hash = digest({n: hashlib.sha256(v.detach().cpu().numpy().tobytes()).hexdigest() for n, v in core.items()})
    control = base.original.read(args.control_gate)
    assert control['status'] == 'passed'
    assert core_hash == control['shared_head_sha256'], 'Shared initialization differs from A'
    init = base.curve_heads.initialization(head)
    probe = min((r for r in train if r['spans'][0]['end'] < r['tokens']), key=lambda r: r['tokens'])
    x = base.features(args, probe)
    cut = probe['spans'][0]['end']
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        z, h = head(x)
        changed = x.clone()
        changed[:, cut:] = 0
        za, ha = head(changed)
        assert torch.equal(z[:, :cut], za[:, :cut]) and torch.equal(h[:, :cut], ha[:, :cut])
    with torch.no_grad():
        before = head.event_logit(h[0].float(), probe['spans'][0])
        after = head.event_logit(ha[0].float(), probe['spans'][0])
        assert torch.equal(before, after)
    torch.save(head.state_dict(), args.root / 'gate_reload.pt')
    head.load_state_dict(torch.load(args.root / 'gate_reload.pt', weights_only=True))
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        zr, hr = head(x)
        assert torch.equal(z, zr) and torch.equal(h, hr)
    with torch.no_grad():
        assert torch.equal(before, head.event_logit(hr[0].float(), probe['spans'][0]))
    del x, changed, z, h, za, ha, zr, hr
    head.train()
    z, h = forward(head, probe, args)
    z.retain_grad()
    h.retain_grad()
    token, event = loss_parts(z, h, probe['spans'], args)
    norms = {}
    for kind, loss in [('token', token), ('event', event)]:
        watched = [(n, p) for n, p in head.named_parameters() if n.startswith('blocks.') or
                   (kind == 'token' and n.startswith('readout.')) or
                   (kind == 'event' and (n.startswith(('evidence.', 'window.', 'event.')) or n == 'alpha_raw'))]
        gradients = torch.autograd.grad(loss, [p for _, p in watched], retain_graph=True)
        norms[kind] = {n: g.float().norm().item() for (n, _), g in zip(watched, gradients)}
        assert all(math.isfinite(v) for v in norms[kind].values())
        for prefix in ('blocks.0.', 'blocks.1.', 'readout.' if kind == 'token' else ('evidence.' if args.arm == 'B' else 'window.')):
            assert any(v > 0 for n, v in norms[kind].items() if n.startswith(prefix))
        if kind == 'event':
            assert norms[kind]['alpha_raw' if args.arm == 'B' else 'event.weight'] > 0
    # Direct output support, independent of valid recurrent history gradients.
    zd, hd = z.detach().requires_grad_(), h.detach().requires_grad_()
    td, ed = loss_parts(zd, hd, probe['spans'], args)
    gz, gh = torch.autograd.grad(td + .1 * ed, (zd, hd))
    mask = torch.zeros_like(z, dtype=torch.bool)
    for span in probe['spans']:
        mask[span['start']:span['end']] = True
    assert torch.count_nonzero(gz[~mask]) == torch.count_nonzero(gh[~mask]) == 0
    (token + .1 * event).backward()
    del z, h, zd, hd, td, ed, gz, gh, token, event, gradients
    head.zero_grad(set_to_none=True)
    longest = max(train, key=lambda r: r['tokens'])
    assert longest['tokens'] == 15322
    torch.cuda.reset_peak_memory_stats()
    z, h = forward(head, longest, args)
    token, event = loss_parts(z, h, longest['spans'], args)
    loss = token + .1 * event
    assert torch.isfinite(loss)
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in head.parameters() if p.grad is not None)
    env = base.environment()
    env['attention'] = 'none'
    write(args.root / 'gate.json', {'status': 'passed', 'config': config(args), **init,
          'shared_control_core_sha256': core_hash, 'environment': env, 'gradient_contributions': norms,
          'causal_error': 0, 'reload_error': 0, 'loss_support_passed': True,
          'longest_tokens': longest['tokens'], 'longest_key': longest['key'],
          'longest_backward_input': 'real cached train_small features',
          'longest_backward_peak_bytes': torch.cuda.max_memory_allocated(),
          'parameters': sum(p.numel() for p in head.parameters()), 'qwen_loaded': False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('gate', 'train'))
    for name in ('root', 'data', 'cache', 'splits', 'control-gate'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--arm', choices=('B', 'C'), required=True)
    parser.add_argument('--deadline-unix', type=float, required=True)
    args = parser.parse_args()
    args.seed, args.epochs, args.event_weight = 20261017, 2, .1
    args.train_split, args.readout, args.curve, args.attention = 'train_small', args.arm, 'step', False
    args.root.mkdir(parents=True, exist_ok=True)
    base.make_head, base.forward, base.event_logit = make_head, forward, event_logit
    base.loss_parts, base.config, base.validate = loss_parts, config, validate
    base.environment = environment
    try:
        (gate if args.phase == 'gate' else base.train)(args)
    except Exception:
        write(args.root / f'failure_{args.phase}_{time.time_ns()}.json',
              {'status': 'failed', 'traceback': traceback.format_exc()})
        raise


if __name__ == '__main__':
    main()
