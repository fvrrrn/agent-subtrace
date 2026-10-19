"""Validation-selected curve control: frozen Qwen -> Linear(1024,1), no Mamba."""
import argparse
import json
import os
from pathlib import Path
import select
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

import train_action_models as training


def run(root, parent, curves, external, arch='linear_head', linear_reference=None,
        hard_reference=None, transformer_reference=None):
    # Block on the registered experiment completion event, not repeated polling.
    pid = training.read(external / 'launch.json')['pid']
    try:
        fd = os.pidfd_open(pid)
    except ProcessLookupError:
        fd = None
    if fd is not None:
        print('Waiting for four-epoch curve pipeline ' + str(pid), flush=True)
        select.select([fd], [], [])
        os.close(fd)
    assert training.read(curves / 'scheduler_status.json')['status'] == 'complete'
    assert (external / 'curves_complete.json').exists()
    candidates = []
    for objective in ('exponential', 'cosine_rank', 'linear', 'quadratic', 'sqrt', 'cosine_plain'):
        source = (parent if objective in ('exponential', 'cosine_rank') else curves) / 'arms' / ('mamba_' + objective)
        assert (source / 'complete.json').exists()
        selected = training.read(source / 'selection.json')
        assert selected['epoch'] <= 4
        candidates.append({'objective': objective, 'source': str(source), **selected})
    chosen = min(candidates, key=lambda x: (x['validation_bce'], x['objective']))
    if hard_reference is not None:
        assert arch == 'transformer_head'
        source = hard_reference / 'arms/mamba_token_hard'
        assert (source / 'complete.json').exists()
        chosen = {'objective': 'token_hard', 'source': str(source),
                  **training.read(source / 'selection.json')}
    root.mkdir(parents=True, exist_ok=False)
    for name in ('records.json', 'features'):
        (root / name).symlink_to(parent / name, target_is_directory=name == 'features')
    cfg = training.read(parent / 'protocol.json')
    cfg.update(source_sha256=training.sha(training.__file__), records_sha256=training.sha(root / 'records.json'),
               control=arch, transformer_config=({'layers': 1, 'width': 256, 'heads': 4,
                   'ff_width': 256, 'dropout': 0, 'positions': 'sinusoidal',
                   'attention': 'PyTorch SDPA is_causal=True; right padding cannot affect earlier tokens'}
                   if arch == 'transformer_head' else None),
               epochs=4, chosen_curve=chosen, candidate_validation_selections=candidates,
               curve_selection=('User-specified token_hard, no curve selection' if hard_reference is not None else
                   'Minimum original validation endpoint BCE only. StepGuard metrics are not read to choose the curve.'),
               ranking='Same as chosen curve: 0.1 unsafe ranking only if cosine_rank wins, otherwise none')
    training.write(root / 'protocol.json', cfg)
    locked = root / 'reference_mamba'; locked.mkdir()
    source = Path(chosen['source'])
    for filename in ('selected_head.pt', 'selection.json', 'run.json'):
        shutil.copy2(source / filename, locked / filename)
    assert training.sha(locked / 'selected_head.pt') == chosen['head_sha256']
    print('Chosen curve using validation only: ' + json.dumps(chosen), flush=True)

    import torch
    import torch.nn.functional as F
    args = SimpleNamespace(root=root)
    rows = training.read(root / 'records.json')
    row = next(r for r in rows if r['split'] == 'train' and r['tokens'] < 2048)
    torch.manual_seed(cfg['seed'])
    make_head = training.make_transformer_head if arch == 'transformer_head' else training.make_linear_head
    head = make_head()
    parameters = sum(p.numel() for p in head.parameters())
    initial_hash = training.parameter_hash(head)
    if transformer_reference is not None:
        previous = training.read(transformer_reference / 'arms/transformer_head_linear/run.json')
        assert initial_hash == previous['initial_head_sha256']
    # Same-shape suffix perturbation checks the actual readout's causal mask.
    head.eval()
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        x = torch.randn(1, 256, 1024, device='cuda')
        original = head(x)
        changed = x.clone(); changed[:, 128:] = torch.randn_like(changed[:, 128:])
        prefix_error = (original[:, :128] - head(changed)[:, :128]).abs().max().item()
        assert prefix_error == 0
        state = {k: v.clone() for k, v in head.state_dict().items()}
        head.load_state_dict(state)
        reload_error = (original - head(x)).abs().max().item()
        assert reload_error == 0
    del x, changed, original, state
    head.train()
    logits = training.token_forward(args, None, head, [row])
    logits.retain_grad()
    loss = F.binary_cross_entropy_with_logits(logits[0, row['start']:row['tokens']].float(),
                                               training.targets(row, chosen['objective'], 'cuda'))
    loss.backward()
    if hard_reference is not None:
        assert torch.count_nonzero(logits.grad[0, :row['start']]) == 0
        assert torch.count_nonzero(logits.grad[0, row['tokens']:]) == 0
        assert torch.count_nonzero(logits.grad[0, row['start']:row['tokens']]) == row['tokens'] - row['start']
    norms = {n: p.grad.float().norm().item() for n, p in head.named_parameters() if p.grad is not None}
    assert any(v > 0 for v in norms.values()) and all(__import__('math').isfinite(v) for v in norms.values())
    if arch == 'transformer_head':
        assert any(v > 0 for n, v in norms.items() if 'blocks.' in n)
        assert norms['readout.weight'] > 0
    training.write(root / 'cuda_gate.json', {'status': 'passed', 'parameters': parameters,
        'gradient_norms': norms, 'prefix_error': prefix_error, 'reload_error': reload_error,
        'initial_head_sha256': initial_hash, 'objective': chosen['objective'],
        'no_mamba_or_lora': True, 'audit_gradients_discarded': True})
    del head, logits, loss
    torch.cuda.empty_cache()
    command = [sys.executable, '-u', str(Path(training.__file__).resolve()), 'train', '--root', str(root),
               '--arch', arch, '--objective', chosen['objective'], '--epochs', '4']
    training.write(root / 'train_launch.json', {'command': command, 'started_unix': time.time()})
    with (root / 'training.log').open('w') as log:
        p = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        assert p.wait() == 0, 'Linear-probe training failed'

    arm = root / 'arms' / (arch + '_' + chosen['objective'])
    selected = training.read(arm / 'selection.json')
    assert training.sha(arm / 'selected_head.pt') == selected['head_sha256']
    evaluation = root / 'stepguard'; evaluation.mkdir()
    (evaluation / 'records.json').symlink_to(external / 'records.json')
    (evaluation / 'features').symlink_to(external / 'features', target_is_directory=True)
    eval_cfg = dict(cfg, records_sha256=training.sha(evaluation / 'records.json'))
    training.write(evaluation / 'protocol.json', eval_cfg)
    args = SimpleNamespace(root=evaluation, batch_size=4, token_budget=16384)
    head = make_head().eval().requires_grad_(False)
    head.load_state_dict(torch.load(arm / 'selected_head.pt', weights_only=True, map_location='cuda'))
    rows = training.read(evaluation / 'records.json')
    for split in ('external_test', 'external_dev'):
        training.predict(args, None, head, [r for r in rows if r['split'] == split], evaluation, split)
    compare(root, external, chosen, selected, arch, linear_reference, transformer_reference)
    training.write(root / 'status.json', {'status': 'complete', 'curve': chosen['objective'],
                                        'selected_epoch': selected['epoch']})


def compare(root, external, chosen, selected, arch='linear_head', linear_reference=None,
            transformer_reference=None):
    reference = 'mamba_' + chosen['objective']
    assert training.read(external / reference / 'environment.json')['head_sha256'] == chosen['head_sha256']
    result = {'curve_selected_by_validation': chosen, 'architecture': arch, 'selected_epoch': selected['epoch'],
              'limits': 'One seed, previously viewed StepGuard. Same curve/objective; architectures and parameter counts differ. Not a pure isolation of memory benefit.',
              'splits': {}}
    for split in ('external_test', 'external_dev'):
        paths = {arch: root / 'stepguard' / (split + '.jsonl')}
        if linear_reference is not None:
            paths['linear_head'] = linear_reference / 'stepguard' / (split + '.jsonl')
        if transformer_reference is not None:
            paths['transformer_linear'] = transformer_reference / 'stepguard' / (split + '.jsonl')
        paths.update({name: external / name / (split + '.jsonl') for name in ('kev_original', 'random_mamba_head', reference)})
        predictions = {name: {r['key']: r for r in map(json.loads, p.read_text().splitlines())} for name, p in paths.items()}
        common = sorted(set.intersection(*(set(v) for v in predictions.values())))
        for key in common:
            assert len({(v[key]['label'], v[key]['state_sha256']) for v in predictions.values()}) == 1
        metrics = {}
        for name, out in predictions.items():
            m = training.evaluate([out[k]['label'] for k in common], [out[k]['score'] for k in common])
            m.pop('pr_curve'); metrics[name] = m
        result['splits'][split] = {'common': len(common), 'metrics': metrics,
                                  'matching_source_states_and_labels': True}
    training.write(root / 'comparison.json', result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'parent', 'curves', 'external'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--arch', choices=('linear_head', 'transformer_head'), default='linear_head')
    p.add_argument('--linear-reference', type=Path)
    p.add_argument('--hard-reference', type=Path)
    p.add_argument('--transformer-reference', type=Path)
    a = p.parse_args()
    run(a.root, a.parent, a.curves, a.external, a.arch, a.linear_reference,
        a.hard_reference, a.transformer_reference)
