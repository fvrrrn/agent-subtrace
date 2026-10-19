"""Add frozen-Mamba token-target controls without changing existing experiments."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import train_action_models as training


def prepare(root, parent, objectives=('token_hard', 'endpoint_hard'), epochs=8):
    import torch
    import torch.nn.functional as F
    parent_cfg = training.read(parent / 'protocol.json')
    cached = training.read(parent / 'features/complete.json')
    assert cached['records_sha256'] == parent_cfg['records_sha256'] == training.sha(parent / 'records.json')
    assert training.read(parent / 'cuda_gate.json')['status'] == 'passed'
    root.mkdir(parents=True, exist_ok=False)
    for name in ('records.json', 'features', 'baselines'):
        (root / name).symlink_to(parent / name, target_is_directory=name != 'records.json')
    cfg = dict(parent_cfg, source_sha256=training.sha(training.__file__),
               parent=str(parent), parent_protocol_sha256=training.sha(parent / 'protocol.json'),
               conditions=list(objectives), epochs=epochs,
               objective='Explicit token/action target conditions, no ranking; see per-arm objective.',
               ranking='Disabled for all conditions in this extension',
               curve_definitions={'linear': 'a+(y-a)*u', 'quadratic': 'a+(y-a)*u^2',
                                  'sqrt': 'a+(y-a)*sqrt(u)', 'cosine_plain': 'a+(y-a)*(1-cos(pi*u))/2',
                                  'token_hard': 'y on every action token', 'endpoint_hard': 'y on last action token only'},
               curve_anchor='previous verified annotated hard label, initially zero; not an inferred safe-history label',
               cache_reused_without_modification=True)
    training.write(root / 'protocol.json', cfg)
    args = SimpleNamespace(root=root)
    rows = training.read(root / 'records.json')
    sample = [next(r for r in rows if r['split'] == 'train' and r['label'] == y and r['tokens'] <= 2048)
              for y in (0, 1)]
    audits = {}
    for objective in cfg['conditions']:
        torch.manual_seed(cfg['seed'])
        head = training.make_head(1)
        init_hash = training.parameter_hash(head)
        logits = training.token_forward(args, None, head, sample)
        if objective != 'endpoint_hard':
            loss = torch.stack([F.binary_cross_entropy_with_logits(
                logits[i, r['start']:r['tokens']].float(), training.targets(r, objective, 'cuda'))
                for i, r in enumerate(sample)]).mean()
        else:
            loss = torch.stack([F.binary_cross_entropy_with_logits(logits[i, r['tokens'] - 1].float(),
                logits.new_tensor(float(r['label']), dtype=torch.float32)) for i, r in enumerate(sample)]).mean()
        direct = torch.autograd.grad(loss, logits, retain_graph=True)[0]
        for i, row in enumerate(sample):
            start = row['tokens'] - 1 if objective == 'endpoint_hard' else row['start']
            assert (direct[i, :start] == 0).all()
            assert (direct[i, row['tokens']:] == 0).all()
            assert (direct[i, start:row['tokens']] != 0).all()
        loss.backward()
        norms = {n: p.grad.float().norm().item() for n, p in head.named_parameters() if p.grad is not None}
        assert all(__import__('math').isfinite(v) for v in norms.values())
        assert any(v > 0 for n, v in norms.items() if n.startswith('blocks.'))
        assert any(v > 0 for n, v in norms.items() if n.startswith('readout.'))
        audits[objective] = {'status': 'passed', 'initial_head_sha256': init_hash,
                            'loss': loss.item(), 'gradient_norms': norms,
                            'direct_loss_gradient_only_on_authorized_tokens': True}
        del head, logits, direct, loss
    expected = training.read(parent / 'baselines' / 'random_mamba_head.json')
    assert expected['n_examples'] == sum(r['split'] == 'validation' for r in rows)
    assert len({a['initial_head_sha256'] for a in audits.values()}) == 1
    trained_init = training.read(parent / 'arms/mamba_exponential/run.json')['initial_head_sha256']
    assert next(iter(audits.values()))['initial_head_sha256'] == trained_init
    training.write(root / 'cuda_gate.json', {'status': 'passed', 'audits': audits,
                   'parent_gate_sha256': training.sha(parent / 'cuda_gate.json'),
                   'parent_head_init_matches': True, 'sample_train_keys': [r['key'] for r in sample],
                   'no_optimizer_steps_reused': True})
    print(json.dumps({'status': 'prepared_and_audited', 'root': str(root),
                      'conditions': cfg['conditions'], 'shared_cache': str(parent / 'features')}, indent=2))


def run(root):
    cfg = training.configuration(root)
    assert training.read(root / 'cuda_gate.json')['status'] == 'passed'
    logs = root / 'logs'; logs.mkdir(exist_ok=False)
    commands = {objective: [sys.executable, '-u', str(Path(training.__file__).resolve()), 'train',
                           '--root', str(root), '--arch', 'mamba', '--objective', objective,
                           '--epochs', str(cfg['epochs'])]
                for objective in cfg['conditions']}
    training.write(root / 'launch_manifest.json', {'pid': os.getpid(), 'started_unix': time.time(),
        'gpu': 2, 'concurrent_conditions': commands, 'epochs': cfg['epochs'], 'max_concurrent_jobs': 2,
        'protocol_sha256': training.sha(root / 'protocol.json'), 'source_sha256': training.sha(training.__file__),
        'launcher_sha256': training.sha(__file__), 'parent_run_not_modified': True})

    def one(name):
        with (logs / (name + '.log')).open('w') as log:
            p = subprocess.Popen(commands[name], env=dict(os.environ, CUDA_VISIBLE_DEVICES='2'),
                                 stdout=log, stderr=subprocess.STDOUT)
            training.write(logs / (name + '_start.json'), {'pid': p.pid, 'command': commands[name]})
            print('START ' + name + ' pid=' + str(p.pid), flush=True)
            code = p.wait()
        training.write(logs / (name + '_status.json'), {'returncode': code, 'finished_unix': time.time()})
        print('END ' + name + ' returncode=' + str(code), flush=True)
        return code == 0

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(one, name) for name in commands]
        success = all([f.result() for f in futures])
    if success:
        training.compare(SimpleNamespace(root=root))
    training.write(root / 'scheduler_status.json', {'status': 'complete' if success else 'failed',
                                                   'finished_unix': time.time()})
    if not success:
        raise SystemExit(1)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=('prepare', 'run'))
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--parent', type=Path)
    p.add_argument('--objectives', nargs='+', choices=('token_hard', 'endpoint_hard', 'linear', 'quadratic', 'sqrt', 'cosine_plain'),
                   default=['token_hard', 'endpoint_hard'])
    p.add_argument('--epochs', type=int, default=8)
    a = p.parse_args()
    if a.phase == 'prepare':
        if a.parent is None:
            p.error('prepare requires --parent')
        if a.epochs <= 0 or len(set(a.objectives)) != len(a.objectives):
            p.error('Positive epoch count and unique conditions required')
        prepare(a.root, a.parent, a.objectives, a.epochs)
    else:
        run(a.root)
