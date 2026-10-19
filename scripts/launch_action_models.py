"""CUDA admission audit and explicit two-GPU queues for the authorized action run."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import train_action_models as training


def gate(root):
    import torch
    import torch.nn.functional as F
    from peft import get_peft_model_state_dict, set_peft_model_state_dict
    cfg = training.configuration(root)
    args = SimpleNamespace(root=root, attention='sdpa', lora_rank=16, checkpointing=False)
    rows = training.read(root / 'records.json')
    sample = next(r for r in rows if r['split'] == 'train' and r['previous_unsafe'] and r['tokens'] < 2048)
    by_key = {r['key']: r for r in rows}
    earlier = by_key[sample['previous_unsafe']]
    audits = {}
    torch.manual_seed(cfg['seed'])
    for arch in ('mamba', 'mamba_lora', 'mamba2_lora'):
        print('CUDA audit: ' + arch, flush=True)
        model = training.backbone(args, arch != 'mamba')
        head = training.make_head(2 if arch == 'mamba2_lora' else 1)
        frozen_before = training.frozen_hash(model)
        model.train(arch != 'mamba')
        head.train()
        logits = training.token_forward(args, model, head, [sample])
        endpoint = logits[0, sample['tokens'] - 1].float()
        ce = F.binary_cross_entropy_with_logits(logits[0, sample['start']:sample['tokens']].float(),
                                                training.targets(sample, 'cosine_rank', 'cuda'))
        prior = training.token_forward(args, model, head, [earlier])[0, earlier['tokens'] - 1].float()
        loss = ce + .1 * F.relu(.2 + prior - endpoint)
        loss.backward()
        norms = {prefix + n: p.grad.detach().float().norm().item()
                 for prefix, module in [('base.', model), ('head.', head)]
                 for n, p in module.named_parameters() if p.requires_grad and p.grad is not None}
        assert norms and all(__import__('math').isfinite(v) for v in norms.values())
        assert any(v > 0 for n, v in norms.items() if 'blocks.' in n)
        assert any(v > 0 for n, v in norms.items() if 'readout.' in n)
        if arch != 'mamba':
            assert any(v > 0 for n, v in norms.items() if 'lora_' in n)
        assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
        assert training.frozen_hash(model) == frozen_before
        model.eval(); head.eval()
        with torch.no_grad():
            original = training.token_forward(args, model, head, [sample])
            state = {k: v.detach().clone() for k, v in head.state_dict().items()}
            head.load_state_dict(state)
            if arch != 'mamba':
                adapter = get_peft_model_state_dict(model)
                set_peft_model_state_dict(model, adapter)
            restored = training.token_forward(args, model, head, [sample])
        reload_error = (original - restored).abs().max().item()
        assert reload_error == 0
        # Check a real input prefix against full-prefix encoding and the downstream recurrent readout.
        cut = min(sample['start'], sample['tokens'] // 2)
        ids = torch.tensor([sample['ids']], device='cuda')
        with torch.no_grad():
            full = model(input_ids=ids, use_cache=False).last_hidden_state
            short = model(input_ids=ids[:, :cut], use_cache=False).last_hidden_state
            relative = ((full[:, :cut].float() - short.float()).norm() / short.float().norm()).item()
            with torch.autocast('cuda', dtype=torch.bfloat16):
                full_z = head(full)
                short_z = head(short)
            changed_ids = ids.clone()
            changed_ids[:, cut:] = (changed_ids[:, cut:] + 123) % model.config.vocab_size
            changed_hidden = model(input_ids=changed_ids, use_cache=False).last_hidden_state
            with torch.autocast('cuda', dtype=torch.bfloat16):
                changed_z = head(changed_hidden)
        prefix_error = (full_z[:, :cut] - short_z).abs().max().item()
        future_encoder_error = (full[:, :cut] - changed_hidden[:, :cut]).abs().max().item()
        future_readout_error = (full_z[:, :cut] - changed_z[:, :cut]).abs().max().item()
        print(f'{arch}: prefix relative_l2={relative}, readout_max_error={prefix_error}', flush=True)
        if not (relative < .03 and future_encoder_error == 0 and future_readout_error == 0):
            training.write(root / 'cuda_gate_failure.json', {'status': 'failed', 'arch': arch,
                'encoder_prefix_relative_l2': relative, 'readout_prefix_error': prefix_error,
                'same_shape_future_encoder_error': future_encoder_error,
                'same_shape_future_readout_error': future_readout_error,
                'cut': cut, 'tokens': sample['tokens'], 'sample_key': sample['key'],
                'source_sha256': training.sha(__file__), 'training_started': False})
        assert relative < .03 and future_encoder_error == 0 and future_readout_error == 0
        audits[arch] = {'status': 'passed', 'loss': loss.item(), 'gradient_norms': norms,
                        'frozen_sha256': frozen_before, 'reload_error': reload_error,
                        'encoder_prefix_relative_l2': relative, 'readout_prefix_error': prefix_error,
                        'same_shape_future_encoder_error': future_encoder_error,
                        'same_shape_future_readout_error': future_readout_error}
        del model, head, logits, prior, endpoint, ce, loss, original, restored, state, full, short, full_z, short_z
        del changed_hidden, changed_z, changed_ids
        if arch != 'mamba':
            del adapter
        gc.collect(); torch.cuda.empty_cache()
    print('CUDA audit: kev', flush=True)
    cp, tok, model = training.native_setup(args, True)
    encs, rejects = training.native_admission(args, tok, model, [sample, earlier])
    assert not rejects
    model.train()
    before = training.frozen_hash(model)
    z = training.native_logodds(model, encs[sample['key']])
    early_z = training.native_logodds(model, encs[earlier['key']])
    loss = F.binary_cross_entropy_with_logits(z, z.new_tensor(1.)) + .1 * F.relu(.2 + early_z - z)
    loss.backward()
    norms = {n: p.grad.detach().float().norm().item() for n, p in model.named_parameters()
             if p.requires_grad and p.grad is not None}
    assert all(__import__('math').isfinite(v) for v in norms.values())
    assert any(v > 0 for n, v in norms.items() if 'lora_' in n)
    assert any(v > 0 for n, v in norms.items() if n.startswith('head.'))
    assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
    assert training.frozen_hash(model) == before
    model.eval()
    with torch.no_grad():
        prediction = training.native_logits(model, encs[sample['key']])
    audits['kev'] = {'status': 'passed', 'gradient_norms': norms, 'temperature': model.head.temperature,
                     'loss': loss.item(), 'frozen_sha256': before, 'eval_raw_logits': prediction.cpu().tolist(),
                     'parent_weights_sha256': cp.weights_sha256()}
    training.write(root / 'cuda_gate.json', {'status': 'passed', 'audits': audits,
        'sample_train_keys': [earlier['key'], sample['key']],
        'note': 'Training data only; audit gradients discarded, all training arms reload fresh weights.',
        'source_sha256': training.sha(__file__), 'torch': torch.__version__, 'gpu': torch.cuda.get_device_name()})
    print('All four CUDA audits passed', flush=True)


def run(root):
    training.configuration(root)
    assert training.read(root / 'cuda_gate.json')['status'] == 'passed'
    script = Path(training.__file__).resolve()
    commands = {
        2: [('cache', []), ('baselines', []),
            ('train', ['--arch', 'mamba', '--objective', 'exponential']),
            ('train', ['--arch', 'mamba', '--objective', 'cosine_rank']),
            ('train', ['--arch', 'kev', '--objective', 'noul_rank', '--epochs', '2'])],
        3: [('train', ['--arch', arch, '--objective', objective])
            for arch in ('mamba_lora', 'mamba2_lora') for objective in ('exponential', 'cosine_rank')],
    }
    planned = {str(gpu): [[sys.executable, '-u', str(script), phase, '--root', str(root)] + opts
                         for phase, opts in queue] for gpu, queue in commands.items()}
    training.write(root / 'launch_manifest.json', {'queues': planned, 'started_unix': time.time(),
        'pid': os.getpid(), 'protocol_sha256': training.sha(root / 'protocol.json'),
        'source_sha256': training.sha(script), 'launcher_sha256': training.sha(__file__),
        'gpu0_gpu1': 'Untouched teacher servers', 'failure_policy': 'Stop the failed queue, no retry or protocol change.'})
    logs = root / 'logs'; logs.mkdir(exist_ok=False)

    def queue(gpu):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
        history = []
        for index, cmd in enumerate(planned[str(gpu)]):
            label = f'gpu{gpu}_{index}_{cmd[3]}'
            print('START ' + label + ' ' + ' '.join(cmd), flush=True)
            started = time.time()
            with (logs / (label + '.log')).open('w') as log:
                process = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
                training.write(logs / (label + '.json'), {'pid': process.pid, 'command': cmd,
                                                        'started_unix': started})
                code = process.wait()
            history.append({'label': label, 'returncode': code, 'seconds': time.time() - started})
            training.write(logs / f'gpu{gpu}_status.json', {'history': history, 'failed': bool(code)})
            print(f'END {label} returncode={code}', flush=True)
            if code:
                return False
        return True

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(queue, gpu) for gpu in (2, 3)]
        success = all([f.result() for f in futures])
    if success:
        subprocess.run([sys.executable, str(script), 'compare', '--root', str(root)], check=True)
    training.write(root / 'scheduler_status.json', {'status': 'complete' if success else 'failed',
                                                   'finished_unix': time.time()})
    if not success:
        raise SystemExit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('gate', 'run'))
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    {'gate': gate, 'run': run}[args.phase](args.root)
