"""Optional two-rank native Qwen full FT; never launched by another trainer."""
import argparse
from datetime import timedelta
from functools import partial
from importlib.metadata import PackageNotFoundError, version
import json
import math
import os
from pathlib import Path
import random
import shutil
import time
import traceback

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP, FullStateDictConfig, MixedPrecision,
    ShardingStrategy, StateDictType,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

import train_whole_dialogs as original
import train_autoresearch_lora as lora
from train_curve_event_heads import Head
from train_autoresearch_heads import seed
from prepare_whole_dialogs import BASE, REVISION, digest, write
from graph_safety.metrics import evaluate


class BudgetStop(Exception):
    pass


def guard(args):
    # Every caller executes this at the same point, including during evaluation.
    stop = torch.tensor([int(time.time() >= args.deadline_unix - 180)
                         if dist.get_rank() == 0 else 0], device='cuda')
    dist.broadcast(stop, src=0)
    if stop.item():
        raise BudgetStop('deadline persistence reserve reached (180 seconds)')


def primary():
    return dist.get_rank() == 0


def bare(head):
    return head.module if isinstance(head, DDP) else head


def rng_state():
    return {'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state(),
            'python': random.getstate()}


def restore_rng(state):
    torch.set_rng_state(state['torch'])
    torch.cuda.set_rng_state(state['cuda'])
    random.setstate(state['python'])


def forward(model, head, row):
    # Autocast covers the native backbone too, not just the Mamba head.
    with torch.autocast('cuda', dtype=torch.bfloat16):
        hidden = original.encode(model, row)
        z, null = head(hidden)
    assert null is None
    return z[0, :row['tokens']].float(), None


def checkpoint(model, head, folder, optimizer=None, progress=None):
    """Collective full FP32 backbone, replicated head, local FP32 Adam shards."""
    if primary():
        folder.mkdir(parents=True, exist_ok=False)
        write(folder / 'status.json', {'status': 'saving', **(progress or {})})
    dist.barrier()
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT,
                             FullStateDictConfig(offload_to_cpu=True, rank0_only=True)):
        state = model.state_dict()  # All ranks must participate.
    if primary():
        assert all(not v.is_floating_point() or v.dtype == torch.float32
                   for v in state.values())
        torch.save(state, folder / 'backbone.pt')
        torch.save({k: v.detach().cpu() for k, v in bare(head).state_dict().items()},
                   folder / 'head.pt')
    del state
    # AdamW is constructed on local FSDP shards, NOT full parameters or BF16 copies.
    local = {'rng': rng_state(), 'rank': dist.get_rank(), 'world_size': 2,
             'progress': progress, 'same_world_and_wrapping_required': True,
             'optimizer': optimizer.state_dict() if optimizer is not None else None}
    torch.save(local, folder / f'training_rank{dist.get_rank()}.pt')
    del local
    dist.barrier()
    if primary():
        files = sorted(folder.glob('*.pt'))
        write(folder / 'hashes.json', {p.name: original.file_hash(p) for p in files})
        write(folder / 'status.json', {'status': 'saved', **(progress or {})})
    dist.barrier()


def reload_gate(model, head, folder):
    # Each rank reads the CPU full state; rank0_only=False is required for loading.
    # Do not pass an empty nonzero-rank dict to FSDP.load_state_dict.
    state = torch.load(folder / 'backbone.pt', map_location='cpu', weights_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT,
                             FullStateDictConfig(offload_to_cpu=True, rank0_only=False)):
        model.load_state_dict(state, strict=True)
    del state
    bare(head).load_state_dict(torch.load(folder / 'head.pt', map_location='cpu',
                                         weights_only=True), strict=True)
    dist.barrier()


def finite_grads(parameters):
    params = list(parameters)
    ok = all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in params)
    flag = torch.tensor(int(ok), device='cuda')
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    assert flag.item(), 'nonfinite gradient on at least one rank'


def backbone_grad_norm(model):
    # FSDP accounts for every shard and nested FSDP unit. Infinity disables clipping.
    norm = model.clip_grad_norm_(float('inf'), norm_type=2.)
    assert torch.isfinite(norm) and norm > 0
    return norm


def clip_combined(model, head):
    finite_grads(list(model.parameters()) + list(head.parameters()))
    bn = backbone_grad_norm(model)
    # Head gradients are already averaged by DDP: count the replica only ONCE.
    hn2 = sum(p.grad.detach().float().square().sum() for p in head.parameters()
              if p.grad is not None)
    norm = (bn.square() + hn2).sqrt()
    assert torch.isfinite(norm)
    scale = (1. / (norm + 1e-6)).clamp(max=1.)
    for p in list(model.parameters()) + list(head.parameters()):
        if p.grad is not None:
            p.grad.mul_(scale)
    return norm.item()


def gates(model, head, training, args):
    saved_rng = rng_state()
    guard(args)
    probe = next(r for r in training if 1 < r['tokens'] < 4096
                 and any(s['label'] for s in r['spans']))
    model.eval(); head.eval()
    cut = min(probe['spans'][0]['end'], probe['tokens'] // 2)
    vocab = model.module.get_input_embeddings().num_embeddings
    changed = dict(probe)
    changed['ids'] = probe['ids'][:cut] + [(i + 1) % vocab for i in probe['ids'][cut:]]
    with torch.no_grad():
        z, _ = forward(model, head, probe)
        altered, _ = forward(model, head, changed)
        causal_error = (z[:cut] - altered[:cut]).abs().max().item()
        assert causal_error == 0., 'causal-prefix perturbation gate failed'
        del altered
    guard(args)
    folder = args.root / 'gate_reload'
    checkpoint(model, head, folder, progress={'optimizer_steps': 0})
    with torch.no_grad():
        # Ensure reload is not just a state_dict no-op.
        for p in model.parameters():
            p.add_(.01)
        head.readout.weight.add_(.01)
    reload_gate(model, head, folder)
    with torch.no_grad():
        restored, _ = forward(model, head, probe)
        reload_error = (z - restored).abs().max().item()
        assert reload_error == 0., 'backbone/head disk reload gate failed'
        del z, restored
    guard(args)
    model.train(); head.train()
    z, n = forward(model, head, probe)
    z.retain_grad()
    c, e = lora.losses(z, n, probe, args)
    # The head is not DDP-wrapped until gates finish: autograd.grad is local here.
    cg = torch.autograd.grad(c, head.readout.weight, retain_graph=True)[0].norm().item()
    eg = torch.autograd.grad(e, head.readout.weight, retain_graph=True)[0].norm().item()
    assert all(math.isfinite(v) and v > 0 for v in (cg, eg))
    (c + .1 * e).backward()
    mask = torch.zeros_like(z, dtype=torch.bool)
    for span in probe['spans']:
        mask[span['start']:span['end']] = True
    assert torch.count_nonzero(z.grad[~mask]) == 0
    finite_grads(list(model.parameters()) + list(head.parameters()))
    # Every nonempty FSDP flat shard must receive a gradient, then inspect the
    # globally sharded backbone norm and BOTH independent Mamba blocks.
    missing = sum(p.numel() > 0 and p.grad is None for p in model.parameters())
    missing = torch.tensor(missing, device='cuda')
    dist.all_reduce(missing)
    assert missing.item() == 0
    bn = backbone_grad_norm(model).item()
    hg = {name: p.grad.float().norm().item() for name, p in head.named_parameters()
          if p.grad is not None}
    for i in (0, 1):
        assert any(v > 0 for name, v in hg.items() if name.startswith(f'blocks.{i}.'))
    assert hg['projection.weight'] > 0 and hg['readout.weight'] > 0
    model.zero_grad(set_to_none=True); head.zero_grad(set_to_none=True)
    del z, n, c, e
    guard(args)
    longest = max(training, key=lambda r: r['tokens'])
    assert longest['tokens'] <= 16384
    z, n = forward(model, head, longest)
    c, e = lora.losses(z, n, longest, args)
    (c + .1 * e).backward()
    finite_grads(list(model.parameters()) + list(head.parameters()))
    longest_norm = backbone_grad_norm(model).item()
    model.zero_grad(set_to_none=True); head.zero_grad(set_to_none=True)
    del z, n, c, e
    restore_rng(saved_rng)
    if primary():
        write(args.root / 'gate.json', {'status': 'passed', 'causal_error': causal_error,
            'reload_error': reload_error, 'token_readout_gradient': cg,
            'event_readout_gradient': eg, 'backbone_gradient_norm': bn,
            'head_gradient_norms': hg, 'loss_mask_passed': True,
            'longest_train_key': longest['key'], 'longest_train_tokens': longest['tokens'],
            'longest_backbone_gradient_norm': longest_norm, 'optimizer_steps': 0,
            'peak_cuda_bytes_rank0': torch.cuda.max_memory_allocated(),
            'rng_restored': True, 'init_weights_reloaded': True})
    dist.barrier()


def validate(model, head, rows, epoch, args):
    guard(args)
    model.eval(); head.eval()
    folder = args.root / f'dev_epoch{epoch}'
    labels, scores, ces, curves = [], [], [], []
    if primary():
        (folder / 'tokens').mkdir(parents=True, exist_ok=False)
        write(folder / 'status.json', {'status': 'evaluating', 'expected_dialogs': len(rows)})
    dist.barrier()
    stream = (folder / 'predictions.jsonl').open('w') if primary() else None
    try:
        with torch.no_grad():
            # Identical rows on both ranks deliberately keep FSDP pre-forward
            # collectives aligned. Only rank0 records/counts the predictions.
            for row in rows:
                guard(args)
                z, n = forward(model, head, row)
                if primary():
                    c, _ = lora.losses(z, n, row, args)
                    curves.append(c.item())
                    torch.save({'key': row['key'], 'ids_sha256': row['ids_sha256'],
                        'raw_logits': z.cpu(), 'null_logits': None, 'spans': row['spans']},
                        folder / 'tokens' / (row['key'] + '.pt'))
                    for span in row['spans']:
                        e = lora.event(z, n, span, 'endpoint')
                        score = e.sigmoid().item()
                        labels.append(span['label']); scores.append(score)
                        ces.append(torch.nn.functional.binary_cross_entropy_with_logits(
                            e, e.new_tensor(float(span['label']))).item())
                        stream.write(json.dumps({'key': span['key'], 'dialog': row['key'],
                            'label': span['label'], 'score': score, 'event_logit': e.item(),
                            'state_sha256': span['state_sha256'], 'ids_sha256': row['ids_sha256'],
                            'sources': row['sources']}) + '\n')
                del z, n
    finally:
        if stream is not None:
            stream.close()
    guard(args)
    metrics = None
    if primary():
        metrics = evaluate(labels, scores)
        pr = metrics.pop('pr_curve')
        metrics.update(event_bce=sum(ces) / len(ces), token_bce=sum(curves) / len(curves),
                       evaluated_dialogs=len(rows), evaluated_events=len(labels))
        for fpr in (.01, .05):
            metrics[f'recall_at_fpr_{fpr}'] = max((p['recall'] for p in pr
                if p['recall'] is not None and p.get('specificity') is not None
                and 1 - p['specificity'] <= fpr + 1e-12), default=0.)
        write(folder / 'metrics.json', metrics)
        write(folder / 'status.json', {'status': 'complete', 'dialogs': len(rows), 'events': len(labels)})
    dist.barrier()
    return metrics


def load_rows(args):
    rows, audit = original.load_data(args.data)
    splits = json.loads(args.splits.read_text())
    assert set(splits) == {'train_small', 'train_full', 'dev_select', 'dev_confirm', 'original_validation'}
    bykey = {r['key']: r for r in rows}
    training = [bykey[k] for k in splits['train_small']]
    validation = [bykey[k] for k in splits['dev_select']]
    assert len(training) == 2009 and len(validation) == 301
    assert len({r['key'] for r in training + validation}) == 2310
    assert not ({r['group'] for r in training} & {r['group'] for r in validation})
    for row in training + validation:
        assert digest(row['ids']) == row['ids_sha256']
        assert len(row['ids']) == row['tokens'] <= 16384 and row['spans']
        end = 0
        for span in row['spans']:
            assert end <= span['start'] < span['end'] <= row['tokens']
            assert span['label'] in (0, 1)
            end = span['end']
    return training, validation, audit


def optimizer_fp32(optimizer):
    for group in optimizer.param_groups:
        assert all(p.dtype == torch.float32 for p in group['params'])
    for state in optimizer.state.values():
        for name in ('exp_avg', 'exp_avg_sq'):
            if name in state:
                assert state[name].dtype == torch.float32


def run(args):
    model = head = optimizer = None
    owned_root = False
    epoch = steps = processed = applied = 0
    history = []; best = None
    started = time.monotonic()
    status = {'status': 'initializing'}
    try:
        if primary():
            args.root.mkdir(parents=True, exist_ok=False)
        dist.barrier()
        owned_root = True
        guard(args)
        assert args.epochs > 0 and original.SEED == 20261017
        training, validation, audit = load_rows(args)
        guard(args)
        seed(original.SEED)
        head = Head('curve_linear_softmax', False).float().cuda()
        native = original.backbone(False)
        native.requires_grad_(True)
        native.float()  # FP32 masters BEFORE FSDP, never BF16 optimizer parameters.
        native.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        assert all(p.requires_grad and p.dtype == torch.float32 for p in native.parameters())
        classes = {type(layer) for layer in native.layers}
        model = FSDP(native, auto_wrap_policy=partial(transformer_auto_wrap_policy,
            transformer_layer_cls=classes), sharding_strategy=ShardingStrategy.FULL_SHARD,
            mixed_precision=MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.float32,
                                           buffer_dtype=torch.float32, keep_low_precision_grads=False),
            device_id=torch.cuda.current_device(), sync_module_states=True,
            use_orig_params=False, limit_all_gathers=True)
        del native
        if primary():
            dependencies = (original, lora, __import__('train_curve_event_heads'),
                            __import__('train_autoresearch_heads'), __import__('prepare_whole_dialogs'),
                            __import__('graph_safety.metrics', fromlist=['evaluate']))
            source = args.root / 'source'; source.mkdir()
            for path in [Path(__file__)] + [Path(m.__file__) for m in dependencies]:
                shutil.copy2(path, source / path.name)
            versions = {}
            for package in ('transformers', 'mamba-ssm', 'causal-conv1d', 'triton', 'numpy'):
                try:
                    versions[package] = version(package)
                except PackageNotFoundError:
                    versions[package] = None
            protocol = {**{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                'base': BASE, 'revision': REVISION, 'data_sha256': audit['dialogs_sha256'],
                'template_sha256': audit['template_sha256'], 'splits_sha256': original.file_hash(args.splits),
                'source_sha256': {p.name: original.file_hash(p) for p in source.iterdir()},
                'torch': str(torch.__version__), 'cuda': torch.version.cuda, 'versions': versions,
                'gpu': torch.cuda.get_device_name(), 'world_size': 2, 'seed': original.SEED,
                'gpu_total_memory': torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory,
                'cudnn': torch.backends.cudnn.version(),
                'cudnn_deterministic': torch.backends.cudnn.deterministic,
                'cuda_matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
                'layer_types': sorted(c.__name__ for c in classes), 'master_and_adam_dtype': 'float32',
                'compute_dtype': 'bfloat16', 'reduce_dtype': 'float32', 'model_lr': 2e-6, 'head_lr': 3e-4,
                'global_batch_dialogs': 4, 'microsteps_per_rank': 2, 'gradient_clip_global_norm': 1.,
                'synchronization_only_forwards_per_epoch': (-len(training)) % 4,
                'padding': 'right to multiple of128; factual unpadded lengths in inputs.json',
                'gradient_checkpointing': 'native non-reentrant', 'readout': 'endpoint', 'curve': 'broad',
                'event_weight': .1, 'no_truncation': True, 'eval': 'identical collective forward on both ranks; rank0 counts once',
                'final_partial': 'world_size/window_size scaling; zero-weight dummy collective forwards',
                'deadline_reserve_seconds': 180, 'train_dialogs': 2009, 'dev_select_dialogs': 301,
                'original_weight_precision': 'native BF16 pretrained values promoted to FP32 masters',
                'training_objective': 'global mean dialog(mean action token BCE + 0.1 mean action event BCE)'}
            write(args.root / 'protocol.json', protocol)
            write(args.root / 'inputs.json', {'train_small': training, 'dev_select': validation})
        dist.barrier()
        guard(args)
        gates(model, head, training, args)
        guard(args)
        head = DDP(head, device_ids=[torch.cuda.current_device()], broadcast_buffers=False)
        optimizer = torch.optim.AdamW([{'params': model.parameters(), 'lr': 2e-6},
            {'params': head.parameters(), 'lr': 3e-4}], weight_decay=.01, foreach=False)
        optimizer_fp32(optimizer)
        for epoch in range(1, args.epochs + 1):
            guard(args)
            processed = applied = 0
            model.train(); head.train()
            order = list(training)
            random.Random(original.SEED + epoch).shuffle(order)
            totals = torch.zeros(2, device='cuda', dtype=torch.float64)
            optimizer.zero_grad(set_to_none=True)
            for start in range(0, len(order), 4):
                window = order[start:start + 4]
                # Two forwards/backwards per rank even for a short final window.
                # DDP and FSDP each average by world_size, hence 2/real_window.
                for micro in range(2):
                    guard(args)
                    index = micro * 2 + dist.get_rank()
                    real = index < len(window)
                    row = window[index] if real else order[0]
                    z, n = forward(model, head, row)
                    if real:
                        c, e = lora.losses(z, n, row, args)
                        loss = (c + .1 * e) * (2. / len(window))
                        totals += torch.stack((c.detach(), e.detach())).double()
                    else:
                        # Synchronization-only input: NO duplicated supervision.
                        loss = z.sum() * 0.
                    loss.backward()
                    processed = start + min((micro + 1) * 2, len(window))
                    del z, n, loss
                    if real:
                        del c, e
                guard(args)
                clip_combined(model, head)
                optimizer.step(); optimizer.zero_grad(set_to_none=True)
                optimizer_fp32(optimizer)
                steps += 1; applied = start + len(window)
            dist.all_reduce(totals)
            guard(args)
            folder = args.root / f'epoch{epoch}'
            checkpoint(model, head, folder, optimizer, {'epoch': epoch,
                'optimizer_steps': steps, 'training_epoch_complete': True,
                'dialogs_applied_this_epoch': applied, 'resumable_same_world_and_wrapping': True})
            metrics = validate(model, head, validation, epoch, args)
            if primary():
                entry = {'epoch': epoch, 'train_token_bce': totals[0].item() / len(order),
                    'train_event_bce': totals[1].item() / len(order), 'validation': metrics,
                    'seconds': time.monotonic() - started}
                history.append(entry)
                write(args.root / 'history.json', history)
                key = (metrics['average_precision'], -metrics['event_bce'])
                if best is None or key > best:
                    best = key
                    write(args.root / 'selection.json', {'epoch': epoch,
                        'criterion': 'dev_select AP; ties lower event BCE; exact ties earlier epoch',
                        'checkpoint_hashes': json.loads((folder / 'hashes.json').read_text()),
                        'metrics': metrics})
                write(folder / 'status.json', {'status': 'complete', 'epoch': epoch})
            dist.barrier()
        status = {'status': 'complete', 'epochs': args.epochs, 'optimizer_steps': steps}
    except BudgetStop as error:
        status = {'status': 'budget_stopped', 'reason': str(error), 'partial_epoch': epoch,
            'dialogs_processed_this_epoch': processed, 'dialogs_applied_this_epoch': applied,
            'pending_dialogs_discarded': processed - applied, 'optimizer_steps': steps,
            'diagnostic_resumable': False, 'reason_nonresumable': 'partial-run continuation is not implemented; pending gradients discarded'}
        if model is not None:
            model.zero_grad(set_to_none=True); head.zero_grad(set_to_none=True)
            checkpoint(model, head, args.root / 'deadline_partial', optimizer, status)
        if primary():
            write(args.root / 'budget_stop.json', status)
    except BaseException as error:
        status = {'status': 'failed', 'error': repr(error), 'rank': dist.get_rank(),
                  'traceback': traceback.format_exc(), 'optimizer_steps': steps}
        if owned_root:
            write(args.root / f'failure_rank{dist.get_rank()}.json', status)
        raise
    finally:
        status['seconds'] = time.monotonic() - started
        status['peak_cuda_bytes'] = torch.cuda.max_memory_allocated()
        if owned_root:
            write(args.root / f'status_rank{dist.get_rank()}.json', status)
            if primary():
                write(args.root / 'status.json', status)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--splits', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--deadline-unix', type=float, required=True)
    args = parser.parse_args()
    args.readout = 'endpoint'; args.curve = 'broad'; args.event_weight = .1
    try:
        assert int(os.environ['WORLD_SIZE']) == 2, 'launch with torchrun --nproc-per-node=2'
        torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
        dist.init_process_group('nccl', timeout=timedelta(minutes=20))
        run(args)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
