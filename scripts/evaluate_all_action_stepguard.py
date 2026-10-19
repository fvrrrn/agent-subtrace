"""Reuse one frozen StepGuard cache for fixed validation-selected action heads."""
import argparse
import gc
import json
import os
from pathlib import Path
import select
import shutil
from types import SimpleNamespace
import time

import train_action_models as training


def arguments(root, online=False):
    return SimpleNamespace(root=root, attention='sdpa', lora_rank=16, checkpointing=False,
                           batch_size=1 if online else 4, token_budget=8192 if online else 16384)


def lock_head(root, source, name):
    assert (source / 'complete.json').exists()
    selected = training.read(source / 'selection.json')
    assert training.sha(source / 'selected_head.pt') == selected['head_sha256']
    locked = root / 'locked' / name
    locked.mkdir(exist_ok=False)
    for filename in ('selected_head.pt', 'selection.json', 'run.json', 'complete.json'):
        shutil.copy2(source / filename, locked / filename)
    assert training.sha(locked / 'selected_head.pt') == selected['head_sha256']
    return {'name': name, 'source': str(source), 'selected_epoch': selected['epoch'],
            'head_sha256': selected['head_sha256']}


def predict_head(root, name, lora=False):
    import torch
    args = arguments(root, online=lora)
    cfg = training.configuration(root)
    rows = training.read(root / 'records.json')
    locked = root / 'locked' / name
    selected = training.read(locked / 'selection.json')
    assert training.sha(locked / 'selected_head.pt') == selected['head_sha256']
    model = None
    if lora:
        from peft import PeftModel
        snapshot = training.read(locked / 'snapshot_audit.json')
        assert selected['epoch'] == snapshot['epoch'] == 1
        for filename, expected in snapshot['files_sha256'].items():
            assert training.sha(locked / filename) == expected
        model = PeftModel.from_pretrained(training.backbone(args), locked / 'adapter', is_trainable=False)
        model.eval().requires_grad_(False)
    head = training.make_head(1).eval().requires_grad_(False)
    head.load_state_dict(torch.load(locked / 'selected_head.pt', weights_only=True, map_location='cuda'))
    folder = root / name; folder.mkdir(exist_ok=False)
    started = time.monotonic()
    for split in ('external_test', 'external_dev'):
        training.predict(args, model, head, [r for r in rows if r['split'] == split], folder, split)
    training.write(folder / 'environment.json', {'selected_epoch': selected['epoch'],
        'head_sha256': selected['head_sha256'], 'online_lora': lora, 'no_truncation': True,
        'shared_cache': not lora, 'batch_size': args.batch_size, 'token_budget': args.token_budget,
        'seconds': time.monotonic() - started, 'torch': torch.__version__,
        'note': 'BF16 inference batching can change scores numerically; no equivalence of different batch shapes is claimed.'})
    del model, head
    gc.collect(); torch.cuda.empty_cache()


def initial(root, parent, hard, previous, curves):
    assert not (root / 'protocol.json').exists()
    assert training.read(previous / 'status.json')['status'] == 'complete'
    (root / 'records.json').symlink_to(previous / 'records.json')
    cfg = training.read(previous / 'protocol.json')
    cfg.update(source_sha256=training.sha(training.__file__), records_sha256=training.sha(root / 'records.json'),
               evaluation_script_sha256=training.sha(__file__),
               plan='Existing completed frozen Mamba heads, explicitly epoch1 LoRA, four new curves after four epochs, and native trained Kev when complete. Never select on StepGuard.',
               pending_curves=['linear', 'quadratic', 'sqrt', 'cosine_plain'], curves_root=str(curves))
    training.write(root / 'protocol.json', cfg)
    for name in ('kev_original', 'mamba_exponential'):
        shutil.copytree(previous / name, root / name)
    locked = []
    for name, source in [('mamba_cosine_rank', parent / 'arms/mamba_cosine_rank'),
                         ('mamba_token_hard', hard / 'arms/mamba_token_hard'),
                         ('mamba_endpoint_hard', hard / 'arms/mamba_endpoint_hard')]:
        locked.append(lock_head(root, source, name))
    training.write(root / 'existing_head_manifest.json', locked)
    random_head = root / 'locked/random_mamba_head'
    random_head.mkdir(exist_ok=False)
    shutil.copy2(parent / 'baselines/random_mamba_head.pt', random_head / 'selected_head.pt')
    training.write(random_head / 'selection.json', {'epoch': 0,
        'head_sha256': training.sha(random_head / 'selected_head.pt'), 'criterion': 'Untrained baseline, no selection'})
    # Epoch1 LoRA is already separately copied and hash-audited before this process starts.
    predict_head(root, 'lora_epoch1', lora=True)
    training.cache(arguments(root))
    predict_head(root, 'random_mamba_head')
    for item in locked:
        predict_head(root, item['name'])
    compare(root, 'initial')
    training.write(root / 'initial_complete.json', {'status': 'complete'})


def curves(root):
    cfg = training.configuration(root)
    curve_root = Path(cfg['curves_root'])
    assert training.read(curve_root / 'scheduler_status.json')['status'] == 'complete'
    manifest = []
    for objective in cfg['pending_curves']:
        name = 'mamba_' + objective
        item = lock_head(root, curve_root / 'arms' / name, name)
        assert item['selected_epoch'] <= 4
        manifest.append(item)
    training.write(root / 'curve_head_manifest.json', manifest)
    for item in manifest:
        predict_head(root, item['name'])
    compare(root, 'with_curves')
    training.write(root / 'curves_complete.json', {'status': 'complete'})


def trained_kev(root, parent):
    import torch
    training.configuration(root)
    process = training.read(parent / 'logs/gpu2_4_train.json')['pid']
    try:
        fd = os.pidfd_open(process)
    except ProcessLookupError:
        fd = None
    if fd is not None:
        print('Waiting for native Kev training process ' + str(process), flush=True)
        select.select([fd], [], [])
        os.close(fd)
    source = parent / 'arms/kev_noul_rank'
    assert (source / 'complete.json').exists(), 'Native Kev did not complete successfully'
    selected = training.read(source / 'selection.json')
    locked = root / 'locked/kev_trained'; locked.mkdir(exist_ok=False)
    shutil.copytree(source / 'checkpoint', locked / 'checkpoint')
    training.write(locked / 'selection.json', selected)
    cp, tok, model = training.load(locked / 'checkpoint')
    model.eval().requires_grad_(False)
    args = arguments(root, online=True)
    rows = training.read(root / 'records.json')
    encs, rejected = training.native_admission(args, tok, model, rows)
    folder = root / 'kev_trained'; folder.mkdir(exist_ok=False)
    training.write(folder / 'rejections.json', rejected)
    for split in ('external_test', 'external_dev'):
        training.predict(args, model, None, [r for r in rows if r['split'] == split and r['key'] in encs],
                         folder, split, encs)
    training.write(folder / 'environment.json', {'weights_sha256': cp.weights_sha256(),
        'head_sha256': training.sha(cp.file('head.pt')), 'selected_epoch': selected['epoch'],
        'temperature': model.head.temperature, 'torch': torch.__version__})
    compare(root, 'with_kev')
    training.write(root / 'kev_complete.json', {'status': 'complete'})


def compare(root, stage):
    training.configuration(root)
    snapshots = root / 'comparisons'; snapshots.mkdir(exist_ok=True)
    result = {'stage': stage, 'main_split': 'external_test', 'selection': 'Original validation only; LoRA explicitly epoch1',
              'limitations': 'Previously viewed external benchmark; exploratory comparisons, one seed, no threshold fitting. No unfinished prediction file enters metrics.', 'splits': {}}
    for split in ('external_test', 'external_dev'):
        paths = [p for p in root.glob('*/' + split + '.jsonl') if (p.parent / (split + '.json')).exists()]
        outputs = {p.parent.name: {r['key']: r for r in map(json.loads, p.read_text().splitlines())} for p in paths}
        common = sorted(set.intersection(*(set(v) for v in outputs.values())))
        for key in common:
            assert len({(v[key]['label'], v[key]['state_sha256']) for v in outputs.values()}) == 1
        metrics = {}
        for name, rows in outputs.items():
            m = training.evaluate([rows[k]['label'] for k in common], [rows[k]['score'] for k in common])
            m.pop('pr_curve'); metrics[name] = m
        result['splits'][split] = {'common': len(common), 'metrics': metrics,
            'excluded_by_arm': {name: sorted(set(rows) - set(common)) for name, rows in outputs.items()},
            'matched_source_states_and_labels': True}
    training.write(snapshots / (stage + '.json'), result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=('initial', 'curves', 'trained_kev', 'compare'))
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--parent', type=Path)
    p.add_argument('--hard', type=Path)
    p.add_argument('--previous', type=Path)
    p.add_argument('--curves', type=Path)
    a = p.parse_args()
    if a.phase == 'initial':
        initial(a.root, a.parent, a.hard, a.previous, a.curves)
    elif a.phase == 'curves':
        curves(a.root)
    elif a.phase == 'trained_kev':
        trained_kev(a.root, a.parent)
    else:
        compare(a.root, 'final')
