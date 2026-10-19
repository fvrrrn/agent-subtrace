"""Fixed-checkpoint external StepGuard evaluation; no training or model selection."""
import argparse
from collections import Counter
import gc
import hashlib
import json
from pathlib import Path
import shutil
import time
from types import SimpleNamespace

import train_action_models as training


def prepare(root, parent, data):
    from transformers import AutoTokenizer
    arm = parent / 'arms/mamba_exponential'
    assert (arm / 'complete.json').exists()
    selected = training.read(arm / 'selection.json')
    assert training.sha(arm / 'selected_head.pt') == selected['head_sha256']
    source = [json.loads(s) for s in data.open()]
    assert len({r['key'] for r in source}) == len(source)
    assert all(r['label'] in (0, 1) and r['original_split'] in ('external_test', 'external_dev') for r in source)
    root.mkdir(parents=True, exist_ok=False)
    shutil.copy2(arm / 'selected_head.pt', root / 'selected_head.pt')
    tok = AutoTokenizer.from_pretrained(training.BASE, revision=training.REVISION, local_files_only=True)
    records = []
    for row in source:
        s = json.loads(row['state'])
        assert set(s) == set(training.FIELDS)
        prefix = (training.QUESTION['instructions'] + '\n[TOOL INTERFACES]\n' + training.serial(s['tool_interfaces'])
                  + '\n[USER REQUEST]\n' + training.serial(s['user_request'])
                  + '\n[OBSERVED HISTORY]\n' + training.serial(s['observed_history']) + '\n[AGENT ACTION]\n')
        ids = tok.encode(prefix, add_special_tokens=False)
        start = len(ids)
        ids += tok.encode(training.serial(s['proposed_action']), add_special_tokens=False)
        records.append(dict(row, split=row['original_split'], ids=ids, start=start, tokens=len(ids),
                            text_sha256=hashlib.sha256((prefix + training.serial(s['proposed_action'])).encode()).hexdigest(),
                            ids_sha256=training.fingerprint(ids)))
    training.write(root / 'records.json', records)
    cfg = training.read(parent / 'protocol.json')
    cfg.update(source_sha256=training.sha(training.__file__), records_sha256=training.sha(root / 'records.json'),
               evaluation_script_sha256=training.sha(__file__), stepguard_sha256=training.sha(data),
               selected_checkpoint=selected, parent=str(parent),
               counts=dict(Counter(r['split'] for r in records)),
               class_counts=dict(Counter(f'{r["split"]}:{r["label"]}' for r in records)),
               evaluation_only=True, no_training_or_checkpoint_selection=True,
               main_split='external_test', auxiliary_split='external_dev',
               evaluation_note='Previously viewed StepGuard, exploratory transfer; prior in-domain pilot weights are not reused. No new training, threshold fitting or checkpoint choice.')
    training.write(root / 'protocol.json', cfg)
    print(json.dumps({'counts': cfg['counts'], 'class_counts': cfg['class_counts'],
                      'selected_checkpoint': selected, 'max_tokens': max(r['tokens'] for r in records)}, indent=2), flush=True)


def run(root):
    import torch
    args = SimpleNamespace(root=root, attention='sdpa', lora_rank=16, checkpointing=False,
                           batch_size=4, token_budget=16384)
    cfg = training.configuration(root)
    assert cfg['evaluation_script_sha256'] == training.sha(__file__)
    assert training.sha(root / 'selected_head.pt') == cfg['selected_checkpoint']['head_sha256']
    rows = training.read(root / 'records.json')
    started = time.monotonic()
    torch.manual_seed(cfg['seed'])
    model = training.backbone(args).eval()
    head = training.make_head(1).eval().requires_grad_(False)
    head.load_state_dict(torch.load(root / 'selected_head.pt', weights_only=True, map_location='cuda'))
    before = training.parameter_hash(model)
    assert before == training.read(Path(cfg['parent']) / 'features/complete.json')['base_sha256']
    folder = root / 'mamba_exponential'; folder.mkdir(exist_ok=False)
    admitted = [r for r in rows if r['start'] < r['tokens'] <= cfg['eval_limit']]
    admitted_keys = {r['key'] for r in admitted}
    training.write(folder / 'rejections.json', [dict(key=r['key'], label=r['label'], split=r['split'],
        tokens=r['tokens'], reason='empty_action' if r['tokens'] == r['start'] else 'context_overflow')
        for r in rows if r['key'] not in admitted_keys])
    for split in ('external_test', 'external_dev'):
        training.predict(args, model, head, [r for r in admitted if r['split'] == split], folder, split)
    assert training.parameter_hash(model) == before
    training.write(folder / 'environment.json', {'base_sha256': before, 'base_revision': training.REVISION,
        'head_sha256': training.sha(root / 'selected_head.pt'), 'seconds': time.monotonic() - started,
        'device': torch.cuda.get_device_name(), 'torch': torch.__version__, 'no_truncation': True})
    del model, head
    gc.collect(); torch.cuda.empty_cache()

    started = time.monotonic()
    cp, tok, model = training.native_setup(args, False)
    model.eval()
    folder = root / 'kev_original'; folder.mkdir(exist_ok=False)
    encs, rejected = training.native_admission(args, tok, model, rows)
    training.write(folder / 'rejections.json', rejected)
    for split in ('external_test', 'external_dev'):
        training.predict(args, model, None, [r for r in rows if r['split'] == split and r['key'] in encs],
                         folder, split, encs)
    training.write(folder / 'environment.json', {'parent_weights_sha256': cp.weights_sha256(),
        'head_sha256': training.sha(cp.file('head.pt')), 'temperature': model.head.temperature,
        'seconds': time.monotonic() - started, 'device': torch.cuda.get_device_name(),
        'torch': torch.__version__, 'no_truncation': True})
    compare(root)
    training.write(root / 'status.json', {'status': 'complete', 'training': False})


def compare(root):
    cfg = training.configuration(root)
    original = training.read(root / 'records.json')
    result = {'selected_epoch': cfg['selected_checkpoint']['epoch'], 'main_split': 'external_test',
              'selection': 'Fixed before external scoring, minimum original validation endpoint BCE',
              'limits': cfg['evaluation_note'], 'splits': {}}
    for split in ('external_test', 'external_dev'):
        outputs = {arm: {r['key']: r for r in map(json.loads, (root / arm / (split + '.jsonl')).read_text().splitlines())}
                   for arm in ('kev_original', 'mamba_exponential')}
        planned = {r['key'] for r in original if r['split'] == split}
        common = sorted(set.intersection(*(set(v) for v in outputs.values())))
        for key in common:
            assert len({(v[key]['label'], v[key]['state_sha256']) for v in outputs.values()}) == 1
        summary = {'planned': len(planned), 'common': len(common), 'metrics': {},
                   'excluded_keys': {a: sorted(planned - set(v)) for a, v in outputs.items()},
                   'labels_and_state_hashes_match': True}
        for arm, out in outputs.items():
            m = training.evaluate([out[k]['label'] for k in common], [out[k]['score'] for k in common])
            m.pop('pr_curve')
            summary['metrics'][arm] = m
        result['splits'][split] = summary
    training.write(root / 'comparison.json', result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=('prepare', 'run'))
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--parent', type=Path)
    p.add_argument('--data', type=Path)
    a = p.parse_args()
    if a.phase == 'prepare':
        if a.parent is None or a.data is None:
            p.error('prepare requires --parent and --data')
        prepare(a.root, a.parent, a.data)
    else:
        run(a.root)
