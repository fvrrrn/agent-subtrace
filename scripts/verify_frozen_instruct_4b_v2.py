"""CPU-only complete read-only cache admission; preserve the v1 byte hashes."""
import hashlib
import argparse
import io
import json
from pathlib import Path
import time
from types import SimpleNamespace

import torch
from tqdm import tqdm

ROOT = Path('/var/local/danefedov/experiments/agent-safety')
OLD = ROOT / 'runs/frozen-instruct-4b-v1'
RUN = ROOT / 'runs/frozen-instruct-4b-v2'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    run = parser.parse_args().output
    torch.set_num_threads(4)
    read = lambda p: json.loads(p.read_text())
    rows = read(OLD / 'data/dialogs.json')
    complete = read(OLD / 'cache/features/complete0.json')
    assert complete['count'] == len(rows) == 2310 and complete['shard'] == 0
    identity = {k: v for k, v in complete.items() if k not in ('count', 'shard')}
    capability = identity['environment']['capability']
    assert isinstance(capability, list) and len(capability) == 2
    assert all(type(v) is int for v in capability)
    identity['environment']['capability'] = tuple(capability)
    audit = read(OLD / 'data/data_audit.json')
    assert identity['base'] == 'Qwen/Qwen3.5-4B'
    assert identity['revision'] == '851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'
    assert all(identity[k] == v for k, v in audit.items())
    assert sha(OLD / 'data/dialogs.json') == identity['dialogs_sha256']
    assert sha(OLD / 'data/splits.json') == identity['splits_sha256']
    assert identity['weight_files_sha256'] == read(OLD / 'model_weights.json')['sha256']
    assert identity['weights_sha256'] == read(OLD / 'cache/backbone_gate.json')['weights_sha256']
    import train_frozen_instruct_4b_v1 as trainer
    configurations = {}
    for arm in tqdm(('A_broad', 'A_linear', 'A_step', 'B', 'C'), desc='CPU config serialization boundaries'):
        args = SimpleNamespace(data=OLD / 'data', cache=OLD / 'cache', splits=OLD / 'data/splits.json',
            arm=arm, seed=20261017, epochs=2, event_weight=.1, train_split='train_small', attention=False,
            curve=arm.split('_')[1] if arm.startswith('A_') else 'step',
            readout='endpoint' if arm.startswith('A_') else arm, deadline_unix=1792345185,
            control_gate=ROOT / 'runs/sound-anomaly-4b-control-v1/candidates/control_a/gate.json')
        trainer.load_data(args)
        assert args.cache_identity == identity
        config = trainer.config(args)
        assert config == json.loads(json.dumps(config)), 'Gate/train JSON config equality failed'
        buffer = io.BytesIO()
        torch.save(config, buffer)
        buffer.seek(0)
        assert torch.load(buffer, weights_only=True) == config
        # Same predicate as base.train, with the actual JSON serialization boundary.
        gate_result = json.loads(json.dumps({'status': 'passed', 'config': config}))
        assert gate_result['status'] == 'passed' and gate_result['config'] == trainer.config(args)
        assert args.cache_identity == identity, 'Config serialization mutated runtime identity'
        configurations[arm] = config
    (run / 'serialization_admission.json').write_text(json.dumps({'status': 'passed',
        'configs': configurations, 'boundaries': ['gate JSON -> train config', 'Torch config',
        'runtime identity unchanged']}, indent=2) + '\n')
    hashes = {}
    for row in tqdm(rows, desc='CPU verify cache identities/IDs/shapes + SHA256', unit='dialog'):
        assert time.time() < 1792345185
        path = OLD / 'cache/features' / (row['key'] + '.pt')
        with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
            saved = torch.load(path, map_location='cpu', weights_only=True)
        from prepare_whole_dialogs import digest
        assert saved['ids_sha256'] == row['ids_sha256'] == digest(row['ids'])
        assert saved['identity'] == identity, row['key']
        assert saved['hidden'].shape == (row['tokens'], 2560)
        assert len(row['ids']) == row['tokens'] <= 16384
        assert saved['hidden'].dtype == torch.bfloat16
        assert torch.isfinite(saved['hidden']).all()
        hashes[str(path.relative_to(OLD))] = sha(path)
        del saved
    for path in tqdm(sorted(p for p in OLD.rglob('*') if p.is_file() and p.suffix != '.pt'), desc='Hash v1 metadata'):
        hashes[str(path.relative_to(OLD))] = sha(path)
    (run / 'v1_hashes.json').write_text(json.dumps(hashes, indent=2) + '\n')
    assert hashes == read(ROOT / 'runs/frozen-instruct-4b-v2/v1_hashes.json'), 'v1 changed since v2 admission'
    (run / 'cache_admission.json').write_text(json.dumps({'status': 'passed', 'count': len(rows),
        'cache': str(OLD / 'cache'), 'normalization': 'environment.capability JSON list -> tuple only',
        'v1_hashes_sha256': sha(run / 'v1_hashes.json'), 'read_only': True}, indent=2) + '\n')
    print('CPU cache admission passed: 2310/2310', flush=True)


if __name__ == '__main__':
    main()
