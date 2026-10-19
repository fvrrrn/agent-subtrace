"""Read-only post-run artifact consistency audit; no inference or fitting."""
import hashlib
import json
from pathlib import Path
import torch
from tqdm import tqdm
from prepare_whole_dialogs import digest, write
from train_streaming_weighted_multilayer_v1 import RUN, OLD


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    read = lambda p: json.loads(p.read_text())
    source = RUN.parent / (RUN.name + '-launch/source')
    manifest_path = source.parent / 'source_manifest.json'
    manifest = read(manifest_path)
    assert all(sha(source / name) == expected for name, expected in manifest.items())
    rows = {r['key']: r for r in read(OLD / 'data/dialogs.json')}
    mapping = read(OLD / 'data/splits.json')
    expected_keys = set(mapping['dev_select'])
    assert len(expected_keys) == 301
    summary, hashes = {}, {}
    total = 0
    initialization = None
    for feature in ('last', 'three'):
        for curve in ('step', 'linear', 'broad'):
            arm = feature + '_' + curve
            folder = RUN / arm
            assert read(folder / 'fit/complete.json')['epochs'] == 2
            gate = read(folder / 'gate.json')
            assert gate['status'] == 'passed' and gate['causal_error'] == gate['reload_error'] == 0
            state = torch.load(folder / 'gate_reload.pt', weights_only=True, map_location='cpu')
            shared = {k: hashlib.sha256(v.numpy().tobytes()).hexdigest() for k, v in state.items()
                      if not k.startswith(('input_norm.', 'projection.'))}
            if initialization is None:
                initialization = shared
            assert shared == initialization
            selection = read(folder / 'fit/selection.json')
            assert sha(folder / 'fit/selected_head.pt') == selection['head_sha256']
            for epoch in (1, 2):
                checkpoint = folder / f'fit/head_epoch{epoch}.pt'
                assert sha(checkpoint) == read(folder / f'fit/checkpoint_epoch{epoch}.json')['sha256']
                evaluation = folder / f'fit/dev_select_epoch{epoch}'
                events = [json.loads(line) for line in (evaluation / 'events.jsonl').read_text().splitlines()]
                assert len(events) == 886 and sum(e['span']['label'] for e in events) == 147
                by_dialog = {}
                for event in events:
                    by_dialog.setdefault(event['dialog'], []).append(event)
                assert set(by_dialog) == expected_keys
                assert len(list((evaluation / 'tokens').glob('*.pt'))) == 301
                for key in tqdm(mapping['dev_select'], desc=f'{arm} epoch{epoch} raw endpoint audit', unit='dialog'):
                    path = evaluation / 'tokens' / (key + '.pt')
                    saved = torch.load(path, weights_only=True, map_location='cpu')
                    row = rows[key]
                    assert saved['ids'] == row['ids']
                    assert saved['ids_sha256'] == row['ids_sha256'] == digest(saved['ids'])
                    assert saved['spans'] == row['spans'] and saved['tokens'] == row['tokens']
                    z = saved['raw_token_logits']
                    assert z.shape == (row['tokens'],) and torch.isfinite(z).all()
                    assert [e['span'] for e in by_dialog[key]] == row['spans']
                    for event in by_dialog[key]:
                        assert z[event['span']['end']-1].item() == event['event_logit']
                        total += 1
                    hashes[str(path.relative_to(RUN))] = sha(path)
                metrics = read(evaluation / 'metrics.json')
                assert metrics['n_examples'] == 886 and metrics['n_positive'] == 147
                for name in ('metrics.json', 'events.jsonl', 'fixed_threshold_slices.json'):
                    hashes[str((evaluation / name).relative_to(RUN))] = sha(evaluation / name)
            metrics = read(folder / f'fit/dev_select_epoch{selection["epoch"]}/metrics.json')
            summary[arm] = {'epoch': selection['epoch'], 'metrics': metrics,
                            'seconds': read(folder / 'fit/complete.json')['seconds']}
    write(RUN / 'artifact_audit.json', {'status': 'passed', 'raw_event_endpoints_checked': total,
        'raw_endpoint_error': 0, 'dev_dialogs_each': 301, 'events_each': 886, 'unsafe_each': 147,
        'source_manifest_sha256': sha(manifest_path), 'source_unchanged': True,
        'shared_mamba_readout_initialization': initialization, 'hashes': hashes,
        'audit_source_sha256': sha(Path(__file__))})
    write(RUN / 'summary.json', summary)
    print(json.dumps({k: {'epoch': v['epoch'], 'ap': v['metrics']['average_precision']}
                      for k, v in summary.items()}, indent=2))


if __name__ == '__main__':
    main()
