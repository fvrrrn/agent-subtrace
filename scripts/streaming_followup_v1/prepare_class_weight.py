"""Prospective train-only class mass under mean-action/mean-dialog loss."""
import json
import math
from pathlib import Path
import hashlib
from tqdm import tqdm

ROOT = Path('/var/local/danefedov/experiments/agent-safety')
DATA = ROOT/'runs/frozen-instruct-4b-v1/data'


def main():
    data = json.loads((DATA/'dialogs.json').read_text())
    rows = data['dialogs'] if isinstance(data, dict) else data
    index = {r['key']: r for r in rows}
    keys = json.loads((DATA/'splits.json').read_text())['train_small']
    assert len(keys) == len(set(keys)) == 2009
    masses = []; counts = [0, 0]
    for key in tqdm(keys, desc='Train-only effective class counts', unit='dialog'):
        spans = index[key]['spans']
        assert spans and all(s['label'] in (0, 1) for s in spans)
        positives = sum(s['label'] for s in spans)
        counts[0] += len(spans)-positives; counts[1] += positives
        masses.append(positives/len(spans))
    unsafe = math.fsum(masses); safe = len(keys)-unsafe
    result = {'train_dialogs': len(keys), 'raw_safe_actions': counts[0], 'raw_unsafe_actions': counts[1],
        'effective_safe': safe, 'effective_unsafe': unsafe, 'positive_weight': safe/unsafe,
        'formula': 'sum_d(safe_d/actions_d) / sum_d(unsafe_d/actions_d)',
        'train_ids': keys, 'input_hashes': {name: hashlib.sha256((DATA/name).read_bytes()).hexdigest()
            for name in ('dialogs.json', 'splits.json')}, 'uses_dev_labels': False}
    output = ROOT/'runs/streaming-followup-v1/train_class_weight.json'
    with output.open('x') as stream:
        json.dump(result, stream, indent=2); stream.write('\n')
    print({k:v for k,v in result.items() if k not in ('train_ids', 'input_hashes')}, flush=True)


if __name__ == '__main__':
    main()
