"""Post-hoc F1 threshold diagnostic on saved TS-Bench-Harm predictions only."""
import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path

from graph_safety.metrics import evaluate, operating_point


def run(root):
    comparison = json.loads((root / 'lora_validation_epoch2_comparison.json').read_text())
    original = json.loads((root / 'protocol.json').read_text())
    assert original['expected'] == comparison['common'] == 731
    # Reconstruct raw benchmark labels from the original immutable local source snapshot.
    known = {json.loads(line)['key']: json.loads(line)['label']
             for line in (root / 'predictions/kev_original/external_test.jsonl').read_text().splitlines()}
    # Original scores are in per-model strict outputs only as binary labels; use
    # the raw files whose hashes are fixed in the protocol, never infer 0.5 from scores.
    source = root / 'raw_labels.json'
    labels = json.loads(source.read_text())
    assert set(labels) == set(known)
    assert all(int(v > 0) == known[k] for k, v in labels.items())
    models = ('lora_validation_epoch2', 'mamba_quadratic', 'kev_original')
    result = {'benchmark': 'TS-Bench-Harm', 'objective': 'Maximize F1 on benchmark labels',
        'decision_rule': 'unsafe iff score >= threshold',
        'tie_break': 'Highest accuracy, then highest threshold among observed-score cutpoints and0.5',
        'limitation': 'Post-hoc optimistic benchmark-label diagnostic, NOT held-out performance or calibration; no model calls or changes to original predictions/thresholds.',
        'raw_labels_sha256': hashlib.sha256(source.read_bytes()).hexdigest(), 'modes': {}}
    for mode in ('strict', 'loose'):
        result['modes'][mode] = {}
        for name in models:
            path = root / 'predictions' / name / 'external_test.jsonl'
            rows = sorted(map(json.loads, path.read_text().splitlines()), key=lambda r: r['key'])
            assert {r['key'] for r in rows} == set(labels)
            y = [int(labels[r['key']] > 0) if mode == 'strict' else int(labels[r['key']] == 1) for r in rows]
            scores = [r['score'] for r in rows]
            measured = evaluate(y, scores)
            candidates = measured.pop('pr_curve')[1:] + [measured['operating_point']]
            best = max(candidates, key=lambda p: (Fraction(2*p['tp'], 2*p['tp']+p['fp']+p['fn']),
                                                  p['tp']+p['tn'], p['threshold']))
            checked = operating_point(y, scores, best['threshold'])
            assert all(checked[k] == best[k] for k in checked)
            result['modes'][mode][name] = {'selected': checked, 'original_at0.5': measured['operating_point'],
                'always_unsafe': operating_point(y, scores, 0),
                'average_precision': measured['average_precision'],
                'pr_auc_trapezoid': measured['pr_auc_trapezoid'], 'roc_auc': measured['roc_auc'],
                'predictions_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
            print(mode, name, json.dumps(checked), flush=True)
    (root / 'posthoc_f1_thresholds.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    run(p.parse_args().root)
