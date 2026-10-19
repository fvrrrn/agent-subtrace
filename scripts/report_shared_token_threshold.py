"""One post-hoc threshold for fixed LoRA top10 readout, equal weight per benchmark."""
import argparse
from itertools import groupby
import hashlib
import json
from pathlib import Path

from graph_safety.metrics import confusion, evaluate

SUITES = ('dojo', 'harm', 'stepguard', 'atbench', 'rjudge')


def read(p):
    return json.loads(p.read_text())


def macro(points):
    names = ('precision', 'recall', 'f1', 'accuracy', 'specificity', 'balanced_accuracy')
    return {name: sum(p[name] if p[name] is not None else 0. for p in points)/len(points) for name in names}


def main(a):
    rows = {}
    hashes = {}
    for suite in SUITES:
        p = a.tokens / suite / 'lora_linear/predictions.jsonl'
        q = a.parent / suite / 'planned.json'
        plans = {r['key']: r for r in read(q)}
        rows[suite] = []
        for r in map(json.loads, p.read_text().splitlines()):
            if suite == 'stepguard' and plans[r['key']]['split'] != 'external_test':
                continue
            assert r['label'] == plans[r['key']]['label']
            rows[suite].append({'key': r['key'], 'label': r['label'],
                'score': max(x['methods']['top10_mean'] for x in r['actions']),
                'original_score': plans[r['key']].get('original_score')})
        for path in (p, q):
            hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    counts = {s: [0, 0, sum(r['label'] for r in rows[s]), sum(1-r['label'] for r in rows[s])] for s in SUITES}
    ordered = sorted(((r['score'], s, r['label']) for s in SUITES for r in rows[s]), reverse=True)
    best = None
    for threshold, group in groupby(ordered, key=lambda x: x[0]):
        for _, suite, label in group:
            counts[suite][0 if label else 1] += 1
        points = [confusion(*counts[s]) for s in SUITES]
        averages = macro(points)
        criterion = (min(averages['precision'], averages['recall']), averages['f1'], averages['recall'], threshold)
        if best is None or criterion > best[0]:
            best = (criterion, threshold, averages)
    threshold = best[1]
    results = {}
    for suite in SUITES:
        labels = [r['label'] for r in rows[suite]]
        scores = [r['score'] for r in rows[suite]]
        m = evaluate(labels, scores, threshold)
        m.pop('pr_curve')
        results[suite] = m
    averages = macro([m['operating_point'] for m in results.values()])
    assert all(abs(averages[k]-best[2][k]) < 1e-12 for k in averages)
    diagnostics = {}
    for mode in ('loose', 'definite_only'):
        chosen = [r for r in rows['harm'] if mode != 'definite_only' or r['original_score'] != .5]
        labels = [int(r['original_score'] == 1) for r in chosen]
        m = evaluate(labels, [r['score'] for r in chosen], threshold)
        m.pop('pr_curve')
        diagnostics['harm_' + mode] = m
    report = {'model': 'LoRA linear without ranking, locked epoch1', 'aggregation': 'mean top ceil10% action probabilities; trajectory max',
        'threshold': threshold, 'criterion': 'max min(macro precision, macro recall), ties macro F1, macro recall, largest threshold',
        'primary': list(SUITES), 'weight_per_benchmark': .2, 'harm_primary': 'strict731; loose and definite416 only additional diagnostics',
        'undefined_precision_for_selection': 'zero when no alerts, so no dataset can disappear from objective',
        'macro': averages, 'metrics': results, 'diagnostics': diagnostics, 'source_sha256': hashes,
        'limitation': 'POST-HOC selected using labels from all five benchmarks. Not held-out transfer or deployment calibration. One shared number, NOT arithmetic mean of separately optimized thresholds.'}
    a.output.mkdir(parents=True, exist_ok=False)
    (a.output / 'shared_threshold.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    lines = ['# Один общий порог: LoRA без mono, top10, epoch1', '',
        f'**Порог: {threshold:.12g}.** Равный вес5наборов. Максимум min(macro precision,macro recall).',
        '**Подбор на метках тех же бенчмарков: оптимистическая диагностика, не независимая оценка.**', '',
        '| Набор | n | AP | PR-AUC | ROC-AUC | Precision,% | Recall,% | F1,% | Acc,% | FPR,% |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name, m in {**results, **diagnostics}.items():
        p = m['operating_point']
        lines.append(f'| {name} | {m["n_examples"]} | {m["average_precision"]:.4f} | {m["pr_auc_trapezoid"]:.4f} | {m["roc_auc"]:.4f} | {100*p["precision"]:.2f} | {100*p["recall"]:.2f} | {100*p["f1"]:.2f} | {100*p["accuracy"]:.2f} | {100*(1-p["specificity"]):.2f} |')
    lines += ['', 'Macro по5основнымнаборам: ' + ', '.join(f'{k}={100*v:.2f}%' for k, v in averages.items()), '',
        'StepGuard — наше3017test разбиение published TRAIN. ATBench981/1000,19 длиновых исключений. R-Judge571, не569изисходнойстатьи.',
        'Дляtrajectoryданных max поevent-агрегатам causal pre-execution, не классификация завершённого trace с исходами.',
        'Harmdefinite_only416 отдельно для сопоставления с StepGuardpaper; это НЕ strict731 и НЕ loose731.', '']
    (a.output / 'SHARED_THRESHOLD.md').write_text('\n'.join(lines))
    print('\n'.join(lines))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tokens', type=Path, required=True)
    p.add_argument('--parent', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    main(p.parse_args())
