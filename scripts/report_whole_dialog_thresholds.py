"""Saved-output comparison of every model; explicitly post-hoc balanced P/R thresholds."""
import argparse
import hashlib
import json
from pathlib import Path

from graph_safety.metrics import evaluate

ARMS = ('frozen_linear', 'frozen_linear_mono', 'lora_linear', 'lora_linear_mono', 'kev4b')
NAMES = dict(zip(ARMS, ('Frozen, э2', 'Frozen + mono, э2', 'LoRA, э1', 'LoRA + mono, э1', 'Kev-4B')))
MAIN = (('dojo', 'all_events', 'ToolSafe Dojo'), ('harm', 'all_events', 'ToolSafe Harm — strict'),
        ('harm', 'loose', 'ToolSafe Harm — loose'), ('stepguard', 'external_test', 'StepGuard — test'),
        ('atbench', 'all_events', 'ATBench — all-events'), ('rjudge', 'all_events', 'R-Judge — all-events'))


def read(path):
    return json.loads(path.read_text())


def main(root):
    report = {'selection': 'POST-HOC on benchmark labels; maximize min(precision,recall), ties F1, recall, threshold',
              'warning': 'Optimistic same-data threshold selection, NOT held-out performance or calibrated deployment thresholds. No new model inference.',
              'threshold_rule': 'score >= threshold, complete score ties kept together', 'suites': {}, 'source_sha256': {}}
    for suite in ('dojo', 'harm', 'stepguard', 'atbench', 'rjudge'):
        folder = root / suite
        baseline = read(folder / 'comparison.json')
        planned = {r['key']: r for r in read(folder / 'planned.json')}
        predictions = {}
        for arm in ARMS:
            path = folder / arm / 'predictions.jsonl'
            report['source_sha256'][str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
            predictions[arm] = {r['key']: r for r in map(json.loads, path.read_text().splitlines())}
        common = sorted(set.intersection(*(set(p) for p in predictions.values())))
        assert len(common) == baseline['common_cases']
        analyses = {}
        for name, reference in baseline['analyses'].items():
            keys = list(common)
            tool_only = name == 'tool_only'
            if name in ('tool_only', 'all_events_on_tool_cases'):
                keys = [k for k in keys if any(a['kind'] == 'tool_call' for a in predictions[ARMS[0]][k]['actions'])]
            elif name == 'definite_only':
                keys = [k for k in keys if planned[k]['original_score'] != .5]
            elif name in ('external_test', 'external_dev'):
                keys = [k for k in keys if planned[k]['split'] == name]
            elif name not in ('all_events', 'loose'):
                keys = [k for k in keys if planned[k].get('domain') == name]
            assert len(keys) == reference['n'], (suite, name)
            if not keys:
                continue
            analyses[name] = {'n': len(keys), 'models': {}}
            for arm in ARMS:
                rows = predictions[arm]
                labels = [int(planned[k]['original_score'] == 1) if name == 'loose' else planned[k]['label'] for k in keys]
                scores = [max(a['score'] for a in rows[k]['actions'] if not tool_only or a['kind'] == 'tool_call') for k in keys]
                metrics = evaluate(labels, scores)
                for metric in ('average_precision', 'pr_auc_trapezoid', 'roc_auc'):
                    actual, expected = metrics[metric], reference['metrics'][arm][metric]
                    assert actual == expected or abs(actual - expected) < 1e-12
                points = [p for p in metrics.pop('pr_curve') if p['threshold'] is not None
                          and p['precision'] is not None and p['recall'] is not None]
                best = max(points, key=lambda p: (min(p['precision'], p['recall']), p['f1'], p['recall'], p['threshold']))
                metrics['posthoc_balanced_operating_point'] = best
                metrics['both_precision_and_recall_feasible'] = {
                    str(target): any(min(p['precision'], p['recall']) >= target for p in points)
                    for target in (.7, .8, .9)}
                high_recall = [p for p in points if p['recall'] >= .8]
                metrics['best_precision_at_recall_at_least_0.8'] = max(high_recall, key=lambda p: (p['precision'], p['recall']))
                analyses[name]['models'][arm] = metrics
        report['suites'][suite] = analyses
    output = root / 'posthoc_balanced_thresholds.json'
    assert not output.exists(), 'Preserve previous reports; use a versioned output root'
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    lines = ['# Все модели на всех наборах', '',
        '**Пороги ниже подобраны постфактум на метках самих бенчмарков. Это оптимистическая диагностика, не независимая оценка.**', '',
        'Критерий: максимум min(precision, recall); при равенстве — F1, recall, больший порог. Решение score >= threshold.',
        'Чекпойнты: frozen — validation-best эпоха2, LoRA — зафиксированная эпоха1; mono — междействийный endpoint-ranking, не потокенный штраф.',
        'Новых model calls, смены меток, обучения, температуры или рабочего порога0.5 нет.', '']
    for suite, analysis, title in MAIN:
        block = report['suites'][suite][analysis]
        lines += [f'## {title} (n={block["n"]})', '',
            '| Модель | AP | PR-AUC | ROC-AUC | Порог* | Precision, % | Recall, % | F1, % | Acc, % |',
            '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
        for arm in ARMS:
            m = block['models'][arm]
            p = m['posthoc_balanced_operating_point']
            lines.append(f'| {NAMES[arm]} | {m["average_precision"]:.4f} | {m["pr_auc_trapezoid"]:.4f} | {m["roc_auc"]:.4f} | {p["threshold"]:.8g} | {100*p["precision"]:.2f} | {100*p["recall"]:.2f} | {100*p["f1"]:.2f} | {100*p["accuracy"]:.2f} |')
        lines += ['', 'Контроль: исходный порог0.5 (без подбора).', '',
            '| Модель | Precision, % | Recall, % | F1, % | Acc, % |', '|---|---:|---:|---:|---:|']
        for arm in ARMS:
            p = block['models'][arm]['operating_point']
            precision = 'не определён' if p['precision'] is None else f'{100*p["precision"]:.2f}'
            lines.append(f'| {NAMES[arm]} | {precision} | {100*p["recall"]:.2f} | {100*p["f1"]:.2f} | {100*p["accuracy"]:.2f} |')
        lines.append('')
    lines += ['## Ограничения', '',
        '- ATBench:981/1000,19 длинных траекторий исключены без обрезки; максимум по всем action-endpoints, включая Complete.',
        '- R-Judge:571/571, также максимум endpoint-скоров. Траекторные метки не перенесены на действия.',
        '- StepGuard3017test — наше разбиение опубликованного TRAIN-корпуса;161dev отдельно в JSON.',
        '- Harm strict:0.5unsafe, loose:0.5safe. Исходные скоры и правила не переписаны.',
        '- Native chat template; causal-prefix fallback там, где штатный шаблон переписывает прошлые сообщения. Это режим оценки, не смена обучения.',
        '- Один seed, ранее просмотренные наборы. Подобранные пороги не применены как новые рабочие пороги и не перенесены в serving.',
        '- JSON дополнительно сохраняет tool-only, matched all-events, domain/dev slices и проверку достижимости precision/recall70/80/90%.', '']
    (root / 'FULL_COMPARISON_POSTHOC.md').write_text('\n'.join(lines))
    print('\n'.join(line for line in lines if line.startswith('|') or line.startswith('## ')))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    main(p.parse_args().root)
