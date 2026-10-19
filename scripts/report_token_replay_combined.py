"""Combine completed frozen/LoRA token reports without new inference or fitting."""
import argparse
import hashlib
import json
from pathlib import Path

NAMES = {'frozen_linear': 'Frozen, э2', 'frozen_linear_mono': 'Frozen + mono, э2',
         'lora_linear': 'LoRA, э1', 'lora_linear_mono': 'LoRA + mono, э1'}
MAIN = [('dojo', 'main'), ('harm', 'main'), ('harm', 'loose'),
        ('stepguard', 'external_test'), ('atbench', 'main'), ('rjudge', 'main')]


def read(p):
    return json.loads(p.read_text())


def combine(frozen, lora, output):
    reports = [read(frozen / 'frozen_comparison.json'), read(lora / 'lora_comparison.json')]
    assert reports[0]['methods'] == reports[1]['methods']
    arms = reports[0]['arms'] + reports[1]['arms']
    merged = {'arms': arms, 'methods': reports[0]['methods'], 'suites': {}, 'audits': {},
              'source_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in (frozen / 'frozen_comparison.json', lora / 'lora_comparison.json')}}
    for suite in reports[0]['suites']:
        signatures = []
        for root, report in zip((frozen, lora), reports):
            for arm in report['arms']:
                rows = [json.loads(s) for s in (root / suite / arm / 'predictions.jsonl').read_text().splitlines()]
                signatures.append({r['key']: (r['label'], r['ids_sha256'], tuple(
                    (a['key'], a['state_sha256'], a['action_tokens']) for a in r['actions'])) for r in rows})
                done = read(root / suite / arm / 'complete.json')
                assert done['max_endpoint_probability_error'] == done['max_endpoint_raw_error'] == 0
                merged['audits'][suite + '/' + arm] = done
        assert all(s == signatures[0] for s in signatures)
        blocks = {}
        for analysis, first in reports[0]['suites'][suite].items():
            second = reports[1]['suites'][suite][analysis]
            assert first['n'] == second['n']
            blocks[analysis] = {'n': first['n'], 'metrics': {**first['metrics'], **second['metrics']}}
        merged['suites'][suite] = blocks
    output.mkdir(parents=True, exist_ok=False)
    (output / 'comparison.json').write_text(json.dumps(merged, ensure_ascii=False, indent=2) + '\n')
    lines = ['# Все4 Mamba: все токеновые агрегации на всех наборах', '',
        'Обе frozen и обе LoRA завершены. В каждой модели все10955 endpoint-logits и probabilities побитово совпали с исходным benchmark.',
        'Raw token logits/IDs/spans сохранены серверно. Модели не дообучались. Метки и входы всех4условий совпадают.', '',
        '**P/R/F1/Acc/пороги со звёздочкой подобраны постфактум по меткам тех же наборов, max min(P,R). Не held-out calibration.**',
        'R@FPR1%/5% — также диагностика по benchmark-кривой. Рабочий threshold0.5 не изменён; его матрицы ошибок сохранены в JSON.', '']
    for suite, analysis in MAIN:
        block = merged['suites'][suite][analysis]
        lines += [f'## {suite}/{analysis}: n={block["n"]}', '',
            '| Model | Aggregation | AP | PR-AUC | ROC-AUC | Threshold* | P*, % | R*, % | F1*, % | Acc*, % | R@FPR1%, % | R@FPR5%, % |',
            '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
        for arm in arms:
            for method in merged['methods']:
                m = block['metrics'][arm][method]
                p = m['posthoc_balanced_point']
                lines.append(f'| {NAMES[arm]} | {method} | {m["average_precision"]:.4f} | {m["pr_auc_trapezoid"]:.4f} | {m["roc_auc"]:.4f} | {p["threshold"]:.8g} | {100*p["precision"]:.2f} | {100*p["recall"]:.2f} | {100*p["f1"]:.2f} | {100*p["accuracy"]:.2f} | {100*m["recall_at_fpr_0.01"]:.2f} | {100*m["recall_at_fpr_0.05"]:.2f} |')
        lines.append('')
    lines += ['## Ограничения', '',
        '- Фиксированная сетка4методов, один seed, просмотренные наборы; no new model/method/threshold selection for deployment.',
        '- ATBench981/1000,19 исходных длинных исключений. Траекторные метки не используются как action/token gold.',
        '- StepGuard:3017test — наше разбиение опубликованного TRAIN-корпуса;161dev отдельно в JSON.',
        '- Native template и режим causal-prefix fallback не менялись. Endpoint, max/top10/tail8 считаются только по текущему action body.',
        '- Native Kev4B не имеет этих токеновых heads и не перезапускался; его прежние matched action/trajectory результаты сохранены в whole_dialog_benchmarks_v1.', '']
    (output / 'ALL_MODELS_TOKEN_COMPARISON.md').write_text('\n'.join(lines))
    print(json.dumps({'models': len(arms), 'methods': len(merged['methods']), 'main_rows': len(MAIN)*len(arms)*len(merged['methods']),
                      'all_endpoint_errors': 0, 'output': str(output)}, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--frozen', type=Path, required=True)
    p.add_argument('--lora', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    combine(a.frozen, a.lora, a.output)
