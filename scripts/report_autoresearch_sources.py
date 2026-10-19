"""Diagnostic source/domain slices of fixed event scores; no threshold fitting."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path

from graph_safety.metrics import evaluate


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(',', ':')).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def report(args):
    if not math.isfinite(args.threshold) or not 0 <= args.threshold <= 1:
        raise ValueError('threshold must be finite and in [0, 1]')
    action_metadata, dialog_metadata, action_dialogs, expected_actions, inputs = {}, {}, {}, set(), {}
    paths = list(args.manifest)
    if args.data_audit.is_file():
        paths.insert(0, args.data_audit)
    elif not paths:
        raise FileNotFoundError(f'No attribution available: {args.data_audit}; supply --data-audit or --manifest')
    elif args.data_audit != DEFAULT_AUDIT:
        raise FileNotFoundError(args.data_audit)
    for path in paths:
        data = json.loads(path.read_text())
        inputs[str(path.resolve())] = file_hash(path)
        if isinstance(data, dict) and isinstance(data.get('rows'), list):
            for row in data['rows']:
                if row['key'] in dialog_metadata and dialog_metadata[row['key']] != row:
                    raise ValueError(f'Conflicting dialog metadata: {row["key"]}')
                dialog_metadata[row['key']] = row
                expected_actions.update(row['action_keys'])
                for key in row['action_keys']:
                    if key in action_dialogs and action_dialogs[key] != row['key']:
                        raise ValueError(f'Action appears in multiple manifest dialogs: {key}')
                    action_dialogs[key] = row['key']
        elif isinstance(data, dict):
            for key, value in data.items():
                if not isinstance(value, dict) or 'sources' not in value or 'domains' not in value:
                    raise ValueError(f'Expected action metadata or split manifest: {path}')
                if key in action_metadata and action_metadata[key] != value:
                    raise ValueError(f'Conflicting action metadata: {key}')
                action_metadata[key] = value
        else:
            raise ValueError(f'Expected action metadata mapping or split manifest: {path}')

    events, seen = [], set()
    for line_number, line in enumerate(args.events.read_text().splitlines(), 1):
        if not line.strip():
            raise ValueError(f'Blank event row at line {line_number}; no silent skipping')
        raw = json.loads(line)
        span = raw.get('span', raw)
        key = span['key']
        if key in seen:
            raise ValueError(f'Duplicate event key: {key}')
        seen.add(key)
        label, score = span['label'], raw['score']
        if label not in (0, 1) or not isinstance(score, (int, float)) or isinstance(score, bool):
            raise ValueError(f'Invalid label/score: {key}')
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError(f'Score is not finite in [0, 1]: {key}')
        dialog = raw.get('dialog') or action_dialogs.get(key)
        if key in action_dialogs and dialog != action_dialogs[key]:
            raise ValueError(f'Dialog ID mismatch: {key}')
        meta = action_metadata.get(key)
        level = 'action'
        if meta is not None:
            if span.get('state_sha256') is not None and span['state_sha256'] != meta['state_sha256']:
                raise ValueError(f'State hash mismatch: {key}')
        else:
            meta = dialog_metadata.get(dialog)
            level = 'dialog_fallback' if meta is not None else 'unattributed'
        dm = dialog_metadata.get(dialog)
        if dm is not None:
            if key not in dm['action_keys']:
                raise ValueError(f'Action not in supplied dialog manifest: {key}')
            if raw.get('ids_sha256') is not None and raw['ids_sha256'] != dm['ids_sha256']:
                raise ValueError(f'Token-ID hash mismatch: {key}')
        attribution = {'level': level, 'sources': sorted(set(meta['sources'])) if meta else [],
                       'domains': sorted(set(meta['domains'])) if meta else []}
        events.append({'key': key, 'dialog': dialog, 'label': label, 'score': score,
            'state_sha256': span.get('state_sha256'), 'ids_sha256': raw.get('ids_sha256'),
            'input_line': line_number, 'input_row_sha256': digest(raw),
            'attribution': attribution, 'attribution_sha256': digest(attribution)})
    if not events:
        raise ValueError('Empty events file')

    def metrics(rows):
        result = evaluate([r['label'] for r in rows], [r['score'] for r in rows], args.threshold)
        result.pop('pr_curve')
        point = result['operating_point']
        negatives = len(rows) - result['n_positive']
        point['fpr'] = point['fp'] / negatives if negatives else None
        result['undefined_reason'] = ('no positive events: AP, trapezoidal PR-AUC and recall undefined'
                                      if not result['n_positive'] else
                                      'no negative events: ROC-AUC and FPR undefined' if not negatives else None)
        return result

    slices = {'source': defaultdict(list), 'domain': defaultdict(list),
              'source_signature': defaultdict(list), 'attribution_level': defaultdict(list)}
    for row in events:
        attribution = row['attribution']
        for field, category in [('sources', 'source'), ('domains', 'domain')]:
            for name in attribution[field] or ['__unattributed__']:
                slices[category][name].append(row)
        signature = json.dumps(attribution['sources'], ensure_ascii=False, separators=(',', ':'))
        slices['source_signature'][signature].append(row)
        slices['attribution_level'][attribution['level']].append(row)
    result = {'report_version': 1, 'diagnostic_only': True,
        'selection_criterion': 'Unchanged ordinary aggregate event AP; slices do not select candidates or fit thresholds.',
        'threshold': args.threshold, 'threshold_fitted': False,
        'events_path': str(args.events.resolve()), 'events_file_sha256': file_hash(args.events),
        'attribution_input_sha256': inputs, 'event_count': len(events),
        'ordered_event_keys_sha256': digest([r['key'] for r in events]),
        'aggregate': metrics(events),
        'slices': {category: {name: {'metrics': metrics(rows),
                    'event_keys': [r['key'] for r in rows],
                    'ordered_event_keys_sha256': digest([r['key'] for r in rows])}
                    for name, rows in sorted(groups.items())} for category, groups in slices.items()},
        'manifest_case_coverage': {'checked': bool(dialog_metadata),
            'expected_events': len(expected_actions) if dialog_metadata else None,
            'missing_event_keys': sorted(expected_actions - seen) if dialog_metadata else None,
            'unexpected_event_keys': sorted(seen - expected_actions) if dialog_metadata else None,
            'policy': 'All supplied prediction rows retained; coverage differences reported, never filtered.'},
        'events': events,
        'limitations': [
            'Per-action original provenance is preferred. Prediction-row sources are deliberately ignored because trainers emit dialog-level attribution.',
            'Source/domain slices overlap and must not be summed. Source signatures partition events by their full source combination, not independent datasets.',
            'StepSafe repackages AgentDoG; shared source names do not prove independent origin. Within-AgentDoG ranking is diagnostic of source-shortcut risk, not proof of its absence.',
            'AgentAlign may be all-safe: AP/PR-AUC/ROC-AUC are null; false positives, FPR and confusion remain meaningful.',
            'Missing domains are explicit __unattributed__ slices; no domain is inferred. StepShield:train is a partition tag, not a semantic domain.',
            'Dialog fallback can attribute an event to sources/domains belonging to other actions. Attribution level and hashes are recorded for every event.',
            'The original case set is the supplied events file; without split manifests, completeness versus an intended evaluation split cannot be established.',
            'Scores and labels are used only for reporting; no fitting, filtering, reweighting or threshold search occurs.'
        ]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'output': str(args.output), 'events': len(events),
                      'aggregate': result['aggregate'], 'manifest_case_coverage': result['manifest_case_coverage']},
                     ensure_ascii=False, indent=2, allow_nan=False))


DEFAULT_AUDIT = Path('/var/local/danefedov/experiments/agent-safety/runs/autoresearch-sixhour-v1/data/action_metadata_audit.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--events', type=Path, required=True)
    parser.add_argument('--data-audit', type=Path, default=DEFAULT_AUDIT,
                        help='Original per-action metadata; also accepts a split manifest')
    parser.add_argument('--manifest', type=Path, nargs='*', default=[],
                        help='Optional split manifests for case completeness, ID checks and fallback attribution')
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--output', type=Path, required=True, help='New JSON output; refuses to overwrite')
    report(parser.parse_args())
