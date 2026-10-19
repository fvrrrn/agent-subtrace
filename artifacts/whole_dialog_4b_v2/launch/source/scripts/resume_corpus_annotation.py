"""Checkpoint the four-GPU annotation and continue only missing judgments on two GPUs."""
import argparse
import asyncio
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path('/var/local/danefedov/experiments/agent-safety')
DATA = ROOT / 'datasets/corpus-v1/annotations'
CHECKPOINT = DATA / 'strong_bulk_checkpoint_v1'


def write(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')


def load_calls(path):
    outputs = defaultdict(dict)
    for line in path.open():
        r = json.loads(line)
        if r['teacher'] in outputs[r['key']]:
            raise ValueError(f'Duplicate checkpoint judgment: {r["key"]}')
        outputs[r['key']][r['teacher']] = r
    return outputs


def aggregate(rows, outputs, folder):
    counts = Counter()
    with (folder / 'silver.jsonl').open('w') as silver, (folder / 'quarantine.jsonl').open('w') as quarantine:
        for row in rows:
            judgments = list(outputs.get(row['key'], {}).values())
            if len(judgments) != 2:
                counts['pending_actions'] += 1
                continue
            valid = [j for j in judgments if j['status'] == 'complete' and
                     j['evidence_grounded'] and j['finish_reason'] == 'stop']
            verdicts = {j['result']['verdict'] for j in valid}
            good = len(valid) == 2 and len(verdicts) == 1 and 'uncertain' not in verdicts
            label = int(next(iter(verdicts)) == 'unsafe') if good else None
            published = {p['source_label'] for p in row['provenance'] if p['source_label'] is not None}
            if good and published and published != {label}:
                good = False
            result = dict(row, label=label if good else None,
                          status='silver_agreement' if good else 'quarantined',
                          teacher_labels=[{'teacher': j['teacher'], 'result': j.get('result'),
                                           'status': j['status']} for j in judgments])
            (silver if good else quarantine).write(json.dumps(result, ensure_ascii=False) + '\n')
            counts['silver' if good else 'quarantine'] += 1
            if good:
                counts[f'label_{label}'] += 1
                counts['source:' + row['provenance'][0]['source']] += 1
    return dict(counts)


def frozen_imports():
    sys.path.insert(0, str(ROOT / 'runs/corpus-v1/strong_bulk_source'))
    import annotate_safety_corpus as original
    return original


def snapshot():
    original = frozen_imports()
    old = DATA / 'strong_bulk'
    paused = json.loads((old / 'PAUSED_FOR_2GPU.json').read_text())
    h = hashlib.sha256((old / 'calls.jsonl').read_bytes()).hexdigest()
    assert h == paused['calls_sha256']
    CHECKPOINT.mkdir(exist_ok=False)
    rows = original.selection(40000, whole_groups=True, input_name='candidates_v3')
    expected = json.loads((old / 'selection.json').read_text())
    assert [(r['key'], r['group'], r['split']) for r in rows] == [
        (r['key'], r['group'], r['split']) for r in expected]
    with (CHECKPOINT / 'selection_records.jsonl').open('w') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    outputs = load_calls(old / 'calls.jsonl')
    summary = aggregate(rows, outputs, CHECKPOINT)
    summary.update(original_saved_calls=paused['saved_calls'], original_calls_sha256=h,
                   selected=len(rows), immutable_partial_snapshot=True, human_gold=False,
                   selection_sha256=hashlib.sha256((CHECKPOINT / 'selection_records.jsonl').read_bytes()).hexdigest())
    write(CHECKPOINT / 'summary.json', summary)
    print(json.dumps(summary, indent=2), flush=True)


async def resume():
    import httpx
    original = frozen_imports()
    from audit_structured_evidence import support
    folder = DATA / 'strong_bulk_2gpu'
    folder.mkdir(exist_ok=False)
    workers = json.loads((ROOT / 'runs/corpus-v1/serving-strong-2gpu/readiness.json').read_text())
    assert {w['gpu'] for w in workers} == {0, 1}
    assert len({w['model'] for w in workers}) == 2
    rows = [json.loads(s) for s in (CHECKPOINT / 'selection_records.jsonl').open()]
    outputs = load_calls(DATA / 'strong_bulk/calls.jsonl')
    missing = [(w, r) for r in rows for w in workers if w['model'] not in outputs.get(r['key'], {})]
    write(folder / 'protocol.json', {'workers': workers, 'missing_judgments': len(missing),
          'original_protocol': json.loads((DATA / 'strong_bulk/protocol.json').read_text()),
          'resume_policy': 'Only absent key/checkpoint pairs; existing errors/disagreements are terminal and not retried.',
          'source': str(CHECKPOINT), 'original_calls_sha256': json.loads((CHECKPOINT / 'summary.json').read_text())['original_calls_sha256']})
    sems = {w['gpu']: asyncio.Semaphore(8) for w in workers}
    completed = 0
    started = time.monotonic()
    async with httpx.AsyncClient(timeout=httpx.Timeout(900), limits=httpx.Limits(max_connections=24)) as client:
        with (folder / 'calls.jsonl').open('w') as stream:
            async def one(worker, row):
                nonlocal completed
                async with sems[worker['gpu']]:
                    result = await original.call(client, worker, row, thinking=True)
                    choices = (result.get('raw_response') or {}).get('choices', [])
                    if result['status'] == 'error' and choices and choices[0].get('finish_reason') == 'length':
                        first = result
                        result = await original.call(client, worker, row, thinking=True, max_tokens=4096)
                        result['initial_length_attempt'] = first
                    if result['status'] == 'complete':
                        result['evidence_support'] = [support(q, row['state']) for q in result['result'].get('evidence', [])]
                        result['evidence_grounded'] = bool(result['evidence_support']) and all(result['evidence_support'])
                stream.write(json.dumps(result, ensure_ascii=False) + '\n')
                stream.flush()
                outputs[row['key']][worker['model']] = result
                completed += 1
                if completed % 64 == 0:
                    print(f'2GPU judgments {completed}/{len(missing)}', flush=True)
            await asyncio.gather(*(one(w, r) for w, r in missing))
    summary = aggregate(rows, outputs, folder)
    summary.update(new_judgments=completed, seconds=time.monotonic() - started, original_calls_preserved=True)
    write(folder / 'summary.json', summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('phase', choices=('snapshot', 'resume'))
    a = p.parse_args()
    if a.phase == 'snapshot':
        snapshot()
    else:
        asyncio.run(resume())
