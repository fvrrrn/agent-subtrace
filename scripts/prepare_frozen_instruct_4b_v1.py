"""Strict native whole-dialog admission for the authorized frozen Instruct comparison."""
from collections import Counter
import hashlib
from pathlib import Path
import shutil
import time

from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

import prepare_autoresearch_backbone as backbone
from prepare_whole_dialogs import digest, write

ROOT = Path('/var/local/danefedov/experiments/agent-safety')
RUN = ROOT / 'runs/frozen-instruct-4b-v1'
MODEL = 'Qwen/Qwen3.5-4B'
REVISION = '851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'
DEADLINE = 1792345185


def main():
    assert time.time() < DEADLINE
    out = RUN / 'data'
    out.mkdir(parents=True, exist_ok=False)
    snapshot = Path(snapshot_download(MODEL, revision=REVISION, cache_dir=backbone.HUB, allow_patterns=[
        'config.json', 'tokenizer*', 'vocab.json', 'merges.txt', 'README.md', 'LICENSE',
        'chat_template*', 'model.safetensors.index.json']))
    config = backbone.parent.read(snapshot / 'config.json')
    assert config['model_type'] == 'qwen3_5' and config['text_config']['hidden_size'] == 2560
    assert 'post-trained model' in (snapshot / 'README.md').read_text()
    old_snapshot = backbone.HUB / ('models--' + backbone.PARENT_BASE.replace('/', '--')) / 'snapshots' / backbone.PARENT_REVISION
    old = AutoTokenizer.from_pretrained(str(old_snapshot), local_files_only=True)
    new = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True)
    assert old.is_fast and new.is_fast
    assert hashlib.sha256(old.chat_template.encode()).hexdigest() == backbone.parent.TEMPLATE_SHA256
    mapping = backbone.parent.read(backbone.parent.DEFAULT_SPLITS)
    mapping = {name: mapping[name] for name in ('train_small', 'dev_select')}
    assert {k: len(v) for k, v in mapping.items()} == {'train_small': 2009, 'dev_select': 301}
    lookup = {key: split for split, keys in mapping.items() for key in keys}
    assert len(lookup) == 2310
    source = ROOT / 'datasets/whole-dialog-4b-v2/dialogs.json'
    # Only the authorized rows are retained, rendered, inspected or evaluated.
    rows = {r['key']: r for r in backbone.parent.read(source) if r['key'] in lookup}
    assert rows.keys() == lookup.keys()
    accepted, rejected = [], []
    for key, split in lookup.items():
        assert time.time() < DEADLINE
        try:
            accepted.append(backbone.prepare_row(old, new, rows[key], split))
        except (ValueError, TypeError) as error:
            rejected.append({'key': key, 'split': split, 'reason': str(error),
                             'parent_row_sha256': digest(rows[key])})
    write(out / 'dialogs.json', accepted)
    write(out / 'rejections.json', rejected)
    write(out / 'splits.json', mapping)
    sources = RUN / 'model_sources'
    sources.mkdir()
    for name in ('README.md', 'config.json', 'tokenizer_config.json', 'LICENSE'):
        shutil.copy2(snapshot / name, sources / name)
    audit = {'base': MODEL, 'revision': REVISION, 'snapshot': str(snapshot),
             'hidden_size': 2560, 'dialogs_sha256': backbone.parent.file_hash(out / 'dialogs.json'),
             'template_sha256': hashlib.sha256(new.chat_template.encode()).hexdigest(),
             'parent_template_sha256': hashlib.sha256(old.chat_template.encode()).hexdigest(),
             'source_dialogs_sha256': backbone.parent.file_hash(source),
             'splits_sha256': backbone.parent.file_hash(out / 'splits.json'),
             'requested_counts': {k: len(v) for k, v in mapping.items()},
             'admitted_counts': dict(Counter(r['split'] for r in accepted)),
             'exclusion_reasons': dict(Counter(r['reason'] for r in rejected)),
             'launch_allowed': not rejected, 'limit': 16384, 'no_truncation': True,
             'actions': sum(len(r['spans']) for r in accepted),
             'max_tokens': max((r['tokens'] for r in accepted), default=0),
             'source_files_sha256': {p.name: backbone.parent.file_hash(p) for p in sources.iterdir()},
             'protocol': 'Native apply_chat_template defaults, no rationale generation; every labeled causal prefix must match full dialogue; any rejection blocks all GPU work.'}
    write(out / 'data_audit.json', audit)
    print(audit, flush=True)
    if rejected:
        write(RUN / 'status.json', {'status': 'blocked_native_admission', 'audit': audit})
        raise SystemExit('Native whole-dialog admission failed; no weight download/cache/training permitted.')


if __name__ == '__main__':
    main()
