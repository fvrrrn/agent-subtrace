"""Collect portable historical results and index files omitted from GitHub.

Run from the repository root. Original historical files are never removed.
"""
import hashlib
import json
from pathlib import Path
import shutil

from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
LIMIT = 10 * 1024 * 1024
SKIP_DIRS = {'.git', '.venv', '__pycache__', 'hf-cache', 'weights', 'model', 'node_modules'}
BINARY = {'.pt', '.pth', '.safetensors', '.bin', '.zip', '.pyc', '.so', '.dylib'}
PORTABLE = {'.md', '.json', '.jsonl', '.csv', '.tsv', '.txt', '.log', '.png', '.svg', '.pdf', '.py', '.toml', '.yaml', '.yml'}


def main():
    excluded = []
    copied = []
    experiment = ROOT.parent / 'experiment'
    if experiment.exists():
        files = [p for p in experiment.rglob('*') if p.is_file()
                 and not any(part in SKIP_DIRS for part in p.relative_to(experiment).parts)]
        for source in tqdm(files, desc='Collect historical reports/results', unit='file'):
            relative = source.relative_to(experiment)
            if source.suffix not in PORTABLE or source.suffix in BINARY:
                continue
            if source.stat().st_size > LIMIT:
                excluded.append({'path': '../experiment/' + relative.as_posix(),
                                 'bytes': source.stat().st_size, 'reason': 'large historical raw artifact'})
                continue
            target = ROOT / 'legacy' / 'experiment' / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                if source.read_bytes() != target.read_bytes():
                    raise RuntimeError(f'Refusing to overwrite different file: {target}')
            else:
                shutil.copy2(source, target)
            copied.append(target.relative_to(ROOT).as_posix())

    for source in tqdm(list((ROOT / 'artifacts').rglob('*')), desc='Index research artifacts', unit='entry'):
        if not source.is_file() or '__pycache__' in source.parts:
            continue
        if source.stat().st_size > LIMIT or source.suffix in BINARY:
            excluded.append({'path': source.relative_to(ROOT).as_posix(),
                             'bytes': source.stat().st_size, 'reason': 'weights, binary archive or large raw artifact'})

    # Hash only omitted files; retain their identity without putting weights in Git.
    for entry in tqdm(excluded, desc='Hash preserved external artifacts', unit='file'):
        source = ROOT / entry['path']
        digest = hashlib.sha256()
        with source.open('rb') as stream:
            for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
                digest.update(chunk)
        entry['sha256'] = digest.hexdigest()

    inventory = ROOT / 'manifests' / 'repository_collection.json'
    inventory.write_text(json.dumps({'portable_limit_bytes': LIMIT,
        'historical_copies': copied, 'excluded_preserved_files': excluded,
        'server_root': '/var/local/danefedov/experiments/agent-safety',
        'note': 'Local collection, not a full server inventory. excluded_preserved_files is a historical size/binary classification, not current Git exclusion; .gitattributes defines LFS storage.'},
        ensure_ascii=False, indent=2) + '\n')
    # Git/LFS policy lives in .gitignore/.gitattributes; do not generate thousands
    # of per-file ignore rules or overwrite that policy during collection.

    rows = ['# Полный указатель локальных экспериментов', '',
            'Автоматически собранный список; наличие каталога не означает успешное завершение.', '',
            '| Каталог | Отчёты |', '|---|---|']
    for directory in sorted((ROOT / 'artifacts').iterdir()):
        if not directory.is_dir():
            continue
        reports = sorted(directory.glob('*.md'))
        links = ' · '.join(f'[{p.name}](../{p.relative_to(ROOT).as_posix()})' for p in reports)
        rows.append(f'| [{directory.name}](../artifacts/{directory.name}/) | {links or "JSON/логи в каталоге"} |')
    (ROOT / 'docs').mkdir(exist_ok=True)
    (ROOT / 'docs' / 'EXPERIMENT_INDEX.md').write_text('\n'.join(rows) + '\n')
    print(f'Collected {len(copied)} historical files; indexed {len(excluded)} preserved external files.')


if __name__ == '__main__':
    main()
