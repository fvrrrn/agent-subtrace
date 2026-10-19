"""Native 0.8B whole-dialog preparation; cache is a separately authorized GPU phase."""
import argparse
from collections import Counter
import hashlib
import json
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from prepare_autoresearch_prompt import action_span, file_hash, read, render
from prepare_whole_dialogs import BASE as PARENT_BASE, REVISION as PARENT_REVISION, digest, write

BASE = 'Qwen/Qwen3.5-0.8B-Base'
REVISION = 'dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68'
TEMPLATE = '273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80'
SERVER = Path('/var/local/danefedov/experiments/agent-safety')
SNAPSHOT = SERVER / 'models/huggingface/hub/models--Qwen--Qwen3.5-0.8B-Base/snapshots' / REVISION
COUNTS = {'train_small': 2009, 'dev_select': 301, 'dev_confirm': 299, 'original_validation': 425}
LIMIT = 16384
DIM = 1024


def splits(path):
    source = read(path)
    if set(source) != set(COUNTS) | {'train_full'}:
        raise ValueError('expected fixed five-list prospective manifest')
    mapping = {k: source[k] for k in COUNTS}
    if {k: len(v) for k, v in mapping.items()} != COUNTS:
        raise ValueError('fixed split counts changed')
    if not set(mapping['train_small']) <= set(source['train_full']):
        raise ValueError('train_small not contained in train_full')
    lookup = {}
    for split, keys in mapping.items():
        for key in keys:
            if not isinstance(key, str) or key in lookup:
                raise ValueError('invalid or overlapping split keys')
            lookup[key] = split
    return mapping, lookup


def model_files(snapshot):
    if snapshot.name != REVISION:
        raise ValueError('wrong snapshot revision')
    config = read(snapshot / 'config.json')
    if config['text_config']['hidden_size'] != DIM:
        raise ValueError('unexpected hidden size')
    index = read(snapshot / 'model.safetensors.index.json')
    names = ['config.json', 'tokenizer_config.json', 'tokenizer.json', 'vocab.json',
             'merges.txt', 'model.safetensors.index.json'] + sorted(set(index['weight_map'].values()))
    return {name: {'bytes': (snapshot / name).stat().st_size,
                   'sha256': file_hash(snapshot / name)} for name in names}


def prepare_row(parent_tok, tok, row, split):
    messages, tools = row['messages'], row['tools']
    old_text, old_ids, old_offsets = render(parent_tok, messages, tools)
    if old_ids != row['ids'] or digest(old_ids) != row['ids_sha256'] or len(old_ids) != row['tokens']:
        raise ValueError('parent_native_input_mismatch')
    wanted = {(s['start'], s['end']) for s in row['spans']}
    endpoints = {}
    for stop, message in enumerate(messages, 1):
        if message['role'] != 'assistant':
            continue
        prefix = parent_tok.apply_chat_template(messages[:stop], tools=tools, tokenize=False,
                                                add_generation_prompt=False)
        try:
            start, end, sha = action_span(parent_tok, prefix, old_text, old_ids, old_offsets)
        except ValueError:
            continue
        if (start, end) in wanted:
            if (start, end) in endpoints:
                raise ValueError('ambiguous_parent_endpoint')
            endpoints[start, end] = (stop, sha)
    if set(endpoints) != wanted or len(wanted) != len(row['spans']):
        raise ValueError('unrecoverable_parent_endpoints')
    text, ids, offsets = render(tok, messages, tools)
    if len(ids) > LIMIT:
        raise ValueError(f'whole_dialog_context_overflow:{len(ids)}')
    # Verify every assistant causal prefix, including unlabelled history. No fallback,
    # template surgery, truncation, independent-prefix examples or injected instruction.
    native_spans = {}
    for stop, message in enumerate(messages, 1):
        if message['role'] == 'assistant':
            prefix = tok.apply_chat_template(messages[:stop], tools=tools, tokenize=False,
                                             add_generation_prompt=False)
            native_spans[stop] = action_span(tok, prefix, text, ids, offsets)
    spans = []
    for old in row['spans']:
        stop, parent_sha = endpoints[old['start'], old['end']]
        start, end, sha = native_spans[stop]
        spans.append(dict(old, start=start, end=end, message_stop=stop,
                          parent_start=old['start'], parent_end=old['end'],
                          parent_prefix_ids_sha256=parent_sha, prefix_ids_sha256=sha))
    if any(a['end'] > b['start'] for a, b in zip(spans, spans[1:])):
        raise ValueError('overlapping_action_spans')
    return dict(row, split=split, parent_split=row['split'], ids=ids, tokens=len(ids),
                ids_sha256=digest(ids), spans=spans, parent_ids_sha256=row['ids_sha256'],
                parent_tokens=row['tokens'], messages_tools_sha256=digest([messages, tools]))


def prepare(args):
    if args.shard != 0 or args.shard_count != 1:
        raise ValueError('prepare is unsharded')
    from transformers import AutoTokenizer
    mapping, lookup = splits(args.splits)
    source = args.data / 'dialogs.json' if args.data.is_dir() else args.data
    rows = read(source)
    if len({r['key'] for r in rows}) != len(rows) or set(lookup) - {r['key'] for r in rows}:
        raise ValueError('duplicate or missing parent keys')
    files = model_files(args.snapshot)
    parent = AutoTokenizer.from_pretrained(PARENT_BASE, revision=PARENT_REVISION, local_files_only=True)
    tok = AutoTokenizer.from_pretrained(str(args.snapshot), local_files_only=True)
    if not tok.is_fast or hashlib.sha256(tok.chat_template.encode()).hexdigest() != TEMPLATE:
        raise ValueError('native_template_mismatch')
    args.root.mkdir(parents=True, exist_ok=False)
    accepted, rejected = [], []
    for row in rows:
        if row['key'] not in lookup:
            continue
        try:
            accepted.append(prepare_row(parent, tok, row, lookup[row['key']]))
        except (ValueError, TypeError) as error:
            rejected.append(dict(key=row['key'], split=lookup[row['key']], reason=str(error),
                                 parent_ids_sha256=row['ids_sha256'], parent_tokens=row['tokens']))
    write(args.root / 'dialogs.json', accepted)
    write(args.root / 'rejections.json', rejected)
    write(args.root / 'splits.json', mapping)
    common = {k: [r['key'] for r in accepted if r['split'] == k] for k in mapping}
    write(args.root / 'common_splits.json', common)
    write(args.root / 'matched_action_keys.json', {k: [s['key'] for r in accepted if r['split'] == k
                                                      for s in r['spans']] for k in mapping})
    audit = dict(base=BASE, revision=REVISION, hidden_size=DIM, model_files=files,
                 template_sha256=TEMPLATE, tokenizer_vocab_sha256=digest(tok.get_vocab()),
                 source_dialogs_sha256=file_hash(source), splits_sha256=file_hash(args.splits),
                 dialogs_sha256=file_hash(args.root / 'dialogs.json'),
                 common_splits_sha256=file_hash(args.root / 'common_splits.json'),
                 requested_counts=COUNTS, common_counts={k: len(v) for k, v in common.items()},
                 exclusion_reasons=dict(Counter(r['reason'] for r in rejected)),
                 dialogs=len(accepted), actions=sum(len(r['spans']) for r in accepted),
                 max_tokens=max((r['tokens'] for r in accepted), default=0), limit=LIMIT,
                 no_truncation=True, extra_instruction=False,
                 causal_prefix_audit='all assistant endpoints; exact native full-dialog prefix IDs',
                 state_hash_policy='original per-action state_sha256 preserved unchanged; not redefined as token hash',
                 labels_used_only_as_output_metadata=True)
    write(args.root / 'data_audit.json', audit)
    print(json.dumps(audit, indent=2), flush=True)


def cache(args):
    if not args.authorize_gpu_cache:
        raise ValueError('cache requires explicit --authorize-gpu-cache; prepare is CPU-only')
    import torch
    import transformers
    from transformers import Qwen3_5TextModel
    from train_whole_dialogs import encode, frozen_hash
    _, lookup = splits(args.splits)
    audit = read(args.data / 'data_audit.json')
    rows = read(args.data / 'dialogs.json')
    if (audit['base'], audit['revision'], audit['template_sha256']) != (BASE, REVISION, TEMPLATE):
        raise ValueError('prepared_identity_mismatch')
    if file_hash(args.data / 'dialogs.json') != audit['dialogs_sha256'] or file_hash(args.splits) != audit['splits_sha256']:
        raise ValueError('prepared_data_or_manifest_changed')
    if model_files(args.snapshot) != audit['model_files']:
        raise ValueError('snapshot_files_changed')
    if not rows or len({r['key'] for r in rows}) != len(rows):
        raise ValueError('empty_or_duplicate_prepared_rows')
    for row in rows:
        if (lookup.get(row['key']) != row['split'] or digest(row['ids']) != row['ids_sha256']
                or len(row['ids']) != row['tokens'] or row['tokens'] > LIMIT
                or digest([row['messages'], row['tools']]) != row['messages_tools_sha256']):
            raise ValueError('prepared_row_integrity_failure')
    folder = args.root / 'features'
    folder.mkdir(parents=True, exist_ok=True)
    shard_rows = rows[args.shard::args.shard_count]
    if (folder / f'complete{args.shard}.json').exists() or any((folder / (r['key'] + '.pt')).exists() for r in shard_rows):
        raise ValueError('refusing_existing_shard')
    model = Qwen3_5TextModel.from_pretrained(str(args.snapshot), local_files_only=True,
                    dtype=torch.bfloat16, attn_implementation='sdpa').cuda().eval()
    model.requires_grad_(False)
    weights = frozen_hash(model)
    packages = {}
    for name in ('triton', 'causal-conv1d', 'mamba-ssm', 'flash-linear-attention'):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    identity = dict(base=BASE, revision=REVISION, weights_sha256=weights, hidden_size=DIM,
                    model_files_sha256=digest(audit['model_files']), template_sha256=TEMPLATE,
                    tokenizer_vocab_sha256=audit['tokenizer_vocab_sha256'],
                    source_dialogs_sha256=audit['source_dialogs_sha256'],
                    splits_sha256=audit['splits_sha256'], prepared_dialogs_sha256=audit['dialogs_sha256'],
                    dtype='bf16', attention='sdpa', padding='right to128; one whole dialog per forward',
                    torch=str(torch.__version__), transformers=transformers.__version__,
                    cuda=torch.version.cuda, device=torch.cuda.get_device_name(),
                    capability=list(torch.cuda.get_device_capability()),
                    kernel_packages=packages, cudnn=torch.backends.cudnn.version(),
                    tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
                    script_sha256=file_hash(Path(__file__)), shard_count=args.shard_count)
    with torch.inference_mode():
        for row in shard_rows:
            hidden = encode(model, row)[0, :row['tokens']].cpu().clone()
            if hidden.dtype != torch.bfloat16 or hidden.shape != (row['tokens'], DIM) or not torch.isfinite(hidden).all():
                raise ValueError(f'invalid_features:{row["key"]}')
            torch.save(dict(hidden=hidden, ids_sha256=row['ids_sha256'], identity=identity,
                            parent_ids_sha256=row['parent_ids_sha256'],
                            messages_tools_sha256=row['messages_tools_sha256']), folder / (row['key'] + '.pt'))
    if weights != frozen_hash(model):
        raise ValueError('frozen_weights_changed')
    write(folder / f'complete{args.shard}.json', dict(identity, shard=args.shard,
          dialogs_sha256=audit['dialogs_sha256'], count=len(shard_rows)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('prepare', 'cache'))
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--splits', type=Path, default=SERVER / 'runs/autoresearch-sixhour-v1/data/splits.json')
    parser.add_argument('--snapshot', type=Path, default=SNAPSHOT)
    parser.add_argument('--shard-count', type=int, default=1)
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--authorize-gpu-cache', action='store_true')
    args = parser.parse_args()
    if args.shard_count < 1 or not 0 <= args.shard < args.shard_count:
        parser.error('require shard-count >= 1 and 0 <= shard < shard-count')
    {'prepare': prepare, 'cache': cache}[args.phase](args)
