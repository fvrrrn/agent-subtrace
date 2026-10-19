"""Native alternate-backbone preparation and separately authorized frozen GPU cache."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import prepare_autoresearch_prompt as parent
from prepare_whole_dialogs import BASE as PARENT_BASE, REVISION as PARENT_REVISION, digest, write

HUB = Path('/var/local/danefedov/experiments/agent-safety/models/huggingface/hub')
BACKBONES = {
    '0.8b': ('Qwen/Qwen3.5-0.8B-Base', 'dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68', 1024),
    '9b': ('Qwen/Qwen3.5-9B-Base', '68c46c4b3498877f3ef123c856ecfde50c39f404', 4096),
}
COUNTS = dict(zip(parent.SPLITS, (2009, 301, 299, 425)))


def make_head(hidden_size, arm='curve_linear_softmax', attention=False):
    """Parent trainer opt-in: seed before calling; replace only CPU input modules."""
    import torch
    from train_curve_event_heads import Head
    head = Head(arm, attention)
    head.input_norm = torch.nn.LayerNorm(hidden_size)
    head.projection = torch.nn.Linear(hidden_size, 256)
    return head


def manifest(path):
    source = parent.read(path)
    if set(source) not in (set(parent.SPLITS), set(parent.SPLITS) | {'train_full'}):
        raise ValueError('unexpected split manifest schema')
    selected = {name: source[name] for name in parent.SPLITS}
    if {name: len(keys) for name, keys in selected.items()} != COUNTS:
        raise ValueError('fixed split counts changed')
    lookup = {}
    for split, keys in selected.items():
        for key in keys:
            if not isinstance(key, str) or key in lookup:
                raise ValueError('invalid or overlapping split key')
            lookup[key] = split
    return selected, lookup


def snapshot_identity(args):
    config = parent.read(args.snapshot / 'config.json')
    if config['text_config']['hidden_size'] != args.hidden:
        raise ValueError('unexpected hidden size')
    index = parent.read(args.snapshot / 'model.safetensors.index.json')
    shards = sorted(set(index['weight_map'].values()))
    if not all((args.snapshot / name).is_file() for name in shards):
        raise ValueError('incomplete local weights')
    files = ['config.json', 'tokenizer_config.json', 'tokenizer.json',
             'merges.txt', 'vocab.json', 'model.safetensors.index.json']
    return {'base': args.base, 'revision': args.revision, 'hidden_size': args.hidden,
            'snapshot_files_sha256': {name: parent.file_hash(args.snapshot / name) for name in files},
            'weight_shards': {name: (args.snapshot / name).stat().st_size for name in shards}}


def prepare_row(old_tok, tok, row, split):
    messages, tools = row['messages'], row['tools']
    old_text, old_ids, old_offsets = parent.render(old_tok, messages, tools)
    if old_ids != row['ids'] or digest(old_ids) != row['ids_sha256'] or len(old_ids) != row['tokens']:
        raise ValueError('parent_native_input_mismatch')
    endpoints = {}
    wanted = {(span['start'], span['end']) for span in row['spans']}
    for stop, message in enumerate(messages, 1):
        if message['role'] != 'assistant':
            continue
        prefix = old_tok.apply_chat_template(messages[:stop], tools=tools, tokenize=False,
                                              add_generation_prompt=False)
        try:
            start, end, prefix_hash = parent.action_span(old_tok, prefix, old_text, old_ids, old_offsets)
        except ValueError:
            continue
        if (start, end) in wanted:
            if (start, end) in endpoints:
                raise ValueError('ambiguous_parent_action_endpoint')
            endpoints[start, end] = (stop, prefix_hash)
    if set(endpoints) != wanted or len(wanted) != len(row['spans']):
        raise ValueError('unrecoverable_parent_action_endpoints')
    text, ids, offsets = parent.render(tok, messages, tools)
    spans = []
    for old in row['spans']:
        stop, parent_hash = endpoints[old['start'], old['end']]
        prefix = tok.apply_chat_template(messages[:stop], tools=tools, tokenize=False,
                                         add_generation_prompt=False)
        start, end, prefix_hash = parent.action_span(tok, prefix, text, ids, offsets)
        spans.append(dict(old, start=start, end=end, message_stop=stop,
                          parent_start=old['start'], parent_end=old['end'],
                          parent_prefix_ids_sha256=parent_hash, prefix_ids_sha256=prefix_hash))
    if any(a['end'] > b['start'] for a, b in zip(spans, spans[1:])):
        raise ValueError('overlapping_action_spans')
    if len(ids) > parent.LIMIT:
        raise ValueError(f'whole_dialog_context_overflow:{len(ids)}')
    # Preserve every original key/group/label/actor/tool field; no instruction is injected.
    return dict(row, split=split, parent_split=row['split'], ids=ids, tokens=len(ids),
                ids_sha256=digest(ids), spans=spans, parent_ids_sha256=row['ids_sha256'],
                parent_tokens=row['tokens'], messages_tools_sha256=digest([messages, tools]),
                parent_row_sha256=digest(row))


def prepare(args):
    if args.shard_count != 1 or args.shard != 0:
        raise ValueError('prepare is unsharded')
    from transformers import AutoTokenizer
    mapping, lookup = manifest(args.splits)
    identity = snapshot_identity(args)
    source = args.data / 'dialogs.json' if args.data.is_dir() else args.data
    rows = parent.read(source)
    index = {row['key']: row for row in rows}
    if len(index) != len(rows) or not set(lookup) <= index.keys():
        raise ValueError('duplicate or missing parent keys')
    old_tok = AutoTokenizer.from_pretrained(PARENT_BASE, revision=PARENT_REVISION, local_files_only=True)
    tok = AutoTokenizer.from_pretrained(str(args.snapshot), local_files_only=True)
    if not old_tok.is_fast or not tok.is_fast or not isinstance(tok.chat_template, str):
        raise ValueError('native fast tokenizer/template unavailable')
    if hashlib.sha256(old_tok.chat_template.encode()).hexdigest() != parent.TEMPLATE_SHA256:
        raise ValueError('parent_template_identity_changed')
    args.root.mkdir(parents=True, exist_ok=False)
    accepted, rejected = [], []
    for key, split in lookup.items():
        try:
            accepted.append(prepare_row(old_tok, tok, index[key], split))
        except (ValueError, TypeError) as error:
            rejected.append({'key': key, 'split': split, 'parent_row_sha256': digest(index[key]),
                             'reason': str(error)})
    write(args.root / 'dialogs.json', accepted)
    write(args.root / 'rejections.json', rejected)
    write(args.root / 'splits.json', mapping)
    audit = dict(identity, template_sha256=hashlib.sha256(tok.chat_template.encode()).hexdigest(),
                 tokenizer_vocab_sha256=digest(tok.get_vocab()),
                 source_dialogs_sha256=parent.file_hash(source),
                 splits_sha256=parent.file_hash(args.splits), selected_splits_sha256=digest(mapping),
                 dialogs_sha256=parent.file_hash(args.root / 'dialogs.json'),
                 requested_counts=COUNTS,
                 admitted_counts=dict(Counter(row['split'] for row in accepted)),
                 exclusion_reasons=dict(Counter(row['reason'] for row in rejected)),
                 launch_allowed=not rejected, limit=parent.LIMIT, no_truncation=True,
                 dialogs=len(accepted), actions=sum(len(row['spans']) for row in accepted),
                 tokens=sum(row['tokens'] for row in accepted),
                 max_tokens=max((row['tokens'] for row in accepted), default=0),
                 parent_state_sha256=digest([index[key] for key in lookup]),
                 action_group_state_sha256=digest([
                     {k: v for k, v in row.items() if k not in ('ids', 'tokens', 'spans')}
                     for row in accepted]),
                 protocol='native messages/tools; all original endpoints independently reconstructed; '
                          'all labeled new causal prefixes equal full-dialog IDs through action end; '
                          'unlabeled context has no loss; no state reset inside a dialog',
                 exclusions_policy='any exclusion blocks cache/training pending matched comparison protocol')
    write(args.root / 'data_audit.json', audit)
    print(json.dumps(audit, indent=2), flush=True)


def cache(args):
    if not args.authorized_gpu_cache:
        raise ValueError('cache requires explicit main-session GPU authorization')
    mapping, lookup = manifest(args.splits)
    audit = parent.read(args.data / 'data_audit.json')
    if not audit['launch_allowed'] or audit['exclusion_reasons']:
        raise ValueError('exclusions require a new matched comparison protocol; no cache launch')
    if parent.file_hash(args.data / 'dialogs.json') != audit['dialogs_sha256']:
        raise ValueError('prepared input changed')
    if audit['splits_sha256'] != parent.file_hash(args.splits) or audit['selected_splits_sha256'] != digest(mapping):
        raise ValueError('split identity changed')
    for key, value in snapshot_identity(args).items():
        if audit[key] != value:
            raise ValueError(f'snapshot identity changed:{key}')
    rows = parent.read(args.data / 'dialogs.json')
    if len(rows) != len(lookup) or {row['key'] for row in rows} != set(lookup):
        raise ValueError('prepared population mismatch')
    for row in rows:
        if (row['split'] != lookup[row['key']] or row['tokens'] != len(row['ids']) or
                row['tokens'] > parent.LIMIT or digest(row['ids']) != row['ids_sha256']):
            raise ValueError('prepared row integrity mismatch')
    folder = args.root / 'features'
    folder.mkdir(parents=True, exist_ok=True)
    shard_rows = rows[args.shard::args.shard_count]
    if (folder / f'complete{args.shard}.json').exists() or any((folder / (r['key']+'.pt')).exists() for r in shard_rows):
        raise ValueError('refusing to overwrite shard')
    import torch
    import transformers
    from transformers import Qwen3_5TextModel
    from train_whole_dialogs import encode, frozen_hash
    from train_autoresearch_heads import environment
    model = Qwen3_5TextModel.from_pretrained(str(args.snapshot), local_files_only=True,
                dtype=torch.bfloat16, attn_implementation='sdpa').cuda().eval()
    model.requires_grad_(False)
    weights = frozen_hash(model)
    runtime = environment()
    visible_devices = runtime.pop('CUDA_VISIBLE_DEVICES')
    identity = dict(audit, weights_sha256=weights, dtype='bf16', attention='sdpa',
                    hidden_layer='last_hidden_state (final normalized text layer)',
                    hidden_shape=['tokens', args.hidden], torch=str(torch.__version__),
                    transformers=transformers.__version__, environment=runtime,
                    padding='right to128; one full dialog per forward', shard_count=args.shard_count,
                    cache_source_sha256=parent.file_hash(Path(__file__)),
                    encoder_source_sha256=parent.file_hash(Path(__file__).with_name('train_whole_dialogs.py')))
    with torch.inference_mode():
        for row in shard_rows:
            hidden = encode(model, row)[0, :row['tokens']].cpu().clone()
            if hidden.dtype != torch.bfloat16 or hidden.shape != (row['tokens'], args.hidden) or not torch.isfinite(hidden).all():
                raise ValueError(f'invalid features:{row["key"]}')
            torch.save({'hidden': hidden, 'ids_sha256': row['ids_sha256'], 'identity': identity,
                        'parent_ids_sha256': row['parent_ids_sha256'],
                        'parent_row_sha256': row['parent_row_sha256']}, folder / (row['key']+'.pt'))
    if weights != frozen_hash(model):
        raise ValueError('frozen weights changed')
    write(folder / f'runtime{args.shard}.json', {'CUDA_VISIBLE_DEVICES': visible_devices})
    write(folder / f'complete{args.shard}.json', dict(identity, shard=args.shard, count=len(shard_rows)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('prepare', 'cache'))
    parser.add_argument('--backbone', choices=BACKBONES, default='0.8b')
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--splits', type=Path, default=parent.DEFAULT_SPLITS)
    parser.add_argument('--shard-count', type=int, default=1)
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--authorized-gpu-cache', action='store_true')
    args = parser.parse_args()
    args.base, args.revision, args.hidden = BACKBONES[args.backbone]
    args.snapshot = HUB / ('models--' + args.base.replace('/', '--')) / 'snapshots' / args.revision
    if args.shard_count < 1 or not 0 <= args.shard < args.shard_count:
        parser.error('require shard-count >=1 and 0 <= shard < shard-count')
    {'prepare': prepare, 'cache': cache}[args.phase](args)
