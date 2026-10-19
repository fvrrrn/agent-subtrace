"""Explicit-phase Kev/Mamba training. Importing or preparing never starts training."""
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import random
import time

from qwen_mamba_pilot import BASE, REVISION, parameter_hash, sha, write
from graph_safety.step_training import CHECKPOINT, QUESTION, encode, load
from graph_safety.metrics import evaluate

ARCHES = ('mamba', 'mamba_lora', 'mamba2_lora', 'kev', 'linear_head', 'transformer_head')
OBJECTIVES = ('exponential', 'cosine_rank', 'noul_rank', 'token_hard', 'endpoint_hard',
              'linear', 'quadratic', 'sqrt', 'cosine_plain')
FIELDS = ('tool_interfaces', 'user_request', 'observed_history', 'proposed_action')


def read(path):
    return json.loads(Path(path).read_text())


def serial(value):
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def action_in_event(action, event):
    """Exact structural match; only an observed result may be absent in a proposal."""
    if not isinstance(event, dict):
        return False
    if event.get('role') in ('agent', 'assistant') and event.get('action') == action:
        return True
    if isinstance(action, dict) and action:
        visible = {k: v for k, v in event.items() if k not in ('tool_result',)}
        return visible == action
    return False


def link_rows(rows):
    """Verify exact prefix continuation, not trajectory IDs or coincident labels."""
    index = defaultdict(list)
    states = {r['key']: json.loads(r['state']) for r in rows}
    for r in rows:
        s = states[r['key']]
        if isinstance(s['observed_history'], list):
            index[(r['split'], r['group'], fingerprint([s[k] for k in FIELDS[:3]]))].append(r)
    ordered = sorted(rows, key=lambda r: len(states[r['key']]['observed_history']))
    audit = Counter()
    for r in ordered:
        s = states[r['key']]
        r['previous_annotated'] = r['previous_unsafe'] = None
        r['curve_start'] = 0.0  # stipulated initial target, NOT a safe label for history
        r['cosine_start'] = 0.0
        history = s['observed_history']
        if not isinstance(history, list):
            audit['opaque_history_no_pairs'] += 1
            continue
        ancestors = []
        for pos, event in enumerate(history):
            key = (r['split'], r['group'], fingerprint([
                s['tool_interfaces'], s['user_request'], history[:pos]]))
            matches = [p for p in index.get(key, ())
                       if action_in_event(states[p['key']]['proposed_action'], event)]
            if len(matches) == 1:
                ancestors.append(matches[0])
            elif matches:
                audit['ambiguous_history_positions'] += 1
        if ancestors:
            r['previous_annotated'] = ancestors[-1]['key']
            p = ancestors[-1]
            r['curve_start'] = .2 * p['curve_start'] + .8 * p['label']
            r['cosine_start'] = float(p['label'])
        unsafe = [p for p in ancestors if p['label'] == 1]
        if r['label'] == 1 and unsafe:
            r['previous_unsafe'] = unsafe[-1]['key']
            audit[r['split'] + '_unsafe_pairs'] += 1
    return dict(audit)


def prepare(args):
    from transformers import AutoTokenizer
    root = args.root
    root.mkdir(parents=True, exist_ok=False)
    rows = read(args.records)
    if len({r['key'] for r in rows}) != len(rows):
        raise ValueError('Duplicate record keys')
    if any(r['split'] not in ('train', 'validation') or r['label'] not in (0, 1) for r in rows):
        raise ValueError('Expected only binary training/validation records, no benchmarks')
    groups = defaultdict(set)
    for r in rows:
        groups[r['group']].add(r['split'])
        if set(r.get('sources', ())) & {'ToolSafe', 'StepGuard', 'R-Judge', 'ATBench'}:
            raise ValueError('Protected benchmark source in training data')
        s = json.loads(r['state'])
        if set(s) != set(FIELDS):
            raise ValueError('Unexpected model-visible fields')
    if any(len(s) != 1 for s in groups.values()):
        raise ValueError('Group crosses train/validation')
    input_splits = defaultdict(set)
    for r in rows:
        input_splits[fingerprint(json.loads(r['state']))].add(r['split'])
    if any(len(s) != 1 for s in input_splits.values()):
        raise ValueError('Input crosses train/validation')
    pair_audit = link_rows(rows)
    tok = AutoTokenizer.from_pretrained(BASE, revision=REVISION, local_files_only=True)
    prepared, rejected = [], []
    for r in rows:
        s = json.loads(r['state'])
        # Fixed segment tokenization keeps the supervised action span exact. No chat-template change.
        prefix = (QUESTION['instructions'] + '\n[TOOL INTERFACES]\n' + serial(s['tool_interfaces'])
                  + '\n[USER REQUEST]\n' + serial(s['user_request'])
                  + '\n[OBSERVED HISTORY]\n' + serial(s['observed_history']) + '\n[AGENT ACTION]\n')
        ids = tok.encode(prefix, add_special_tokens=False)
        start = len(ids)
        ids += tok.encode(serial(s['proposed_action']), add_special_tokens=False)
        limit = args.train_limit if r['split'] == 'train' else args.eval_limit
        if len(ids) == start or len(ids) > limit:
            rejected.append({'key': r['key'], 'split': r['split'], 'label': r['label'],
                             'tokens': len(ids), 'reason': 'empty_action' if len(ids) == start else 'context_overflow'})
            continue
        prepared.append(dict(r, ids=ids, start=start, tokens=len(ids),
                             text_sha256=hashlib.sha256((prefix + serial(s['proposed_action'])).encode()).hexdigest(),
                             ids_sha256=fingerprint(ids)))
    keys = {r['key'] for r in prepared}
    for r in prepared:
        if r['previous_unsafe'] not in keys:
            r['previous_unsafe'] = None
    write(root / 'records.json', prepared)
    write(root / 'rejections.json', rejected)
    write(root / 'protocol.json', {
        'base': BASE, 'revision': REVISION, 'kev': CHECKPOINT, 'seed': args.seed,
        'source_records_sha256': sha(args.records), 'records_sha256': sha(root / 'records.json'),
        'source_sha256': sha(__file__), 'train_limit': args.train_limit, 'eval_limit': args.eval_limit,
        'counts': dict(Counter(r['split'] for r in prepared)), 'pair_audit_before_admission': pair_audit,
        'admitted_pairs': dict(Counter(r['split'] for r in prepared if r['previous_unsafe'])),
        'rejected': len(rejected), 'rendering': 'Explicit tools/user/history/action segments; no labels or outcomes added; no truncation.',
        'curves': {'exponential': 'y + (a-y)*exp(-ln(5)*u); a from verified previous annotated endpoints, initially zero',
                   'cosine_rank': 'a + (y-a)*(1-cos(pi*u))/2; a = previous annotated hard label, initially zero'},
        'ranking': '0.1 * mean relu(0.2 + early_raw_logodds - late_raw_logodds), verified unsafe action endpoints only; independent uniform pair batches',
        'kev_objective': 'Native hard-label NOUL CE + unsafe endpoint ranking. No token-curve supervision on the Kev pointer head.',
        'selection': 'Minimum hard-label endpoint validation BCE. No benchmark-based checkpoint selection.',
        'limitations': ['Artificial curves depend on action labels and full action lengths, not causal token-level gold.',
                       'Missing labels remain unknown. Initial curve anchor zero is a stipulated training target, not evidence of safety.',
                       'Unsafe temporal ranking is a prior, not verified growth of severity.',
                       'Each example replays its complete causal prefix; recurrent state resets between independent forwards.',
                       'The input renderer differs from native Kev; compare systems, not identical representations.',
                       'Code preparation only; no training or evaluation automatically scheduled.']})
    print(json.dumps(read(root / 'protocol.json'), indent=2))


def configuration(root):
    cfg = read(root / 'protocol.json')
    if sha(root / 'records.json') != cfg['records_sha256'] or sha(__file__) != cfg['source_sha256']:
        raise ValueError('Prepared data/code changed; prepare a separate version')
    return cfg


def backbone(args, trainable=False):
    import torch
    from transformers import Qwen3_5TextModel
    model = Qwen3_5TextModel.from_pretrained(BASE, revision=REVISION, local_files_only=True,
        dtype=torch.bfloat16, attn_implementation=args.attention).cuda()
    model.requires_grad_(False)
    if trainable:
        from peft import LoraConfig, get_peft_model
        # Includes projections of both full-attention and Gated DeltaNet blocks, no LM vocabulary head.
        model = get_peft_model(model, LoraConfig(r=args.lora_rank, lora_alpha=2 * args.lora_rank,
                                               lora_dropout=0., target_modules='all-linear', bias='none'))
        if args.checkpointing:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    return model


def make_head(layers):
    import torch.nn as nn
    from mamba_ssm import Mamba2

    class Head(nn.Module):
        def __init__(self):
            super().__init__()
            self.project = nn.Sequential(nn.LayerNorm(1024), nn.Linear(1024, 256))
            self.blocks = nn.ModuleList([Mamba2(d_model=256, d_state=64, expand=2, headdim=64,
                chunk_size=128, use_mem_eff_path=True) for _ in range(layers)])
            self.readout = nn.Sequential(nn.LayerNorm(256), nn.Linear(256, 1))

        def forward(self, x):
            x = self.project(x)
            for block in self.blocks:
                x = x + block(x)
            return self.readout(x).squeeze(-1)

    return Head().cuda()


def make_transformer_head():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = nn.LayerNorm(256)
            self.qkv = nn.Linear(256, 768)
            self.out = nn.Linear(256, 256)
            self.ff = nn.Sequential(nn.LayerNorm(256), nn.Linear(256, 256),
                                    nn.GELU(), nn.Linear(256, 256))

        def forward(self, x):
            b, n, _ = x.shape
            q, k, v = self.qkv(self.norm(x)).reshape(b, n, 3, 4, 64).unbind(2)
            z = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2),
                                               v.transpose(1, 2), is_causal=True)
            x = x + self.out(z.transpose(1, 2).reshape(b, n, 256))
            return x + self.ff(x)

    class Head(nn.Module):
        def __init__(self):
            super().__init__()
            self.input_norm = nn.LayerNorm(1024)
            self.project = nn.Linear(1024, 256)
            self.blocks = nn.Sequential(Block())
            self.output_norm = nn.LayerNorm(256)
            self.readout = nn.Linear(256, 1)

        def forward(self, x):
            x = self.project(self.input_norm(x))
            positions = torch.arange(x.shape[1], device=x.device, dtype=torch.float32)
            frequencies = torch.exp(torch.arange(0, 256, 2, device=x.device).float()
                                    * (-math.log(10000) / 256))
            angles = positions[:, None] * frequencies[None, :]
            encoding = torch.stack((angles.sin(), angles.cos()), dim=-1).flatten(1)
            x = self.blocks(x + encoding.to(x.dtype))
            return self.readout(self.output_norm(x)).squeeze(-1)

    return Head().cuda()


def make_linear_head():
    import torch.nn as nn

    class Head(nn.Module):
        def __init__(self):
            super().__init__()
            self.readout = nn.Linear(1024, 1)

        def forward(self, x):
            return self.readout(x).squeeze(-1)

    return Head().cuda()


def cache(args):
    import torch
    from tqdm.auto import tqdm
    cfg = configuration(args.root)
    folder = args.root / 'features'
    folder.mkdir(exist_ok=False)
    model = backbone(args).eval()
    before = parameter_hash(model)
    rows = read(args.root / 'records.json')
    required = sum(r['tokens'] for r in rows) * 1024 * 2
    write(folder / 'estimate.json', {'bf16_hidden_bytes': required, 'note': 'Disk cache, not all loaded into RAM.'})
    for group in tqdm(batches(rows, args.batch_size, args.token_budget), desc='Frozen Qwen features'):
        n = math.ceil(max(r['tokens'] for r in group) / 128) * 128
        ids = torch.zeros(len(group), n, dtype=torch.long, device='cuda')
        mask = torch.zeros_like(ids)
        for i, r in enumerate(group):
            ids[i, :r['tokens']] = torch.tensor(r['ids'], device='cuda')
            mask[i, :r['tokens']] = 1
        with torch.inference_mode():
            h = model(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state
        if not torch.isfinite(h).all():
            raise ValueError('Nonfinite frozen features')
        for i, r in enumerate(group):
            torch.save({'hidden': h[i, :r['tokens']].cpu().clone(), 'ids_sha256': r['ids_sha256']},
                       folder / (hashlib.sha256(r['key'].encode()).hexdigest() + '.pt'))
    if parameter_hash(model) != before:
        raise ValueError('Frozen backbone changed')
    write(folder / 'complete.json', {'records_sha256': cfg['records_sha256'], 'base_sha256': before,
                                    'attention': args.attention, 'rows': len(rows)})


def batches(rows, size, token_budget, seed=None):
    """Length buckets limit right padding without packing unrelated recurrent states."""
    order = sorted(rows, key=lambda r: r['tokens'])
    chunks = [order[i:i + 128] for i in range(0, len(order), 128)]
    if seed is not None:
        rng = random.Random(seed)
        rng.shuffle(chunks)
    result = []
    for chunk in chunks:
        batch = []
        for r in chunk:
            padded = math.ceil(r['tokens'] / 128) * 128
            if batch and (len(batch) >= size or padded * (len(batch) + 1) > token_budget):
                result.append(batch)
                batch = []
            batch.append(r)
        if batch:
            result.append(batch)
    if seed is not None:
        rng.shuffle(result)
    return result


def token_forward(args, model, head, rows):
    import torch
    n = math.ceil(max(r['tokens'] for r in rows) / 128) * 128
    if model is None:
        x = torch.zeros(len(rows), n, 1024, dtype=torch.bfloat16, device='cuda')
        for i, r in enumerate(rows):
            name = hashlib.sha256(r['key'].encode()).hexdigest() + '.pt'
            entry = torch.load(args.root / 'features' / name, weights_only=True, map_location='cpu')
            if entry['ids_sha256'] != r['ids_sha256']:
                raise ValueError('Feature/token mismatch')
            x[i, :r['tokens']] = entry['hidden'].to('cuda')
    else:
        ids = torch.zeros(len(rows), n, dtype=torch.long, device='cuda')
        mask = torch.zeros_like(ids)
        for i, r in enumerate(rows):
            ids[i, :r['tokens']] = torch.tensor(r['ids'], device='cuda')
            mask[i, :r['tokens']] = 1
        x = model(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state
    with torch.autocast('cuda', dtype=torch.bfloat16):
        return head(x)


def targets(row, objective, device):
    import torch
    u = torch.arange(1, row['tokens'] - row['start'] + 1, device=device, dtype=torch.float32)
    u /= len(u)
    if objective == 'token_hard':
        return torch.full_like(u, float(row['label']))
    a = row['curve_start']
    if objective == 'exponential':
        fraction = 1 - torch.exp(-math.log(5) * u)
    else:
        a = row['cosine_start']
        if objective == 'linear':
            fraction = u
        elif objective == 'quadratic':
            fraction = u.square()
        elif objective == 'sqrt':
            fraction = u.sqrt()
        else:
            fraction = (1 - torch.cos(math.pi * u)) / 2
    return a + (row['label'] - a) * fraction


def native_setup(args, trainable):
    from kev.api import SystemOneRequest, to_record
    from graph_safety.step_training import request
    cp, tok, model = load(CHECKPOINT)
    _, mapping = to_record(SystemOneRequest.model_validate(request('mapping')))
    if mapping[0]['keys'] != ['false', 'true']:
        raise ValueError('Unexpected native NOUL mapping')
    for name, p in model.named_parameters():
        p.requires_grad_(trainable and ('lora_' in name or name.startswith('head.')))
    if trainable:
        for config in model.lm.peft_config.values():
            config.inference_mode = False
        if args.checkpointing:
            model.lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    return cp, tok, model


def native_admission(args, tok, model, rows):
    from kev.model import ContextOverflow
    encs, rejected = {}, []
    cfg = configuration(args.root)
    for r in rows:
        try:
            limit = cfg['train_limit'] if r['split'] == 'train' else cfg['eval_limit']
            encs[r['key']], _ = encode(tok, model, r['state'], limit)
        except ContextOverflow as error:
            rejected.append({'key': r['key'], 'label': r['label'], 'reason': str(error)})
    return encs, rejected


def native_logits(model, enc):
    import torch
    with torch.autocast('cuda', dtype=torch.bfloat16):
        z = model(enc)[0].float()
    if not model.training:
        z = z * model.head.temperature  # training loss/ranking uses raw native logits
    return z


def native_logodds(model, enc):
    z = native_logits(model, enc)
    return z[1] - z[0]


def predict(args, model, head, rows, folder, name, encs=None):
    import torch
    import torch.nn.functional as F
    from tqdm.auto import tqdm
    if model is not None:
        model.eval()
    if head is not None:
        head.eval()
    predictions, total = [], 0.
    groups = [[r] for r in rows] if encs is not None else batches(rows, args.batch_size, args.token_budget)
    with torch.no_grad(), (folder / (name + '.jsonl')).open('w') as out:
        for group in tqdm(groups, desc=name):
            if encs is None:
                all_z = token_forward(args, model, head, group)
                z = torch.stack([all_z[i, r['tokens'] - 1].float() for i, r in enumerate(group)])
                scores = z.sigmoid()
            else:
                native_z = native_logits(model, encs[group[0]['key']])
                z = (native_z[1] - native_z[0])[None]
                scores = (z / model.head.temperature).sigmoid()
            y = torch.tensor([r['label'] for r in group], device='cuda', dtype=torch.float32)
            total += F.binary_cross_entropy_with_logits(z, y, reduction='sum').item()
            for i, r in enumerate(group):
                p = {'key': r['key'], 'label': r['label'], 'score': scores[i].item(),
                     'raw_logodds': z[i].item(), 'state_sha256': hashlib.sha256(r['state'].encode()).hexdigest(),
                     'ids_sha256': r['ids_sha256'], 'tokens': r['tokens']}
                if encs is not None:
                    p['native_raw_logits'] = native_z.cpu().tolist()
                    p['temperature'] = model.head.temperature
                predictions.append(p)
                out.write(json.dumps(p) + '\n')
    metrics = evaluate([r['label'] for r in predictions], [r['score'] for r in predictions])
    metrics.pop('pr_curve')
    metrics['endpoint_bce'] = total / len(predictions)
    write(folder / (name + '.json'), metrics)
    return metrics


def train(args):
    import torch
    import torch.nn.functional as F
    import transformers
    from tqdm.auto import tqdm
    cfg = configuration(args.root)
    if (args.arch == 'kev') != (args.objective == 'noul_rank'):
        raise ValueError('Kev uses noul_rank; Mamba uses exponential or cosine_rank')
    if args.objective in ('token_hard', 'endpoint_hard', 'linear', 'quadratic', 'sqrt', 'cosine_plain') and args.arch not in ('mamba', 'linear_head', 'transformer_head'):
        raise ValueError('Token-target extensions require a frozen Mamba or direct linear head')
    folder = args.root / 'arms' / (args.arch + '_' + args.objective)
    folder.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(cfg['seed'])
    random.seed(cfg['seed'])
    rows = read(args.root / 'records.json')
    by_key = {r['key']: r for r in rows}
    native = args.arch == 'kev'
    encs, rejected, cp, tok = None, [], None, None
    model, head = None, None
    if native:
        cp, tok, model = native_setup(args, True)
        encs, rejected = native_admission(args, tok, model, rows)
        rows = [r for r in rows if r['key'] in encs]
    else:
        if args.arch in ('mamba', 'linear_head', 'transformer_head'):
            cached = read(args.root / 'features/complete.json')
            if cached['records_sha256'] != cfg['records_sha256']:
                raise ValueError('Wrong feature cache')
        else:
            model = backbone(args, True)
        # Separate RNG stream makes same-depth heads identical with/without a LoRA backbone.
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            torch.manual_seed(cfg['seed'])
            if args.arch == 'transformer_head':
                head = make_transformer_head()
            else:
                head = make_linear_head() if args.arch == 'linear_head' else make_head(2 if args.arch == 'mamba2_lora' else 1)
    training = [r for r in rows if r['split'] == 'train']
    validation = [r for r in rows if r['split'] == 'validation']
    admitted = {r['key'] for r in rows}
    pairs = [r for r in training if r['previous_unsafe'] in admitted]
    ranking = args.objective in ('cosine_rank', 'noul_rank')
    if not training or not validation or (ranking and not pairs):
        raise ValueError('Need train/validation and verified admitted unsafe pairs for ranking')
    modules = [m for m in (model, head) if m is not None]
    params = [p for m in modules for p in m.parameters() if p.requires_grad]
    groups = []
    if model is not None:
        groups.append({'params': [p for p in model.parameters() if p.requires_grad], 'lr': args.lora_lr})
    if head is not None:
        groups.append({'params': list(head.parameters()), 'lr': args.head_lr})
    optimizer = torch.optim.AdamW(groups, weight_decay=.01, fused=True)
    frozen_before = frozen_hash(model) if model is not None else None
    write(folder / 'run.json', {'args': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        'protocol_sha256': sha(args.root / 'protocol.json'), 'torch': torch.__version__,
        'transformers': transformers.__version__, 'device': torch.cuda.get_device_name(),
        'trainable_parameters': sum(p.numel() for p in params), 'unsafe_pairs': len(pairs),
        'rejections': rejected, 'initial_head_sha256': parameter_hash(head) if head is not None else None,
        'optimizer': 'fused AdamW; BF16 base + FP32 Mamba/head under BF16 autocast; native Kev inherited fp32 + bf16 autocast',
        'head_training_scope': ('Hard endpoint BCE only' if args.objective == 'endpoint_hard' else
                                'Token BCE in Mamba, hard native endpoint NOUL CE in Kev'),
        'temperature': model.head.temperature if native else None})
    best, history = float('inf'), []
    started = time.monotonic()
    for epoch in range(1, args.epochs + 1):
        for m in modules:
            m.train()
        order = ([[r] for r in random.Random(cfg['seed'] + epoch).sample(training, len(training))]
                 if native else batches(training, args.batch_size, args.token_budget, cfg['seed'] + epoch))
        rng = random.Random(cfg['seed'] + epoch + 100000)
        optimizer.zero_grad(set_to_none=True)
        ce_sum = rank_sum = 0.
        count = 0
        for step, group in enumerate(tqdm(order, desc=f'{folder.name} epoch {epoch}')):
            window_start = step // args.accumulation * args.accumulation
            window_groups = order[window_start:window_start + args.accumulation]
            window = len(window_groups)
            window_actions = sum(len(g) for g in window_groups)
            if native:
                z = native_logodds(model, encs[group[0]['key']])
                ce = F.binary_cross_entropy_with_logits(z, z.new_tensor(float(group[0]['label'])))
            else:
                z = token_forward(args, model, head, group)
                if args.objective == 'endpoint_hard':
                    ce = torch.stack([F.binary_cross_entropy_with_logits(
                        z[i, r['tokens'] - 1].float(), z.new_tensor(float(r['label']), dtype=torch.float32))
                        for i, r in enumerate(group)]).mean()
                else:
                    ce = torch.stack([F.binary_cross_entropy_with_logits(
                        z[i, r['start']:r['tokens']].float(), targets(r, args.objective, 'cuda'))
                        for i, r in enumerate(group)]).mean()
            # Separate backward frees the main-batch graph before encoding sampled ranking pairs.
            (ce * len(group) / window_actions).backward()
            rank_value = 0.
            if ranking:
                chosen = rng.choices(pairs, k=args.pair_batch_size)
                if native:
                    ranks = []
                    for late in chosen:
                        early_z = native_logodds(model, encs[late['previous_unsafe']])
                        late_z = native_logodds(model, encs[late['key']])
                        ranks.append(F.relu(.2 + early_z - late_z))
                    rank = torch.stack(ranks).mean()
                else:
                    unique = {r['key']: r for late in chosen for r in (by_key[late['previous_unsafe']], late)}
                    endpoints = {}
                    for pair_group in batches(list(unique.values()), args.batch_size, args.token_budget):
                        zz = token_forward(args, model, head, pair_group)
                        endpoints.update({r['key']: zz[i, r['tokens'] - 1].float() for i, r in enumerate(pair_group)})
                    rank = torch.stack([F.relu(.2 + endpoints[r['previous_unsafe']] - endpoints[r['key']]) for r in chosen]).mean()
                (.1 * rank / window).backward()
                rank_value = rank.detach().item()
            ce_sum += ce.detach().item() * len(group)
            rank_sum += rank_value
            count += len(group)
            if (step + 1) % args.accumulation == 0 or step + 1 == len(order):
                if epoch == 1 and step + 1 == window:
                    gradients = {n: p.grad.detach().float().norm().item()
                                 for m in modules for n, p in m.named_parameters()
                                 if p.requires_grad and p.grad is not None}
                    if not any(v > 0 for v in gradients.values()):
                        raise ValueError('No nonzero training gradients')
                    if head is not None and args.arch != 'linear_head' and not any(v > 0 for n, v in gradients.items() if 'blocks.' in n):
                        raise ValueError('Mamba has no gradient')
                    if model is not None and not any(v > 0 for n, v in gradients.items() if 'lora_' in n):
                        raise ValueError('LoRA has no gradient')
                    write(folder / 'gradient_audit.json', gradients)
                torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        metrics = predict(args, model, head, validation, folder, f'validation_epoch{epoch}', encs)
        history.append({'epoch': epoch, 'curve_or_noul_bce': ce_sum / count,
                        'sampled_rank': rank_sum / len(order), 'validation': metrics,
                        'seconds_since_start': time.monotonic() - started})
        write(folder / 'history.json', history)
        if metrics['endpoint_bce'] < best:
            best = metrics['endpoint_bce']
            if native:
                from kev.checkpoint import write_meta
                checkpoint = folder / 'checkpoint'
                model.lm.save_pretrained(checkpoint)
                tok.save_pretrained(checkpoint)
                meta = deepcopy(cp.meta)
                meta.head = {k: v.detach().cpu() for k, v in model.head.state_dict().items()}
                meta.extra = {'parent': CHECKPOINT, 'experiment': 'action-models-noul-rank',
                              'selected_epoch': epoch, 'protocol_sha256': sha(args.root / 'protocol.json'),
                              'temperature_note': 'Inherited unchanged; not recalibrated'}
                write_meta(checkpoint, meta)
            else:
                torch.save(head.state_dict(), folder / 'selected_head.pt')
                if model is not None:
                    model.save_pretrained(folder / 'adapter')
            write(folder / 'selection.json', {'epoch': epoch, 'validation_bce': best,
                'head_sha256': sha(folder / 'selected_head.pt') if not native else None,
                'criterion': 'hard endpoint BCE, no benchmark opened'})
    frozen_after = frozen_hash(model) if model is not None else None
    if frozen_before != frozen_after:
        raise ValueError('Frozen parameters changed')
    write(folder / 'complete.json', {'seconds': time.monotonic() - started, 'epochs': args.epochs,
                                    'frozen_before': frozen_before, 'frozen_after': frozen_after})


def frozen_hash(model):
    import torch
    h = hashlib.sha256()
    for name, p in model.named_parameters():
        if not p.requires_grad:
            h.update(name.encode())
            h.update(p.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def compare(args):
    configuration(args.root)
    paths = list((args.root / 'baselines').glob('*.jsonl'))
    for folder in sorted((args.root / 'arms').glob('*')):
        if not (folder / 'complete.json').exists():
            raise ValueError(f'Incomplete arm: {folder}')
        epoch = read(folder / 'selection.json')['epoch']
        paths.append(folder / f'validation_epoch{epoch}.jsonl')
    if len(paths) < 3:
        raise ValueError('Need both baselines and at least one trained arm')
    outputs = {str(p.relative_to(args.root)): {r['key']: r for r in
               (json.loads(s) for s in p.read_text().splitlines())} for p in paths}
    common = sorted(set.intersection(*(set(v) for v in outputs.values())))
    if not common:
        raise ValueError('No common evaluation keys')
    for key in common:
        if len({(v[key]['label'], v[key]['state_sha256']) for v in outputs.values()}) != 1:
            raise ValueError('Input or label mismatch')
    metrics = {}
    for name, result in outputs.items():
        m = evaluate([result[k]['label'] for k in common], [result[k]['score'] for k in common])
        m.pop('pr_curve')
        metrics[name] = m
    write(args.root / 'comparison.json', {'validation_common_n': len(common), 'metrics': metrics,
        'excluded_by_arm': {name: sorted(set(result) - set(common)) for name, result in outputs.items()},
        'limitations': 'Validation selected checkpoints; not an independent test. Random Mamba/head is an untrained network, not IID random scores. Native Kev renderer differs.'})


def baselines(args):
    import torch
    cfg = configuration(args.root)
    folder = args.root / 'baselines'
    folder.mkdir(exist_ok=False)
    rows = [r for r in read(args.root / 'records.json') if r['split'] == 'validation']
    cached = read(args.root / 'features/complete.json')
    if cached['records_sha256'] != cfg['records_sha256']:
        raise ValueError('Wrong feature cache')
    torch.manual_seed(cfg['seed'])
    head = make_head(1)
    torch.save(head.state_dict(), folder / 'random_mamba_head.pt')
    predict(args, None, head, rows, folder, 'random_mamba_head')
    del head
    torch.cuda.empty_cache()
    _, tok, model = native_setup(args, False)
    encs, rejected = native_admission(args, tok, model, rows)
    write(folder / 'kev_rejections.json', rejected)
    predict(args, model, None, [r for r in rows if r['key'] in encs], folder, 'kev_original', encs)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=('prepare', 'cache', 'train', 'baselines', 'compare'))
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--records', type=Path)
    p.add_argument('--arch', choices=ARCHES, default='mamba')
    p.add_argument('--objective', choices=OBJECTIVES, default='exponential')
    p.add_argument('--seed', type=int, default=20261017)
    p.add_argument('--epochs', type=int, default=8)
    p.add_argument('--train-limit', type=int, default=8192)
    p.add_argument('--eval-limit', type=int, default=65536)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--token-budget', type=int, default=16384)
    p.add_argument('--accumulation', type=int, default=4)
    p.add_argument('--pair-batch-size', type=int, default=1)
    p.add_argument('--lora-rank', type=int, default=16)
    p.add_argument('--head-lr', type=float, default=3e-4)
    p.add_argument('--lora-lr', type=float, default=1e-5)
    p.add_argument('--attention', choices=('sdpa', 'flash_attention_2'), default='sdpa')
    p.add_argument('--checkpointing', action=argparse.BooleanOptionalAction, default=False)
    args = p.parse_args()
    for name in ('epochs', 'batch_size', 'token_budget', 'accumulation', 'pair_batch_size', 'lora_rank', 'train_limit', 'eval_limit'):
        if getattr(args, name) <= 0:
            p.error(name + ' must be positive')
    if args.phase == 'prepare':
        if args.records is None:
            p.error('prepare requires --records')
        prepare(args)
    else:
        {'cache': cache, 'train': train, 'baselines': baselines, 'compare': compare}[args.phase](args)


if __name__ == '__main__':
    main()
