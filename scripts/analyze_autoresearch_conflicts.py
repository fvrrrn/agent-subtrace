"""CPU-only loss-gradient diagnostic on saved round1 dev_select logits."""
import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import statistics

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '1'

import torch
import torch.nn.functional as F


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def object_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def native_function(path, name):
    # Execute only the requested pure function, never module imports/main/CUDA setup.
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    scope = {'torch': torch, 'F': F, 'math': math}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), scope)
    return scope[name], hashlib.sha256(ast.get_source_segment(path.read_text(), node).encode()).hexdigest()


def mean(values):
    return statistics.mean(values) if values else None


def describe(rows):
    active = sum(r['both_nonzero_tokens'] for r in rows)
    tokens = sum(r['length'] for r in rows)
    opposed = sum(r['opposed_tokens'] for r in rows)
    token_sq = sum((r['token_grad_norm']/r['dialog_actions'])**2 for r in rows)
    event_sq = sum((r['event_grad_norm']/r['dialog_actions'])**2 for r in rows)
    null_sq = sum((r['null_grad']/r['dialog_actions'])**2 for r in rows)
    dot = sum(r['gradient_dot']/r['dialog_actions']**2 for r in rows)
    return {
        'events': len(rows), 'tokens': tokens,
        'opposed_fraction_all_tokens': opposed/tokens if tokens else None,
        'opposed_fraction_both_nonzero': opposed/active if active else None,
        'both_nonzero_fraction': active/tokens if tokens else None,
        'events_with_any_opposition': sum(r['opposed_tokens'] > 0 for r in rows),
        'events_negative_cosine': sum(r['cosine'] is not None and r['cosine'] < 0 for r in rows),
        'mean_event_cosine': mean([r['cosine'] for r in rows if r['cosine'] is not None]),
        'median_event_cosine': statistics.median([r['cosine'] for r in rows if r['cosine'] is not None]) if any(r['cosine'] is not None for r in rows) else None,
        'mean_event_opposed_fraction': mean([r['opposed_fraction_all'] for r in rows]),
        'mean_weighted_norm_ratio': mean([r['weighted_event_to_token_norm'] for r in rows if r['weighted_event_to_token_norm'] is not None]),
        'median_weighted_norm_ratio': statistics.median([r['weighted_event_to_token_norm'] for r in rows if r['weighted_event_to_token_norm'] is not None]) if rows else None,
        'concatenated_dialog_reduced_cosine': dot/math.sqrt(token_sq*event_sq) if token_sq*event_sq > 0 else None,
        'concatenated_dialog_reduced_weighted_norm_ratio': .1*math.sqrt(event_sq/token_sq) if token_sq else None,
        'concatenated_dialog_reduced_weighted_joint_event_to_token_norm': .1*math.sqrt((event_sq+null_sq)/token_sq) if token_sq else None,
        'weighted_null_gradient_norm_dialog_reduced': .1*math.sqrt(null_sq),
        'combined_gradient_reversal_fraction': sum(r['combined_reversal_tokens'] for r in rows)/tokens if tokens else None,
        'mean_token_bce': mean([r['token_bce'] for r in rows]),
        'mean_event_bce': mean([r['event_bce'] for r in rows]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--round-root', type=Path, required=True)
    parser.add_argument('--native-source', type=Path, required=True)
    parser.add_argument('--output-prefix', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    event_fn, event_hash = native_function(args.native_source/'train_autoresearch_heads.py', 'event_logit')
    profile_fn, profile_hash = native_function(args.native_source/'train_curve_event_heads.py', 'profile')
    prefix = args.output_prefix
    prefix.parent.mkdir(parents=True, exist_ok=True)
    paths = {ext: Path(str(prefix)+ext) for ext in ('.jsonl', '.json', '.md')}
    if any(p.exists() for p in paths.values()):
        raise FileExistsError('Use a new output prefix; preserve prior diagnostics')
    arms = {'broad_endpoint': 'endpoint', 'broad_linear_softmax': 'linear_softmax',
            'broad_null': 'null', 'broad_leaky': 'leaky'}
    groups, manifests, identities = {}, {}, {}
    with paths['.jsonl'].open('x') as stream:
        for arm, readout in arms.items():
            folder = args.round_root/arm/'fit'/'dev_select_epoch1'/'tokens'
            files = sorted(folder.glob('*.pt'))
            if not files:
                raise ValueError(f'No tensors: {folder}')
            records, manifest, event_ids = [], [], {}
            for file in files:
                saved = torch.load(file, map_location='cpu', weights_only=True)
                z_all = saved['raw_token_logits'].detach().float().flatten()
                null_all = saved['null_logits']
                spans = saved['spans']
                file_hash = sha(file)
                assert z_all.numel() == saved['tokens'] == len(saved['ids'])
                assert torch.isfinite(z_all).all()
                manifest.append({'path': str(file), 'sha256': file_hash,
                                 'dialog': saved['dialog'], 'ids_sha256': saved['ids_sha256'],
                                 'spans_sha256': object_sha(spans), 'events': len(spans)})
                for index, span in enumerate(spans):
                    start, end, label = span['start'], span['end'], span['label']
                    assert 0 <= start < end <= z_all.numel() and label in (0, 1)
                    z = z_all[start:end].clone().requires_grad_(True)
                    null = null_all.detach().float().flatten()[start:end].clone().requires_grad_(True) if null_all is not None else None
                    local_span = {'start': 0, 'end': len(z), 'label': label}
                    target = float(label)*profile_fn(len(z), 'cpu', 'broad')
                    token_loss = F.binary_cross_entropy_with_logits(z, target)
                    event_logit = event_fn(z, null, local_span, readout)
                    event_loss = F.binary_cross_entropy_with_logits(event_logit, event_logit.new_tensor(float(label)))
                    gt, = torch.autograd.grad(token_loss, z)
                    gradients = torch.autograd.grad(event_loss, (z, null) if readout == 'null' else (z,))
                    ge = gradients[0]
                    gn = gradients[1][-1].item() if readout == 'null' else 0.
                    assert torch.isfinite(gt).all() and torch.isfinite(ge).all() and math.isfinite(gn)
                    # Compute products/norms in float64 to avoid tiny-gradient underflow.
                    gt, ge = gt.double(), ge.double()
                    tn, en = gt.norm().item(), ge.norm().item()
                    dot = torch.dot(gt, ge).item()
                    both = (gt != 0) & (ge != 0)
                    opposed = (gt*ge < 0)
                    identity = {'dialog': saved['dialog'], 'span_index': index,
                                'start': start, 'end': end, 'label': label,
                                'event_key': span.get('key'), 'ids_sha256': saved['ids_sha256'],
                                'span_sha256': object_sha(span),
                                'action_ids_sha256': object_sha(saved['ids'][start:end])}
                    event_id = object_sha(identity)
                    assert event_id not in event_ids
                    event_ids[event_id] = identity
                    row = {**identity, 'event_id': event_id, 'arm': arm, 'readout': readout,
                           'tensor_path': str(file), 'tensor_sha256': file_hash,
                           'length': len(z), 'dialog_actions': len(spans),
                           'token_bce': token_loss.item(), 'event_bce': event_loss.item(),
                           'event_logit': event_logit.item(), 'token_grad_norm': tn,
                           'event_grad_norm': en, 'null_grad': gn, 'weighted_null_grad': .1*gn,
                           'gradient_dot': dot, 'cosine': dot/(tn*en) if tn*en else None,
                           'weighted_event_to_token_norm': .1*en/tn if tn else None,
                           'both_nonzero_tokens': int(both.sum()), 'opposed_tokens': int(opposed.sum()),
                           'opposed_fraction_all': opposed.double().mean().item(),
                           'opposed_fraction_both_nonzero': int(opposed.sum())/int(both.sum()) if both.any() else None,
                           'combined_reversal_tokens': int((gt*(gt+.1*ge) < 0).sum())}
                    stream.write(json.dumps(row, allow_nan=False)+'\n')
                    records.append(row)
            groups[arm] = {name: describe([r for r in records if label is None or r['label'] == label])
                           for name, label in (('all', None), ('safe', 0), ('unsafe', 1))}
            manifests[arm] = manifest
            identities[arm] = event_ids
            print(arm, len(files), len(records), flush=True)
    reference = identities['broad_endpoint']
    assert all(value == reference for value in identities.values()), 'Arms not exact matched events'
    result = {'scope': 'round1 dev_select epoch1 only; no confirm inputs read',
              'event_weight': .1, 'curve': '(.9+1.2u-1.5u²)/1.14; u=i/N, i=1..N',
              'native_source': str(args.native_source),
              'native_source_sha256': {name: sha(args.native_source/name) for name in
                                       ('train_autoresearch_heads.py', 'train_curve_event_heads.py')},
              'extracted_function_sha256': {'event_logit': event_hash, 'profile': profile_hash},
              'analyzer_sha256': sha(Path(__file__)), 'torch_version': str(torch.__version__),
              'CUDA_VISIBLE_DEVICES': os.environ['CUDA_VISIBLE_DEVICES'], 'threads': torch.get_num_threads(),
              'matched_events': len(reference), 'matched_event_identity_sha256': object_sha(reference),
              'per_event_jsonl_sha256': sha(paths['.jsonl']), 'summary': groups, 'tensor_manifest': manifests,
              'limitations': [
                  'Detached-output gradients, not parameter/shared-Mamba gradients or training-step gradients.',
                  'Each arm evaluated on its own epoch1 logits; between-arm differences include learned trajectories.',
                  'Per-event norms use native within-action token mean; dialog-reduced aggregates divide by action count.',
                  'Zero gradients excluded from active-coordinate opposition; all-coordinate rates dilute endpoint conflicts.',
                  'Autograd is float32 native readout/BCE on CPU; summary norms/products float64.',
                  'NULL branch is separate; its larger dimension-level contribution does not identify shared-parameter conflict.',
                  'Validation forward was saved before report failure; no confirmation data, thresholds or external labels used.',
                  'Labels specify the two loss objectives only; targets are artificial, not token gold.',
                  'Not causal evidence localization, calibrated risk, or proof event-weight 1 improves generalization.']}
    paths['.json'].write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    lines = ['# Round1 epoch1 dev_select loss-gradient conflicts', '',
             f'CPU-only; {len(reference)} exact-matched events per arm. Event weight 0.1. No confirmation data read.', '',
             '| Arm | Label | Events | Opposed/all tokens | Opposed/active | Mean cosine | Median .1 event/token norm | Dialog-reduced .1 event/token norm |',
             '|---|---|---:|---:|---:|---:|---:|---:|']
    def fmt(v):
        return 'NA' if v is None else f'{v:.5g}'
    for arm, labels in groups.items():
        for label, row in labels.items():
            lines.append('| '+' | '.join([arm, label, str(row['events']), fmt(row['opposed_fraction_all_tokens']),
                         fmt(row['opposed_fraction_both_nonzero']), fmt(row['mean_event_cosine']),
                         fmt(row['median_weighted_norm_ratio']), fmt(row['concatenated_dialog_reduced_weighted_norm_ratio'])])+' |')
    lines += ['', '## Interpretation safeguards', '']+[f'- {text}' for text in result['limitations']]
    lines += ['', 'NULL endpoint-branch gradients (signed, weighted and unweighted) are recorded separately per event.',
              'Full summaries include NULL joint norms, negative-cosine counts, combined-gradient reversals, exact event identities and source/tensor hashes.',
              'A weight-1 control multiplies the event term by ten relative to this report; effects on learned parameters cannot be inferred by rescaling detached logits.',
              'Existing prespecified control is not selected or altered by this diagnostic. Raw source tensors are read-only.']
    paths['.md'].write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    main()
