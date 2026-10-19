"""Offline pinned three-layer cache on precisely the admitted Instruct dialogs."""
import inspect
import json
from pathlib import Path
import time
import torch
import transformers
from transformers import Qwen3_5TextModel
from tqdm import tqdm
import train_whole_dialogs as original
from prepare_whole_dialogs import write, digest
from train_streaming_weighted_multilayer_v1 import RUN, OLD, DEADLINE, canonical
from train_sound_anomaly_heads import environment

INDICES = (11, 22, 32)


def main():
    assert time.time() < DEADLINE
    audit = original.read(OLD / 'data/data_audit.json')
    assert audit['launch_allowed'] and not audit['exclusion_reasons']
    assert original.file_hash(OLD / 'data/dialogs.json') == audit['dialogs_sha256']
    assert original.file_hash(OLD / 'data/splits.json') == audit['splits_sha256']
    rows = original.read(OLD / 'data/dialogs.json')
    assert len(rows) == 2310
    weights = original.read(OLD / 'model_weights.json')
    snapshot = Path(weights['snapshot'])
    configuration = original.read(snapshot / 'config.json')['text_config']
    assert configuration['num_hidden_layers'] == 32 and configuration['hidden_size'] == 2560
    for name, expected in tqdm(weights['sha256'].items(), desc='Verify local pinned weights', unit='shard'):
        assert original.file_hash(snapshot / name) == expected
    model = Qwen3_5TextModel.from_pretrained(str(snapshot), local_files_only=True,
        dtype=torch.bfloat16, attn_implementation='sdpa').cuda().eval()
    model.requires_grad_(False)
    frozen_hash = original.frozen_hash(model)
    reference = original.read(OLD / 'cache/features/complete0.json')
    assert frozen_hash == reference['weights_sha256']
    folder = RUN / 'cache/features'
    folder.mkdir(parents=True, exist_ok=False)
    probe = min((r for r in rows if r['spans'][0]['end'] < r['tokens']), key=lambda r: r['tokens'])
    captured = {}
    handles = []
    for index in INDICES:
        def capture(module, inputs, output, index=index):
            captured[index] = output
        handles.append(model.layers[index-1].register_forward_hook(capture))
    with torch.inference_mode():
        ids, mask = original.tensor_input(probe)
        out = model(input_ids=ids, attention_mask=mask, use_cache=False, output_hidden_states=True)
        assert len(out.hidden_states) == 33
        for index in INDICES[:-1]:
            assert torch.equal(out.hidden_states[index], captured[index]), f'Wrong hidden-state index {index}'
        assert torch.equal(out.hidden_states[32], out.last_hidden_state)
        assert torch.equal(model.norm(captured[32]), out.last_hidden_state)
        before = torch.cat([out.hidden_states[index] for index in INDICES], dim=-1)
        cut = probe['spans'][0]['end']
        changed = ids.clone()
        changed[:, cut:probe['tokens']] = 0
        altered = model(input_ids=changed, attention_mask=mask, use_cache=False, output_hidden_states=True)
        after = torch.cat([altered.hidden_states[index] for index in INDICES], dim=-1)
        assert torch.equal(before[:, :cut], after[:, :cut])
    for handle in handles:
        handle.remove()
    del ids, mask, out, altered, before, after, changed, captured
    identity = canonical(dict(reference))
    for key in ('count', 'shard'):
        identity.pop(key)
    identity.update(hidden_layer='hidden_states[11,22,32]', hidden_indices=list(INDICES),
        hidden_shape=['tokens', 7680], environment=canonical(environment()),
        hidden_semantics='decoder outputs 11 and22 before final norm; layer32 after native final norm',
        cache_source_sha256=original.file_hash(Path(__file__)),
        transformers_model_source_sha256=original.file_hash(Path(inspect.getfile(Qwen3_5TextModel))),
        backbone_causal_error=0, hidden_index_runtime_gate=True)
    assert identity == json.loads(json.dumps(identity))
    write(RUN / 'cache/backbone_gate.json', {'status': 'passed', 'identity': identity,
        'causal_error': 0, 'hidden_index_hooks_match': True, 'probe': probe['key']})
    started = time.monotonic()
    with torch.inference_mode():
        for row in tqdm(rows, desc='Three-layer frozen cache', unit='dialog'):
            assert time.time() < DEADLINE
            assert len(row['ids']) == row['tokens'] <= 16384
            assert row['ids_sha256'] == digest(row['ids'])
            ids, mask = original.tensor_input(row)
            out = model(input_ids=ids, attention_mask=mask, use_cache=False, output_hidden_states=True)
            hidden = torch.cat([out.hidden_states[index][0, :row['tokens']] for index in INDICES], dim=-1).cpu()
            assert hidden.shape == (row['tokens'], 7680) and hidden.dtype == torch.bfloat16
            assert torch.isfinite(hidden).all()
            with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
                old = torch.load(OLD / 'cache/features' / (row['key'] + '.pt'), map_location='cpu', weights_only=True)
            assert old['ids_sha256'] == row['ids_sha256']
            assert canonical(old['identity']) == {k: v for k, v in reference.items() if k not in ('count', 'shard')}
            assert torch.equal(hidden[:, -2560:], old['hidden']), 'Deep cache does not exactly match last-layer control'
            torch.save({'hidden': hidden, 'identity': identity, 'ids_sha256': row['ids_sha256']}, folder / (row['key'] + '.pt'))
            del ids, mask, out, hidden, old
    assert original.frozen_hash(model) == frozen_hash
    write(folder / 'complete0.json', dict(identity, count=2310, shard=0))
    write(RUN / 'cache/cost.json', {'seconds': time.monotonic()-started,
        'tokens': sum(r['tokens'] for r in rows), 'deep_exact_match_count': 2310,
        'frozen_weights_verified_after': True})


if __name__ == '__main__':
    main()
