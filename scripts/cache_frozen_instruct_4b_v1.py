"""New frozen native Instruct cache; no Base feature reuse."""
from pathlib import Path
import time

from huggingface_hub import snapshot_download
import torch
import transformers
from transformers import Qwen3_5TextModel

from prepare_frozen_instruct_4b_v1 import RUN, MODEL, REVISION, DEADLINE
from prepare_autoresearch_backbone import HUB
from prepare_whole_dialogs import write, digest
import train_whole_dialogs as original
from train_sound_anomaly_heads import environment


def main():
    audit = original.read(RUN / 'data/data_audit.json')
    assert audit['launch_allowed'] and not audit['exclusion_reasons']
    assert time.time() < DEADLINE
    assert original.file_hash(RUN / 'data/dialogs.json') == audit['dialogs_sha256']
    rows = original.read(RUN / 'data/dialogs.json')
    assert len(rows) == 2310
    folder = RUN / 'cache/features'
    folder.mkdir(parents=True, exist_ok=False)
    snapshot = Path(snapshot_download(MODEL, revision=REVISION, cache_dir=HUB,
                    allow_patterns=['*.safetensors', '*.json', '*.txt', 'README.md', 'LICENSE']))
    index = original.read(snapshot / 'model.safetensors.index.json')
    tokenizer_hashes = {name: original.file_hash(snapshot / name) for name in
                        ('config.json', 'tokenizer_config.json', 'tokenizer.json', 'merges.txt', 'vocab.json')}
    for name in ('config.json', 'tokenizer_config.json'):
        assert tokenizer_hashes[name] == audit['source_files_sha256'][name]
    shard_hashes = {name: original.file_hash(snapshot / name)
                    for name in sorted(set(index['weight_map'].values()))}
    write(RUN / 'model_weights.json', {'base': MODEL, 'revision': REVISION,
          'snapshot': str(snapshot), 'sha256': shard_hashes})
    assert time.time() < DEADLINE
    model = Qwen3_5TextModel.from_pretrained(str(snapshot), local_files_only=True,
                dtype=torch.bfloat16, attn_implementation='sdpa').cuda().eval()
    model.requires_grad_(False)
    weights = original.frozen_hash(model)
    probe = min((r for r in rows if r['spans'][0]['end'] < r['tokens']), key=lambda r: r['tokens'])
    cut = probe['spans'][0]['end']
    with torch.inference_mode():
        ids, mask = original.tensor_input(probe)
        before = model(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state
        changed = ids.clone()
        changed[:, cut:probe['tokens']] = 0
        after = model(input_ids=changed, attention_mask=mask, use_cache=False).last_hidden_state
        causal_error = (before[:, :cut] - after[:, :cut]).abs().max().item()
        assert causal_error == 0, f'Backbone causal error: {causal_error}'
    del ids, mask, before, after, changed
    identity = dict(audit, weights_sha256=weights, weight_files_sha256=shard_hashes,
                    tokenizer_files_sha256=tokenizer_hashes,
                    dtype='bf16', attention='sdpa', hidden_layer='last_hidden_state',
                    hidden_shape=['tokens', 2560], torch=str(torch.__version__),
                    transformers=transformers.__version__, environment=environment(),
                    padding='right to128; one complete dialog per forward',
                    cache_source_sha256=original.file_hash(Path(__file__)),
                    encoder_source_sha256=original.file_hash(Path(original.__file__)),
                    backbone_causal_error=causal_error)
    write(RUN / 'cache/backbone_gate.json', {'status': 'passed', 'causal_error': causal_error,
          'probe': probe['key'], 'cut': cut, 'weights_sha256': weights, 'environment': environment()})
    started = time.monotonic()
    with torch.inference_mode():
        for i, row in enumerate(rows):
            assert time.time() < DEADLINE, 'Cache deadline reached'
            assert len(row['ids']) == row['tokens'] <= 16384 and row['ids_sha256'] == digest(row['ids'])
            hidden = original.encode(model, row)[0, :row['tokens']].cpu().clone()
            assert hidden.dtype == torch.bfloat16 and hidden.shape == (row['tokens'], 2560)
            assert torch.isfinite(hidden).all()
            torch.save({'hidden': hidden, 'ids_sha256': row['ids_sha256'], 'identity': identity},
                       folder / (row['key'] + '.pt'))
            if (i + 1) % 50 == 0:
                print({'cached': i + 1, 'total': len(rows), 'seconds': time.monotonic() - started}, flush=True)
    assert weights == original.frozen_hash(model)
    write(folder / 'complete0.json', dict(identity, shard=0, count=len(rows)))
    write(RUN / 'cache/cost.json', {'seconds': time.monotonic() - started,
          'tokens': sum(r['tokens'] for r in rows), 'frozen_weights_verified_after': True})


if __name__ == '__main__':
    main()
