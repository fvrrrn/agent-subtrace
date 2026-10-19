"""Evaluate the original Kev on exactly the admitted training-experiment validation set."""
import argparse
from pathlib import Path
import time

import torch
from tqdm.auto import tqdm
from kev.api import SystemOneRequest, to_record
from kev.predictors import kernel_environment
from graph_safety.step_training import load, encode, request
from graph_safety.io import read_json, write_json, sha256
from graph_safety.metrics import evaluate

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--reference', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
a.output.mkdir(exist_ok=False)
cfg = read_json(a.reference/'protocol.json')
if sha256(a.reference/'records.json') != cfg['records_sha256']:
    raise ValueError('Reference data changed')
epoch = read_json(a.reference/'selected_checkpoint.json')['epoch']
expected = {r['key']:r for r in read_json(a.reference/f'epoch_{epoch}.json')['validation']}
rows = [r for r in read_json(a.reference/'records.json') if r['key'] in expected]
if len(rows) != len(expected) or any(r['split'] != 'validation' or r['label'] != expected[r['key']]['label'] for r in rows):
    raise ValueError('Validation mismatch')
torch.manual_seed(cfg['seed'])
cp, tok, model = load(cfg['checkpoint'])
env = kernel_environment(model, 'cuda')
reference_env = read_json(a.reference/'training_environment.json')
if env != reference_env['environment'] or model.head.temperature != reference_env['temperature']:
    raise ValueError('Evaluation environment mismatch')
if cp.weights_sha256() != reference_env['parent_weights_sha256'] or sha256(cp.file('head.pt')) != reference_env['parent_head_sha256']:
    raise ValueError('Original weights mismatch')
_, mapping = to_record(SystemOneRequest.model_validate(request('mapping')))
if mapping[0]['keys'] != ['false', 'true']: raise ValueError('NOUL mapping changed')
predictions = []
started = time.perf_counter()
with torch.inference_mode():
    for row in tqdm(rows, desc='Original Kev validation', mininterval=5):
        enc, _ = encode(tok, model, row['state'], cfg['train_state_limit'])
        with torch.autocast('cuda', dtype=torch.bfloat16):
            z = model(enc)[0].float()
        if not torch.isfinite(z).all(): raise ValueError('Nonfinite logits')
        predictions.append({'key':row['key'], 'label':row['label'], 'logits':z.cpu().tolist(),
                            'score':torch.softmax(z,-1)[1].item()})
write_json(a.output/'predictions.json', predictions)
write_json(a.output/'summary.json', {'checkpoint':cfg['checkpoint'], 'environment':env,
    'temperature':model.head.temperature, 'seconds':time.perf_counter()-started,
    'reference_protocol_sha256':sha256(a.reference/'protocol.json'),
    'metrics':evaluate([r['label'] for r in predictions],[r['score'] for r in predictions]),
    'mean_score':{str(label):sum(r['score'] for r in predictions if r['label']==label)/sum(r['label']==label for r in predictions) for label in (0,1)}})
