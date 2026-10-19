"""Native fixed-weight Kev-27B check on the same interpreter-labeled teacher controls."""
import hashlib
import argparse
import json
from pathlib import Path
import time

ROOT=Path('/var/local/danefedov/experiments/agent-safety')


def main(split):
    import torch
    from kev.checkpoint import Checkpoint,LoadOptions
    from graph_safety.controlled_training import predict,model_environment
    from graph_safety.io import write_json
    data=json.loads((ROOT/'archive/legacy-shm/danefedov-controlled-monotonic-verified-v1/results/records.json').read_text())
    rows=[]
    for label in (0,1):
        rows.extend(sorted((r for r in data if r['split']==split and r['label']==label),
            key=lambda r:hashlib.sha256(('teacher-control:'+r['key']).encode()).hexdigest())[:64])
    assert len(rows)==128
    folder=ROOT/'datasets/corpus-v1/annotations'/('oracle_controls_kev27'+('_validation' if split=='validation' else ''));folder.mkdir(exist_ok=False)
    started=time.monotonic();torch.backends.cuda.matmul.allow_tf32=False
    cp=Checkpoint('jaredpalmer/kev-27b@af0e6d551bdc2cc724f3e9d7a8bee1cd4fb8f7bf')
    tok,model=cp.load('cuda',LoadOptions(dtype=torch.bfloat16,merge=True,attn='sdpa',fused=True,cuda_graphs=False,backend='torch'))
    write_json(folder/'environment.json',model_environment(cp,model))
    predictions,result=predict(model,tok,rows,{'state_limit':8192,'eval_state_limit':8192},folder,'controls')
    complete=[r for r in predictions if r['status']=='complete']
    summary={'attempts':128,'complete':len(complete),'correct':sum((r['score']>=.5)==r['label'] for r in complete),
        'metrics':result['metrics'],'seconds':time.monotonic()-started,
        'note':'Original fixed Kev-27B, native NOUL and shipped temperature, bf16 fused kernels. No freely generated rationale, no state truncation or model training.'}
    write_json(folder/'summary.json',summary);print(json.dumps(summary,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--split',choices=('train','validation'),default='train');a=p.parse_args();main(a.split)
