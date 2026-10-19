"""Small grouped StepGuard pilot: frozen Qwen token features, native Mamba-2 head."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import random
import time


BASE = 'Qwen/Qwen3.5-0.8B-Base'
REVISION = 'dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68'
SEED = 20261018


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2))


def prepare(root, reference):
    root.mkdir(parents=True, exist_ok=False)
    rows = json.loads((reference/'records.json').read_text())
    groups = defaultdict(list)
    for r in rows:
        if r['split'] == 'external_test':
            groups[r['group']].append(r)
    ordered = sorted(groups, key=lambda k: hashlib.sha256(f'{SEED}:{k}'.encode()).hexdigest())
    selected, cursor = [], 0
    for split, target in [('train', 256), ('validation', 64), ('test', 128)]:
        count = 0
        while count < target:
            key = ordered[cursor]; cursor += 1
            for row in groups[key]:
                row = dict(row, original_split=row['split'], split=split)
                selected.append(row); count += 1
    assert len({r['key'] for r in selected}) == len(selected)
    assert all(len({r['split'] for r in selected if r['group'] == g}) == 1 for g in {r['group'] for r in selected})
    baseline = {r['key']:r for r in map(json.loads, (reference/'full/baseline_external_test.jsonl').read_text().splitlines())}
    test = [r for r in selected if r['split'] == 'test']
    assert all(baseline[r['key']]['label'] == r['label'] and baseline[r['key']]['status'] == 'complete' for r in test)
    write(root/'records.json', selected)
    write(root/'kev_reference.json', [baseline[r['key']] for r in test])
    write(root/'protocol.json', {
        'base': BASE, 'revision': REVISION, 'seed': SEED,
        'source_sha256': sha(__file__), 'records_sha256': sha(root/'records.json'),
        'reference_records_sha256': sha(reference/'records.json'),
        'kev_reference_sha256': sha(root/'kev_reference.json'),
        'counts':dict(Counter(r['split'] for r in selected)),
        'class_counts':dict(Counter(f'{r["split"]}:{r["label"]}' for r in selected)),
        'epochs':8, 'lr':0.0003, 'weight_decay':0.01, 'batch_size':4,
        'max_tokens':8192, 'truncation':False,
        'architecture':'Frozen original Qwen3.5-0.8B-Base text backbone, bf16 SDPA; cached last-layer features for EVERY token. Trainable LayerNorm + 1024->256 projection + one residual Mamba2(d_state=64, expand=2, headdim=64, chunk_size=128, use_mem_eff_path=True) + LayerNorm + binary linear readout at EVERY token.',
        'training':'PyTorch fused AdamW, bf16 autocast, final-token BCE only in this pilot. No Kev, LoRA, temperature, curve labels or monotonic loss. All prefix tokens receive gradients through Mamba. State resets between independent examples; whole prefix recomputed for each example. Right padding is excluded at readout.',
        'split':'New fixed hash partition by existing normalized-request group from StepGuard external_test. Whole groups retained; target counts may be exceeded. No sampling or split changes based on predictions.',
        'selection':'Minimum validation BCE over eight fixed epochs; freeze checkpoint before test. Report random-head initialization and trained head on same held-out keys, plus saved original Kev-0.8B outputs.',
        'limits':['StepGuard labels are action-level, not token-level. Only final-token quality is supervised and evaluated; token traces are not explanations.',
                  'The parent StepGuard pool was evaluated previously. This is an exploratory small-data test, not untouched confirmation.',
                  'Kev is zero-shot; Mamba head is trained in-domain. This compares systems, not architecture under equal supervision.',
                  'Frozen Qwen already contextualizes tokens; improvements do not prove Mamba encodes separate attacks.',
                  'No hyperparameter sweep, no assumption that training improves held-out metrics.']})


def parameter_hash(model):
    h = hashlib.sha256()
    for name, p in model.named_parameters():
        h.update(name.encode());h.update(p.detach().cpu().contiguous().view(__import__('torch').uint8).numpy().tobytes())
    return h.hexdigest()


def run(root):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import transformers
    import mamba_ssm
    from mamba_ssm import Mamba2
    from transformers import AutoTokenizer, Qwen3_5TextModel
    from tqdm.auto import tqdm
    from graph_safety.metrics import evaluate
    from graph_safety.step_training import QUESTION

    cfg = json.loads((root/'protocol.json').read_text())
    assert cfg['source_sha256'] == sha(__file__)
    assert cfg['records_sha256'] == sha(root/'records.json')
    assert cfg['kev_reference_sha256'] == sha(root/'kev_reference.json')
    torch.manual_seed(SEED);random.seed(SEED)
    torch.backends.cuda.matmul.allow_tf32 = False
    rows = json.loads((root/'records.json').read_text())
    cache = root/'features';cache.mkdir(exist_ok=False)
    tokenizer = AutoTokenizer.from_pretrained(BASE, revision=REVISION, local_files_only=True)
    backbone = Qwen3_5TextModel.from_pretrained(BASE, revision=REVISION,
        local_files_only=True, dtype=torch.bfloat16, attn_implementation='sdpa').cuda().eval()
    backbone.requires_grad_(False)
    frozen_before = parameter_hash(backbone)
    write(root/'environment.json', {'torch':torch.__version__, 'transformers':transformers.__version__,
        'mamba_ssm':mamba_ssm.__version__, 'cuda':torch.version.cuda,
        'device':torch.cuda.get_device_name(), 'base_parameters':sum(p.numel() for p in backbone.parameters()),
        'base_sha256':frozen_before, 'base_revision':REVISION})
    features, admitted, rejects = {}, [], []
    started = time.monotonic()
    for r in tqdm(rows, desc='Frozen Qwen / token cache', mininterval=5):
        # No labels or annotation metadata are rendered. JSON state contains only causal input.
        text = QUESTION['instructions'] + '\n\n' + r['state'] + '\n\nSafety of proposed action:'
        ids = tokenizer(text, return_tensors='pt', add_special_tokens=True)['input_ids'].cuda()
        if ids.shape[1] > cfg['max_tokens']:
            rejects.append({'key':r['key'], 'split':r['split'], 'label':r['label'], 'tokens':ids.shape[1], 'reason':'context_overflow'});continue
        with torch.inference_mode():
            hidden = backbone(input_ids=ids, use_cache=False).last_hidden_state[0]
        assert torch.isfinite(hidden).all()
        tensor = hidden.cpu().clone()
        name = hashlib.sha256(r['key'].encode()).hexdigest()
        torch.save({'input_ids':ids[0].cpu(), 'hidden':tensor,
            'input_sha256':hashlib.sha256(text.encode()).hexdigest()}, cache/f'{name}.pt')
        features[r['key']] = tensor
        admitted.append(dict(r, tokens=ids.shape[1], feature_file=f'{name}.pt'))
    assert parameter_hash(backbone) == frozen_before
    assert all(p.grad is None and not p.requires_grad for p in backbone.parameters())
    write(root/'encoder_audit.json', {'frozen_unchanged':True, 'elapsed_seconds':time.monotonic()-started,
        'counts':dict(Counter(r['split'] for r in admitted)), 'rejections':rejects,
        'max_tokens':max(r['tokens'] for r in admitted)})
    write(root/'admitted.json', admitted)
    del backbone, hidden, ids
    torch.cuda.empty_cache()

    class Head(nn.Module):
        def __init__(self):
            super().__init__()
            self.project = nn.Sequential(nn.LayerNorm(1024), nn.Linear(1024,256))
            self.mamba = Mamba2(d_model=256, d_state=64, expand=2, headdim=64,
                chunk_size=128, use_mem_eff_path=True)
            self.readout = nn.Sequential(nn.LayerNorm(256), nn.Linear(256,1))

        def forward(self, x):
            x = self.project(x)
            return self.readout(x+self.mamba(x)).squeeze(-1)

    torch.manual_seed(SEED)
    model = Head().cuda()
    initial = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    torch.save(initial, root/'initial_head.pt')
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'], fused=True)

    def batch(items):
        lengths = [r['tokens'] for r in items]
        n = ((max(lengths)+127)//128)*128
        x = torch.zeros(len(items), n, 1024, dtype=torch.bfloat16, device='cuda')
        for i,r in enumerate(items):x[i,:lengths[i]] = features[r['key']].to('cuda')
        return x, torch.tensor(lengths, device='cuda')-1, torch.tensor([r['label'] for r in items], dtype=torch.float32, device='cuda')

    train = [r for r in admitted if r['split']=='train']
    validation = [r for r in admitted if r['split']=='validation']
    test = [r for r in admitted if r['split']=='test']
    assert all({r['label'] for r in subset}=={0,1} for subset in (train,validation,test))

    # Operational audit: padding/future invariance and actual gradient through the CUDA scan.
    x,ends,y = batch(train[:2]);model.eval()
    with torch.autocast('cuda', dtype=torch.bfloat16):
        output = model(x)
    loss = F.binary_cross_entropy_with_logits(output[torch.arange(len(y),device='cuda'),ends].float(),y)
    loss.backward()
    grad = sum(p.grad.detach().float().square().sum().item() for p in model.mamba.parameters() if p.grad is not None)**.5
    assert grad > 0 and all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        altered=x.clone();cut=min(64,train[0]['tokens']//2)
        altered[:,cut:]=torch.randn_like(altered[:,cut:])
        changed=model(altered)
    causal_error=(output[:,:cut]-changed[:,:cut]).abs().max().item()
    assert causal_error < .02
    optimizer.zero_grad(set_to_none=True)
    write(root/'pilot_gate.json', {'status':'passed', 'mamba_gradient_norm':grad,
        'future_perturbation_max_logit_error':causal_error,
        'trainable_parameters':sum(p.numel() for p in model.parameters()),
        'frozen_encoder_verified':True, 'native_cuda_forward_backward':True})
    del x,altered,output,changed,loss

    def predict(subset, name, save_tokens=False):
        model.eval();predictions=[];total=0.
        token_file=(root/f'{name}_tokens.jsonl').open('w') if save_tokens else None
        with torch.no_grad():
            for offset in tqdm(range(0,len(subset),cfg['batch_size']),desc=name,mininterval=5):
                items=subset[offset:offset+cfg['batch_size']];x,ends,y=batch(items)
                with torch.autocast('cuda',dtype=torch.bfloat16):z=model(x)
                logits=z[torch.arange(len(items),device='cuda'),ends].float()
                total+=F.binary_cross_entropy_with_logits(logits,y,reduction='sum').item()
                for i,r in enumerate(items):
                    predictions.append({'key':r['key'],'label':r['label'],'score':logits[i].sigmoid().item(),
                        'logit':logits[i].item(),'tokens':r['tokens']})
                    if token_file:token_file.write(json.dumps({'key':r['key'],'scores':z[i,:r['tokens']].float().sigmoid().cpu().tolist()})+'\n')
        if token_file:token_file.close()
        metrics=evaluate([r['label'] for r in predictions],[r['score'] for r in predictions]);metrics.pop('pr_curve')
        metrics['bce']=total/len(subset)
        write(root/f'{name}.json',metrics)
        (root/f'{name}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in predictions))
        return metrics

    best=float('inf');selected=None;history=[]
    for epoch in range(1,cfg['epochs']+1):
        model.train();order=list(train);random.Random(SEED+epoch).shuffle(order);total=0.
        for offset in tqdm(range(0,len(order),cfg['batch_size']),desc=f'Mamba train {epoch}/{cfg["epochs"]}',mininterval=5):
            items=order[offset:offset+cfg['batch_size']];x,ends,y=batch(items)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):z=model(x)
            loss=F.binary_cross_entropy_with_logits(z[torch.arange(len(items),device='cuda'),ends].float(),y)
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
            total+=loss.item()*len(items)
        metrics=predict(validation,f'validation_epoch{epoch}')
        history.append({'epoch':epoch,'train_bce':total/len(train),'validation':metrics})
        write(root/'training_history.json',history)
        if metrics['bce']<best:
            best=metrics['bce'];selected=epoch
            torch.save(model.state_dict(),root/'selected_head.pt')
    write(root/'selection_before_test.json',{'epoch':selected,'validation_bce':best,
        'checkpoint_sha256':sha(root/'selected_head.pt')})
    model.load_state_dict(torch.load(root/'selected_head.pt',weights_only=True))
    trained=predict(test,'trained_test',True)
    model.load_state_dict(initial)
    untrained=predict(test,'initial_test')
    indexed={r['key']:r for r in json.loads((root/'kev_reference.json').read_text())}
    kev=evaluate([r['label'] for r in test],[indexed[r['key']]['score'] for r in test]);kev.pop('pr_curve')
    comparison={'common_test_n':len(test),'selected_epoch':selected,'models':{
        'qwen_mamba_random_head':untrained,'qwen_mamba_trained':trained,'kev08_original':kev},
        'rejections':rejects,'protocol_limits':cfg['limits']}
    write(root/'comparison.json',comparison)
    write(root/'status.json',{'phase':'complete'})
    print(json.dumps(comparison,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=('prepare','run'));p.add_argument('--root',type=Path,required=True)
    p.add_argument('--reference',type=Path)
    a=p.parse_args()
    if a.phase=='prepare':prepare(a.root,a.reference)
    else:run(a.root)
