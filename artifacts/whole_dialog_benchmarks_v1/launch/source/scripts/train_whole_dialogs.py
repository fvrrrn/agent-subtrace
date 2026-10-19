"""One complete dialog per forward; two residual Mamba blocks and token safety head."""
import argparse
import gc
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import random
import time

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from prepare_whole_dialogs import BASE, REVISION, digest, write
from graph_safety.metrics import evaluate

SEED = 20261017


def read(path):
    return json.loads(path.read_text())


def file_hash(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def frozen_hash(model):
    h = hashlib.sha256()
    for name, param in model.named_parameters():
        if not param.requires_grad:
            h.update(name.encode())
            h.update(param.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def backbone(lora):
    from transformers import Qwen3_5TextModel
    model = Qwen3_5TextModel.from_pretrained(BASE, revision=REVISION,
        local_files_only=True, dtype=torch.bfloat16, attn_implementation='sdpa').cuda()
    model.requires_grad_(False)
    if lora:
        from peft import LoraConfig, get_peft_model
        model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.,
                              target_modules='all-linear', bias='none'))
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    return model


class Head(torch.nn.Module):
    def __init__(self):
        super().__init__()
        from mamba_ssm import Mamba2
        self.input_norm = torch.nn.LayerNorm(2560)
        self.projection = torch.nn.Linear(2560, 256)
        self.norms = torch.nn.ModuleList([torch.nn.LayerNorm(256) for _ in range(2)])
        self.blocks = torch.nn.ModuleList([Mamba2(d_model=256, d_state=64, expand=2,
            headdim=64, chunk_size=128, use_mem_eff_path=True) for _ in range(2)])
        self.final_norm = torch.nn.LayerNorm(256)
        self.readout = torch.nn.Linear(256, 1)

    def forward(self, x):
        x = self.projection(self.input_norm(x))
        for norm, block in zip(self.norms, self.blocks):
            x = x + block(norm(x))
        return self.readout(self.final_norm(x)).squeeze(-1)


def load_data(root):
    audit = read(root / 'data_audit.json')
    assert file_hash(root / 'dialogs.json') == audit['dialogs_sha256']
    assert audit['base'] == BASE and audit['revision'] == REVISION
    return read(root / 'dialogs.json'), audit


def tensor_input(row):
    n = math.ceil(row['tokens']/128)*128
    ids = torch.zeros((1,n), dtype=torch.long, device='cuda')
    mask = torch.zeros_like(ids)
    ids[0,:row['tokens']] = torch.tensor(row['ids'], device='cuda')
    mask[0,:row['tokens']] = 1
    return ids,mask


def encode(model, row):
    ids,mask = tensor_input(row)
    return model(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state


def cache(args):
    rows, audit = load_data(args.data)
    folder = args.root / 'features'; folder.mkdir(parents=True, exist_ok=True)
    model = backbone(False).eval()
    weights = frozen_hash(model)
    identity = {'base':BASE, 'revision':REVISION, 'weights_sha256':weights,
                'template_sha256':audit['template_sha256'], 'dtype':'bf16', 'attention':'sdpa',
                'torch':str(torch.__version__), 'padding':'right to128; one complete dialog per forward'}
    with torch.inference_mode():
        for row in tqdm(rows[args.shard::2], desc=f'Whole-dialog cache shard{args.shard}'):
            h = encode(model,row)[0,:row['tokens']].cpu().clone()
            assert torch.isfinite(h).all()
            torch.save({'hidden':h, 'ids_sha256':row['ids_sha256'], 'identity':identity}, folder/(row['key']+'.pt'))
    assert weights == frozen_hash(model)
    write(folder/f'complete{args.shard}.json',dict(identity, dialogs_sha256=audit['dialogs_sha256'],
        shard=args.shard, count=len(rows[args.shard::2])))


def forward(model, head, row, root):
    if model is None:
        # Existing local caches serialized torch.__version__ as TorchVersion.
        with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
            entry = torch.load(root/'features'/(row['key']+'.pt'), map_location='cpu', weights_only=True)
        assert entry['ids_sha256']==row['ids_sha256']
        assert entry['identity']==cache_identity(str(root))
        n=math.ceil(row['tokens']/128)*128
        x=torch.zeros((1,n,2560),dtype=torch.bfloat16,device='cuda')
        x[0,:row['tokens']]=entry['hidden'].to('cuda')
    else:
        x=encode(model,row)
    with torch.autocast('cuda',dtype=torch.bfloat16):
        return head(x)[0,:row['tokens']].float()


@lru_cache(maxsize=1)
def cache_identity(root):
    meta=read(Path(root)/'features/complete0.json')
    return {k:v for k,v in meta.items() if k not in ('dialogs_sha256','shard','count')}


def loss_parts(z, spans):
    losses, ranks = [], []
    anchor = 0.
    for span in spans:
        start,end,y=span['start'],span['end'],span['label']
        u=torch.arange(1,end-start+1,device=z.device,dtype=torch.float32)/(end-start)
        target=anchor+(y-anchor)*u
        losses.append(F.binary_cross_entropy_with_logits(z[start:end],target))
        anchor=float(y)
    for early,late in zip(spans,spans[1:]):
        if early['label']==late['label']==1:
            ranks.append(F.relu(.2+z[early['end']-1]-z[late['end']-1]))
    return torch.stack(losses).mean(), torch.stack(ranks).mean() if ranks else z.sum()*0


def predict(model,head,rows,root,path):
    head.eval()
    if model is not None:model.eval()
    labels,scores,ces=[],[],[]
    with path.open('w') as out,torch.no_grad():
        for row in tqdm(rows,desc='Whole-dialog validation'):
            z=forward(model,head,row,root)
            for s in row['spans']:
                logit=z[s['end']-1];score=logit.sigmoid().item()
                labels.append(s['label']);scores.append(score)
                ces.append(F.binary_cross_entropy_with_logits(logit,logit.new_tensor(float(s['label']))).item())
                out.write(json.dumps({'key':s['key'],'dialog':row['key'],'label':s['label'],
                    'score':score,'raw_logodds':logit.item(),'state_sha256':s['state_sha256'],
                    'ids_sha256':row['ids_sha256']})+'\n')
    m=evaluate(labels,scores);m.pop('pr_curve');m['endpoint_bce']=sum(ces)/len(ces)
    return m


def gate(args):
    rows,audit=load_data(args.data)
    row=next(r for r in rows if r['split']=='train' and len(r['spans'])>1 and r['tokens']<4096)
    results={}
    for lora in (False,True):
        torch.manual_seed(SEED)
        model=backbone(lora)
        before=frozen_hash(model)
        torch.manual_seed(SEED);head=Head().cuda()
        model.eval();head.eval()
        ids,mask=tensor_input(row)
        cut=row['spans'][0]['end']
        assert cut<row['tokens']
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            x=model(input_ids=ids,attention_mask=mask,use_cache=False).last_hidden_state
            z=head(x)
            changed=ids.clone();changed[:,cut:row['tokens']]=1234
            xx=model(input_ids=changed,attention_mask=mask,use_cache=False).last_hidden_state
            zz=head(xx)
            encoder_error=(x[:,:cut]-xx[:,:cut]).abs().max().item()
            prefix_error=(z[:,:cut]-zz[:,:cut]).abs().max().item()
            assert encoder_error==prefix_error==0
            state={k:v.detach().clone() for k,v in head.state_dict().items()}
            head.load_state_dict(state)
            reload_error=(head(x)-z).abs().max().item();assert reload_error==0
        del x,z,xx,zz,changed,state,ids,mask
        head.train();model.train(lora)
        z=forward(model,head,row,args.root);z.retain_grad()
        ce,rank=loss_parts(z,row['spans']);(ce+.1*rank).backward()
        support=torch.zeros_like(z,dtype=torch.bool)
        for s in row['spans']:support[s['start']:s['end']]=True
        assert torch.count_nonzero(z.grad[~support])==0
        norms={n:p.grad.float().norm().item() for n,p in head.named_parameters() if p.grad is not None}
        assert all(math.isfinite(v) for v in norms.values())
        assert all(any(v>0 for n,v in norms.items() if n.startswith(f'blocks.{i}.')) for i in (0,1))
        assert norms['readout.weight']>0
        adapter_norms={n:p.grad.float().norm().item() for n,p in model.named_parameters() if p.requires_grad and p.grad is not None}
        if lora:assert any(v>0 for v in adapter_norms.values()) and all(math.isfinite(v) for v in adapter_norms.values())
        assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
        assert before==frozen_hash(model)
        del z,ce,rank
        head.zero_grad(set_to_none=True);model.zero_grad(set_to_none=True)
        longest=max(rows,key=lambda r:r['tokens'])
        torch.cuda.reset_peak_memory_stats()
        long_z=forward(model,head,longest,args.root)
        long_ce,long_rank=loss_parts(long_z,longest['spans'])
        (long_ce+.1*long_rank).backward()
        assert all(torch.isfinite(p.grad).all() for p in head.parameters() if p.grad is not None)
        max_memory=torch.cuda.max_memory_allocated()
        del long_z,long_ce,long_rank
        results[str(lora)]={'encoder_suffix_error':encoder_error,'head_suffix_error':prefix_error,
            'reload_error':reload_error,'head_gradients':norms,'adapter_gradients':adapter_norms,
            'frozen_sha256':before,'loss_support_passed':True,'actions_in_one_forward':len(row['spans']),
            'longest_dialog_tokens':longest['tokens'],'longest_backward_peak_bytes':max_memory}
        del model,head;gc.collect();torch.cuda.empty_cache()
    write(args.root/'cuda_gate.json',{'status':'passed','results':results,'audit_gradients_discarded':True})


def train(args):
    rows,audit=load_data(args.data)
    assert read(args.root/'cuda_gate.json')['status']=='passed'
    folder=args.root/'arms'/args.arm;folder.mkdir(parents=True,exist_ok=False)
    lora=args.arm.startswith('lora');mono=args.arm.endswith('_mono')
    torch.manual_seed(SEED);random.seed(SEED)
    model=backbone(True) if lora else None
    torch.manual_seed(SEED);head=Head().cuda()
    initial=digest({k:hashlib.sha256(v.detach().cpu().numpy().tobytes()).hexdigest() for k,v in head.state_dict().items()})
    frozen_before=frozen_hash(model) if model is not None else None
    if not lora:
        for shard in range(2):
            assert read(args.root/'features'/f'complete{shard}.json')['dialogs_sha256']==audit['dialogs_sha256']
    groups=[{'params':head.parameters(),'lr':3e-4}]
    if model is not None:groups.append({'params':[p for p in model.parameters() if p.requires_grad],'lr':1e-5})
    opt=torch.optim.AdamW(groups,weight_decay=.01,fused=True)
    train_rows=[r for r in rows if r['split']=='train'];validation=[r for r in rows if r['split']=='validation']
    write(folder/'run.json',{'base':BASE,'revision':REVISION,'lora':lora,'mono_weight':.1 if mono else 0,
        'initial_head_sha256':initial,'seed':SEED,'epochs':2,'accumulation_dialogs':4,
        'forward_unit':'one complete dialog, all labeled action spans, no prefix replay',
        'head_parameters':sum(p.numel() for p in head.parameters()),
        'lora_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad) if model is not None else 0,
        'data_audit':audit,'torch':torch.__version__,'source_sha256':file_hash(Path(__file__))})
    best=float('inf');history=[];started=time.monotonic()
    for epoch in range(1,3):
        order=list(train_rows);random.Random(SEED+epoch).shuffle(order)
        head.train()
        if model is not None:model.train()
        opt.zero_grad(set_to_none=True);total_ce=total_rank=0.
        for i,row in enumerate(tqdm(order,desc=f'{args.arm} epoch{epoch}/2',unit='dialog')):
            z=forward(model,head,row,args.root)
            ce,rank=loss_parts(z,row['spans'])
            window=min(4,len(order)-(i//4)*4)
            ((ce+(.1 if mono else 0)*rank)/window).backward()
            total_ce+=ce.detach().item();total_rank+=rank.detach().item()
            if (i+1)%4==0 or i+1==len(order):
                params=[p for g in opt.param_groups for p in g['params']]
                norm=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
                if epoch==1 and i<4:write(folder/'gradient_audit.json',{'total_norm':norm.item(),'finite':True})
                opt.step();opt.zero_grad(set_to_none=True)
            del z,ce,rank
        metrics=predict(model,head,validation,args.root,folder/f'validation_epoch{epoch}.jsonl')
        history.append({'epoch':epoch,'train_bce':total_ce/len(order),'train_rank':total_rank/len(order),
                        'validation':metrics,'seconds':time.monotonic()-started})
        write(folder/'history.json',history)
        if metrics['endpoint_bce']<best:
            best=metrics['endpoint_bce'];torch.save(head.state_dict(),folder/'selected_head.pt')
            if model is not None:model.save_pretrained(folder/'adapter')
            write(folder/'selection.json',{'epoch':epoch,'validation_bce':best,
                'head_sha256':file_hash(folder/'selected_head.pt'),'criterion':'validation hard endpoint BCE'})
    frozen_after=frozen_hash(model) if model is not None else None
    assert frozen_before==frozen_after
    write(folder/'complete.json',{'status':'complete','epochs':2,'frozen_before':frozen_before,
        'frozen_after':frozen_after,'seconds':time.monotonic()-started})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=('gate','cache','train'))
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--shard',type=int,choices=(0,1))
    p.add_argument('--arm',choices=('frozen_linear','frozen_linear_mono','lora_linear','lora_linear_mono'))
    a=p.parse_args();{'gate':gate,'cache':cache,'train':train}[a.phase](a)
