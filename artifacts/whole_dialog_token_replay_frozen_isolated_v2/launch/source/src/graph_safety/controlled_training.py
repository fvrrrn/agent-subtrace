"""Audited four-arm pilot and gated multi-seed controlled Kev experiment."""
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

from .io import read_json, write_json, sha256
from .metrics import evaluate
from .step_training import load, encode, request, digest, QUESTION


def compact(rows):
    m = evaluate([r['label'] for r in rows], [r['score'] for r in rows])
    return {k:v for k,v in m.items() if k!='pr_curve'}


def score(model, enc):
    import torch
    with torch.autocast('cuda',dtype=torch.bfloat16): z=model(enc)[0].float()
    if not torch.isfinite(z).all(): raise ValueError('Nonfinite native logits')
    return z


def predict(model, tok, rows, cfg, folder, name):
    import torch
    from kev.model import ContextOverflow
    from tqdm.auto import tqdm
    folder.mkdir(parents=True,exist_ok=True)
    path=folder/f'{name}.jsonl'
    if path.exists(): raise FileExistsError(path)
    model.eval(); predictions=[]
    with path.open('w') as output, torch.inference_mode():
        for r in tqdm(rows,desc=name,mininterval=5):
            rec={k:r[k] for k in ('key','group','trajectory','split','mechanism','label')}
            rec['input_sha256']=digest(r['state'])
            try:
                enc,_=encode(tok,model,r['state'],cfg.get('eval_state_limit',cfg['state_limit']))
                z=score(model,enc)
                rec.update(status='complete',score=torch.softmax(z,-1)[1].item(),
                           logits=z.cpu().tolist(),state_tokens=enc['state_tokens'])
                raw=z*model.head.temperature
                rec['ce']=torch.nn.functional.cross_entropy(raw[None],torch.tensor([r['label']],device='cuda')).item()
            except ContextOverflow as error:
                rec.update(status='rejected_context',error=str(error))
            except torch.OutOfMemoryError as error:
                rec.update(status='rejected_oom',error=str(error))
                torch.cuda.empty_cache()
            output.write(json.dumps(rec)+'\n');output.flush();predictions.append(rec)
    complete=[r for r in predictions if r['status']=='complete']
    result={'metrics':compact(complete),'attempted':len(rows),
            'rejections':[r for r in predictions if r['status']!='complete'],
            'mean_ce':sum(r['ce'] for r in complete)/len(complete)}
    write_json(folder/f'{name}.json',result)
    return predictions,result


def save(cp,tok,model,folder,epoch,cfg):
    from kev.checkpoint import write_meta
    folder.mkdir(exist_ok=True)
    model.lm.save_pretrained(folder);tok.save_pretrained(folder)
    meta=deepcopy(cp.meta)
    meta.head={k:v.detach().cpu() for k,v in model.head.state_dict().items()}
    meta.extra={'parent':cfg['checkpoint'],'experiment':'controlled_monotonic_verified_v3' if cfg.get('verified_ordering') else 'controlled_monotonic_v2',
                'selected_epoch':epoch,'records_sha256':cfg['records_sha256']}
    write_meta(folder,meta)


def model_environment(cp,model):
    import torch
    from kev.predictors import kernel_environment
    return {'environment':kernel_environment(model,'cuda'), 'initial_weights_sha256':cp.weights_sha256(),
            'initial_head_sha256':sha256(cp.file('head.pt')),'temperature':model.head.temperature,
            'tf32_matmul':torch.backends.cuda.matmul.allow_tf32,'tf32_cudnn':torch.backends.cudnn.allow_tf32,
            'sdpa_flash':torch.backends.cuda.flash_sdp_enabled(),'sdpa_efficient':torch.backends.cuda.mem_efficient_sdp_enabled()}


def frozen_hash(model):
    import torch
    h=hashlib.sha256()
    for name,p in model.named_parameters():
        if p.requires_grad:continue
        h.update(name.encode());h.update(p.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def configure_training(model):
    for name,p in model.named_parameters(): p.requires_grad_('lora_' in name or name.startswith('head.'))
    for cfg in model.lm.peft_config.values(): cfg.inference_mode=False
    model.lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})


def train_arm(root,stage,arm,seed,train,validation,cfg):
    import torch
    import torch.nn.functional as F
    from kev.model import ContextOverflow
    from tqdm.auto import tqdm
    folder=root/stage/f'{arm}_{seed}';folder.mkdir(parents=True,exist_ok=False)
    torch.manual_seed(seed);random.seed(seed)
    cp,tok,model=load(cfg['checkpoint'])
    env=model_environment(cp,model);configure_training(model)
    initial_frozen=frozen_hash(model)
    env['trainable_names']=[n for n,p in model.named_parameters() if p.requires_grad]
    env['trainable_parameters']=sum(p.numel() for p in model.parameters() if p.requires_grad)
    write_json(folder/'environment.json',env)
    encs={}
    for r in tqdm(train,desc='admit train',mininterval=5):
        # No silent dropping changes any arm's training examples or pairs.
        encs[r['key']],_=encode(tok,model,r['state'],cfg['state_limit'])
    lookup={r['key']:r for r in train}
    pairs={field:[(r[field],r['key']) for r in train if r[field]]
           for field in ('previous_unsafe','previous_safe')}
    assert all(v for v in pairs.values())
    assert all(a in lookup and b in lookup for v in pairs.values() for a,b in v)
    params=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(params,lr=cfg['lr'],weight_decay=.01)
    weights=cfg['arms'][arm];best=float('inf');selected=None;gradient_audit=None
    epochs=1 if stage=='pilot' else cfg['epochs']
    schedule=[];started=time.perf_counter()
    for epoch in range(1,epochs+1):
        order=list(train);rng=random.Random(seed+epoch);rng.shuffle(order)
        model.train();ce_sum=up_sum=down_sum=0.;batches=0
        for begin in tqdm(range(0,len(order),cfg['accumulation']),desc=f'{stage}/{arm}/{seed}/epoch{epoch}',mininterval=5):
            batch=order[begin:begin+cfg['accumulation']]
            optimizer.zero_grad(set_to_none=True)
            for r in batch:
                z=score(model,encs[r['key']])
                ce=F.cross_entropy(z[None],torch.tensor([r['label']],device='cuda'))
                (ce/len(batch)).backward();ce_sum+=ce.item()
            chosen={f:rng.choice(v) for f,v in pairs.items()}
            schedule.append({'epoch':epoch,'keys':[r['key'] for r in batch],'pairs':chosen})
            # Exactly the same forwards in every arm, including NOUL-only: matched dropout RNG.
            # Uniform pair sampling estimates a mean per class, without sparse large importance spikes.
            for field,weight,sign in (('previous_unsafe',weights[0],1),('previous_safe',weights[1],-1)):
                early,late=chosen[field]
                ze=score(model,encs[early]);zl=score(model,encs[late])
                delta=(zl[1]-zl[0])-(ze[1]-ze[0])
                hinge=F.relu(cfg['rank_margin']-sign*delta)
                (weight*hinge).backward()
                if sign==1:up_sum+=hinge.item()
                else:down_sum+=hinge.item()
            if gradient_audit is None:
                norms={kind:sum(p.grad.detach().float().square().sum().item()
                               for n,p in model.named_parameters() if p.grad is not None and predicate(n))**.5
                       for kind,predicate in {'head':lambda n:n.startswith('head.'),'lora':lambda n:'lora_' in n}.items()}
                assert all(v>0 for v in norms.values())
                assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
                gradient_audit={'gradient_norms':norms,'frozen_gradients_absent':True}
                write_json(folder/'gradient_audit.json',gradient_audit)
            torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
            optimizer.step();batches+=1
            if batches%10==0:
                write_json(folder/'status.json',{'phase':'training','epoch':epoch,'steps':min(begin+len(batch),len(order)),
                    'total_steps':len(order),'mean_ce':ce_sum/min(begin+len(batch),len(order))})
        _,val=predict(model,tok,validation,cfg,folder,f'validation_epoch{epoch}')
        if val['rejections']:raise ValueError('Synthetic validation rejected')
        write_json(folder/f'epoch{epoch}.json',{'train_ce':ce_sum/len(train),
            'unsafe_hinge':up_sum/batches,'safe_hinge':down_sum/batches,'validation':val})
        if val['mean_ce']<best:
            best=val['mean_ce'];selected=epoch
            save(cp,tok,model,folder/'checkpoint',epoch,cfg)
    write_json(folder/'schedule.json',schedule)
    final_frozen=frozen_hash(model)
    if final_frozen!=initial_frozen:raise ValueError('Frozen backbone weights changed')
    write_json(folder/'frozen_weights_audit.json',{'initial':initial_frozen,'final':final_frozen,'passed':True})
    # Compare an actual saved-and-reloaded native readout before accepting this arm.
    del optimizer,encs,model;gc.collect();torch.cuda.empty_cache()
    cp2,tok2,reloaded=load(folder/'checkpoint')
    write_json(folder/'checkpoint_audit.json',{'selected_epoch':selected,'validation_ce':best,
        'weights_changed':cp2.weights_sha256()!=env['initial_weights_sha256'],
        'head_changed':sha256(cp2.file('head.pt'))!=env['initial_head_sha256'],
        'temperature':reloaded.head.temperature,'seconds':time.perf_counter()-started})
    assert reloaded.head.temperature==env['temperature']
    pred,_=predict(reloaded,tok2,validation[:4],cfg,folder,'reload_probe')
    expected={r['key']:r for r in map(json.loads,(folder/f'validation_epoch{selected}.jsonl').read_text().splitlines())}
    error=max(abs(a-b) for p in pred for a,b in zip(p['logits'],expected[p['key']]['logits']))
    if error>1e-5:raise ValueError(f'Save/reload drift: {error}')
    write_json(folder/'reload_audit.json',{'max_logit_error':error,'passed':True})
    del reloaded;gc.collect();torch.cuda.empty_cache()


def run(root,stage,worker_arm=None,evaluation_worker=False):
    import torch
    from kev.api import SystemOneRequest,to_record
    cfg=read_json(root/'protocol.json')
    assert sha256(root/'records.json')==cfg['records_sha256'] and cfg['question']==QUESTION
    rows=read_json(root/'records.json')
    _,mapping=to_record(SystemOneRequest.model_validate(request('mapping')))
    assert mapping[0]['keys']==['false','true']
    if not worker_arm:write_json(root/f'{stage}_mapping.json',mapping)
    if stage=='full':
        gate=read_json(root/'pilot_gate.json')
        if gate['status']!='passed':raise ValueError('Pilot gate not passed')
        assert gate['records_sha256']==cfg['records_sha256']
        assert gate['code_sha256']=={p.name:sha256(p) for p in Path(__file__).parent.glob('*.py')}
    if read_json(root/'semantic_audit.json')['status']!='passed':raise ValueError('Semantic audit missing')
    folder=root/stage
    if not worker_arm:folder.mkdir(exist_ok=False)
    train=[r for r in rows if r['split']=='train'];validation=[r for r in rows if r['split']=='validation']
    if stage=='pilot':
        groups=sorted({r['group'] for r in train},key=digest)[:32]
        train=[r for r in train if r['group'] in groups]
    if worker_arm:
        for seed in cfg['seeds'][:1] if stage=='pilot' else cfg['seeds']:
            if evaluation_worker:
                if stage!='full':raise ValueError('Only frozen full-stage checkpoints can see final test')
                checkpoint=folder/f'{worker_arm}_{seed}'
                cp,tok,model=load(checkpoint/'checkpoint')
                for split in ('test','external_test'):
                    predict(model,tok,[r for r in rows if r['split']==split],cfg,checkpoint,split)
                del model;gc.collect();torch.cuda.empty_cache()
            else:train_arm(root,stage,worker_arm,seed,train,validation,cfg)
        return
    cp,tok,model=load(cfg['checkpoint'])
    write_json(folder/'baseline_environment.json',model_environment(cp,model))
    splits=('validation','external_dev') if stage=='pilot' else ()
    for split in splits:
        predict(model,tok,[r for r in rows if r['split']==split],cfg,folder,f'baseline_{split}')
    # Native loss-direction sanity: increasing late YES-minus-NO lowers unsafe hinge;
    # increasing it raises safe hinge. No use of final-test predictions.
    early=torch.tensor(0.,requires_grad=True);late=torch.tensor(0.,requires_grad=True)
    up=torch.relu(.2+early-late);gu=torch.autograd.grad(up,(early,late))
    down=torch.relu(.2+late-early);gd=torch.autograd.grad(down,(early,late))
    assert tuple(x.item() for x in gu)==(1.,-1.) and tuple(x.item() for x in gd)==(-1.,1.)
    write_json(folder/'loss_direction_audit.json',{'unsafe_grad':[x.item() for x in gu],'safe_grad':[x.item() for x in gd]})
    del model;gc.collect();torch.cuda.empty_cache()
    workers=[]
    for gpu,arm in enumerate(cfg['arms'],start=1):
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu))
        log=(folder/f'{arm}.log').open('w')
        proc=subprocess.Popen([sys.executable,'-u','-m',__package__+'.controlled_training',stage,
            '--root',str(root),'--worker-arm',arm],env=env,stdout=log,stderr=subprocess.STDOUT)
        log.close();workers.append((arm,proc))
    write_json(folder/'workers.json',{a:p.pid for a,p in workers})
    failed=[]
    for arm,proc in workers:
        code=proc.wait()
        print(f'{stage}/{arm}: exit={code}',flush=True)
        if code:failed.append((arm,code))
    if failed:raise RuntimeError(f'Worker failures: {failed}; full-stage gate remains closed')
    if stage=='full':
        for seed in cfg['seeds']:
            locations=[folder/f'{arm}_{seed}' for arm in cfg['arms']]
            envs=[read_json(p/'environment.json') for p in locations]
            assert all(e==envs[0] for e in envs)
            assert len({sha256(p/'schedule.json') for p in locations})==1
            assert all(read_json(p/'frozen_weights_audit.json')['passed'] for p in locations)
            assert all(read_json(p/'reload_audit.json')['passed'] for p in locations)
        write_json(folder/'matched_training_audit.json',{'passed':True,'seeds':cfg['seeds']})
        frozen={f'{arm}_{seed}':{
            'selection':read_json(folder/f'{arm}_{seed}'/'checkpoint_audit.json'),
            'adapter_sha256':sha256(folder/f'{arm}_{seed}'/'checkpoint'/'adapter_model.safetensors'),
            'head_sha256':sha256(folder/f'{arm}_{seed}'/'checkpoint'/'head.pt')}
            for arm in cfg['arms'] for seed in cfg['seeds']}
        write_json(folder/'all_checkpoints_frozen_before_test.json',frozen)
        evaluators=[]
        for gpu,arm in enumerate(cfg['arms'],start=1):
            log=(folder/f'{arm}_evaluation.log').open('w')
            proc=subprocess.Popen([sys.executable,'-u','-m',__package__+'.controlled_training',stage,
                '--root',str(root),'--worker-arm',arm,'--evaluation-worker'],
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu)),stdout=log,stderr=subprocess.STDOUT)
            log.close();evaluators.append((arm,proc))
        write_json(folder/'evaluation_workers.json',{a:p.pid for a,p in evaluators})
        cp,tok,model=load(cfg['checkpoint'])
        for split in ('test','external_test'):
            predict(model,tok,[r for r in rows if r['split']==split],cfg,folder,f'baseline_{split}')
        del model;gc.collect();torch.cuda.empty_cache()
        failed=[(arm,proc.wait()) for arm,proc in evaluators]
        if any(code for _,code in failed):raise RuntimeError(f'Final evaluation failures: {failed}')
    if stage=='pilot':
        arms=[folder/f'{a}_{cfg["seeds"][0]}' for a in cfg['arms']]
        envs=[read_json(p/'environment.json') for p in arms]
        assert all(e==envs[0] for e in envs)
        assert len({sha256(p/'schedule.json') for p in arms})==1
        assert all(read_json(p/'reload_audit.json')['passed'] for p in arms)
        assert all(read_json(p/'checkpoint_audit.json')['weights_changed'] and read_json(p/'checkpoint_audit.json')['head_changed'] for p in arms)
        # Structural pass is necessary but experiment difficulty still needs inspection by the parent.
        write_json(root/'pilot_gate.json',{'status':'awaiting_review','records_sha256':cfg['records_sha256'],
            'technical_checks_passed':True,'code_sha256':{p.name:sha256(p) for p in Path(__file__).parent.glob('*.py')},
            'no_test_inference':True,'note':'Parent must inspect examples, baseline difficulty and pilot output. Metric gains are not a gate criterion.'})
    write_json(folder/'status.json',{'phase':'complete'})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=('pilot','full'));p.add_argument('--root',type=Path,required=True)
    p.add_argument('--worker-arm',choices=('noul','unsafe_up','bidirectional'))
    p.add_argument('--evaluation-worker',action='store_true')
    a=p.parse_args();run(a.root,a.stage,a.worker_arm,a.evaluation_worker)


if __name__=='__main__':main()
