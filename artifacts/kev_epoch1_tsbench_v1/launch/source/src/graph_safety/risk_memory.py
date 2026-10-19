"""Matched trajectory training with raw, symmetric and fast-attack/slow-release risk."""
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import gc
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys

from .controlled_training import compact, configure_training, frozen_hash, model_environment, predict, score
from .io import read_json, read_jsonl, sha256, write_json
from .step_training import QUESTION, digest, encode, load, request


ARMS={'noul':('raw',0.),'raw_up':('raw',.1),
      'symmetric':('symmetric',.1),'asymmetric':('asymmetric',.1)}


def memory_logits(raw,temperature,mode,alpha=.9):
    """Exact probability EMA in stable log space; output is in native raw-logit units."""
    import torch
    import torch.nn.functional as F
    if mode=='raw':return raw
    if mode not in ('symmetric','asymmetric'):raise ValueError(mode)
    yes=F.logsigmoid(raw/temperature);no=F.logsigmoid(-raw/temperature)
    # Same unbiased cold-start in both smoothers: R_1 = q_1.
    lp,lm=[yes[0]],[no[0]]
    for i in range(1,len(raw)):
        p=torch.logaddexp(lp[-1]+math.log(alpha),yes[i]+math.log1p(-alpha))
        m=torch.logaddexp(lm[-1]+math.log(alpha),no[i]+math.log1p(-alpha))
        if mode=='asymmetric':
            attack=(yes[i]-no[i]) >= (lp[-1]-lm[-1])
            p=torch.where(attack,yes[i],p);m=torch.where(attack,no[i],m)
        lp.append(p);lm.append(m)
    return temperature*(torch.stack(lp)-torch.stack(lm))


def sequences(rows):
    grouped=defaultdict(list)
    for r in rows:grouped[r['trajectory']].append(r)
    return {key:sorted(value,key=lambda r:r['position']) for key,value in grouped.items()}


def permission_epoch(row):
    state=json.loads(row['state']);a=state['proposed_action']
    if a['name']!='deliver':return None
    recipient=a['arguments']['recipient']
    return digest([e['content'] for e in state['observed_history'] if e['role']=='user'
        and (e['content'].get('approve')==recipient or e['content'].get('revoke')==recipient)])


def make_pairs(seq):
    result=[]
    for j,late in enumerate(seq):
        if late['label']!=1 or late.get('ordering_value') is None:continue
        action=json.loads(late['state'])['proposed_action']
        for i,early in enumerate(seq[:j]):
            if early['label']!=1 or early.get('ordering_value') is None:continue
            old=json.loads(early['state'])['proposed_action']
            if old['name']!=action['name'] or early['ordering_value']>=late['ordering_value']:continue
            if action['name']=='deliver':
                if old['arguments']['recipient']!=action['arguments']['recipient']:continue
                if permission_epoch(early)!=permission_epoch(late):continue
            elif action['name']=='charge':
                if old['arguments']['account']!=action['arguments']['account']:continue
            else:raise ValueError('Unsupported ordinal pair')
            result.append({'early':i,'late':j,'early_key':early['key'],'late_key':late['key'],
                'safe_between':sum(r['label']==0 for r in seq[i+1:j])})
    return result


def prepare(root,reference):
    root.mkdir(parents=True,exist_ok=False)
    old=read_json(reference/'protocol.json')
    assert old['records_sha256']==sha256(reference/'records.json')
    shutil.copy2(reference/'records.json',root/'records.json')
    rows=read_json(root/'records.json')
    synthetic=[r for r in rows if r['split'] in ('train','validation','test')]
    seqs=sequences(synthetic);pairs={k:make_pairs(v) for k,v in seqs.items()}
    for seq in seqs.values():
        assert len({r['group'] for r in seq})==1 and len({r['split'] for r in seq})==1
        assert len({r['position'] for r in seq})==len(seq)
        for early,late in zip(seq,seq[1:]):
            a=json.loads(early['state']);b=json.loads(late['state']);history=a['observed_history']
            assert b['observed_history'][:len(history)]==history
            assert b['observed_history'][len(history)]=={'role':'agent','action':a['proposed_action']}
    write_json(root/'pairs.json',pairs)
    cfg={k:old[k] for k in ('checkpoint','question','state_limit','eval_state_limit','branch_limit','records_sha256')}
    cfg.update(experiment='risk_memory_v1',reference=str(reference),seed=20261017,epochs=2,lr=1e-5,
        weight_decay=.01,gradient_clip=1.,alpha=.9,rank_margin=.2,arms=ARMS,
        pairs_sha256=sha256(root/'pairs.json'),source_commit='85ccf56a0afb952323c621055b02876e8498d60c',
        training='One complete chronological trajectory per optimizer step, no state detach. Epoch mean objective: mean CE over all actions + lambda mean hinge over all verified unsafe pairs. Whole trajectories shuffled; same forward order/RNG in all arms. Original LoRA and native head only.',
        smoothing='Probability-space EMA, implemented exactly in log space. R_1=q_1. Symmetric: alpha Rprev+(1-alpha)q. Asymmetric: q if q>=Rprev, otherwise EMA. q=sigmoid(raw YES-NO / shipped temperature) in both training and evaluation. Ranking uses shipped_temperature * logit(R), preserving raw-logit margin units.',
        pairs='Same operator/entity and increasing observable ordinal stage; all matching pairs, including intervening safe actions. Delivery pairs cannot cross authenticated permission updates. Identical pair set and normalization in all three ranking arms; no safe-down loss.',
        selection='Two fixed epochs; choose checkpoint by validation raw NOUL CE, never test or smoothed-state metrics.',
        evaluation='Primary AP/ROC-AUC of raw current-action q. Smoothed R is a separate alarm state, never evaluated against current-action labels as if they were state gold. External StepGuard examples are isolated: no recurrence across them.',
        scope='One-seed exploratory trial after a short engineering pilot. Both evaluation sets have been inspected before; not untouched confirmatory tests. No alpha/lambda/epoch sweep.',
        code_sha256={p.name:sha256(p) for p in Path(__file__).parent.glob('*.py')})
    write_json(root/'protocol.json',cfg)
    write_json(root/'data_audit.json',{
        'records_unchanged':True,'counts':dict(Counter(r['split'] for r in rows)),
        'trajectories':dict(Counter(v[0]['split'] for v in seqs.values())),
        'unsafe_pairs':dict(Counter(seqs[k][0]['split'] for k,v in pairs.items() for _ in v)),
        'unsafe_pairs_with_safe_gap':dict(Counter(seqs[k][0]['split'] for k,v in pairs.items() for p in v if p['safe_between'])),
        'authorization_epoch_guard':True,'causal_prefixes_checked':True})


def audit_math(root):
    import torch
    temperature=2.3510958125672174
    q=torch.tensor([.1,.85,.05,.05,.95],dtype=torch.float64)
    raw=(torch.logit(q)*temperature).requires_grad_()
    modes={m:memory_logits(raw,temperature,m) for m in ('raw','symmetric','asymmetric')}
    values={m:torch.sigmoid(v/temperature) for m,v in modes.items()}
    expected=[.1,.85,.77,.698,.95]
    assert max(abs(x-y) for x,y in zip(values['asymmetric'].tolist(),expected))<1e-10
    g=torch.autograd.grad(modes['asymmetric'][3],raw,retain_graph=True)[0]
    assert g[1]>0 and g[2]>0 and g[3]>0 and g[4]==0
    for mode in ('symmetric','asymmetric'):
        truncated=memory_logits(raw[:4],temperature,mode)
        assert torch.equal(truncated,modes[mode][:4])
        constant=memory_logits(torch.full((20,),2.,dtype=torch.float64),temperature,mode)
        assert torch.max(torch.abs(constant-2.))<1e-10
    extreme=torch.tensor([-100.,100.,-100.,-100.,100.],requires_grad=True)
    stable=memory_logits(extreme,temperature,'asymmetric')
    assert torch.isfinite(stable).all()
    stable.sum().backward();assert torch.isfinite(extreme.grad).all()
    assert abs(values['asymmetric'].max().item()-q.max().item())<1e-10
    gap=(torch.logit(torch.tensor([.2,.9,.05,.8],dtype=torch.float64))*temperature).requires_grad_()
    h=memory_logits(gap,temperature,'asymmetric')
    hinge=torch.relu(.2+h[1]-h[3])
    gap_gradient=torch.autograd.grad(hinge,gap)[0]
    assert gap_gradient[2]!=0
    write_json(root/'math_audit.json',{'passed':True,'toy_q':q.tolist(),
        'toy_R':{k:v.detach().tolist() for k,v in values.items()},
        'gradient_to_earlier_unsafe_and_intermediate_safe':g.tolist(),
        'prefix_invariance':True,'constant_input_stationary':True,'extreme_logit_gradients_finite':True,
        'gap_pair_auxiliary_gradient':gap_gradient.tolist(),
        'known_tradeoff':'Auxiliary ranking can push intermediate safe raw scores upward; CE opposes this. Report raw false alarms, not just attractive smoothed curves.',
        'note':'Asymmetric postprocessing preserves the trajectory maximum. Alarm-state dynamics have no independent active-threat gold here.'})


def describe_memory(rows,predictions,pairs,mode,temperature,alpha,path):
    import torch
    indexed={r['key']:r for r in predictions}
    totals=Counter()
    with path.open('w') as stream:
        for key,seq in sequences(rows).items():
            if any(indexed[r['key']]['status']!='complete' for r in seq):
                totals['rejected_trajectories']+=1;continue
            raw=torch.tensor([(indexed[r['key']]['logits'][1]-indexed[r['key']]['logits'][0])*temperature for r in seq])
            h=memory_logits(raw,temperature,mode,alpha);q=torch.sigmoid(raw/temperature);risk=torch.sigmoid(h/temperature)
            for i,r in enumerate(seq):
                stream.write(json.dumps({'key':r['key'],'trajectory':key,'q':q[i].item(),
                    'R':risk[i].item(),'raw_risk_logodds':h[i].item()})+'\n')
            for pair in pairs[key]:
                a,b=pair['early'],pair['late'];delta=(h[b]-h[a]).item()
                totals['pairs']+=1;totals['strict_up']+=int(delta>0);totals['margin_met']+=int(delta>=.2)
                if pair['safe_between']:
                    totals['gap_pairs']+=1;totals['gap_strict_up']+=int(delta>0)
            if mode=='asymmetric':
                assert abs(risk.max().item()-q.max().item())<1e-6
                assert bool(torch.all(risk[1:]>=alpha*risk[:-1]-1e-6))
                assert bool(torch.all(risk>=q-1e-6))
            totals['complete_trajectories']+=1
    return {'counts':dict(totals),'not_state_accuracy':True}


def save_checkpoint(cp,tok,model,folder,cfg,arm,epoch):
    from kev.checkpoint import write_meta
    folder.mkdir(exist_ok=True);model.lm.save_pretrained(folder);tok.save_pretrained(folder)
    meta=deepcopy(cp.meta);meta.head={k:v.detach().cpu() for k,v in model.head.state_dict().items()}
    meta.extra={'parent':cfg['checkpoint'],'experiment':cfg['experiment'],'arm':arm,
                'selected_epoch':epoch,'records_sha256':cfg['records_sha256'],'alpha':cfg['alpha']}
    write_meta(folder,meta)


def train(root,stage,arm):
    import torch
    import torch.nn.functional as F
    from tqdm.auto import tqdm
    cfg=read_json(root/'protocol.json');mode,weight=cfg['arms'][arm]
    folder=root/stage/arm;folder.mkdir(exist_ok=False)
    rows=read_json(root/'records.json');training=[r for r in rows if r['split']=='train']
    if stage=='pilot':
        groups=sorted({r['group'] for r in training},key=digest)[:32]
        training=[r for r in training if r['group'] in groups]
    validation=[r for r in rows if r['split']=='validation'];seqs=sequences(training)
    allpairs=read_json(root/'pairs.json');npairs=sum(len(allpairs[k]) for k in seqs)
    assert npairs>0
    torch.manual_seed(cfg['seed']);random.seed(cfg['seed'])
    cp,tok,model=load(cfg['checkpoint']);configure_training(model)
    env=model_environment(cp,model);env['frozen_initial']=frozen_hash(model)
    write_json(folder/'environment.json',env)
    encs={r['key']:encode(tok,model,r['state'],cfg['state_limit'])[0] for r in tqdm(training,desc=f'{arm}/admit',mininterval=5)}
    params=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(params,lr=cfg['lr'],weight_decay=cfg['weight_decay'])
    best=float('inf');selected=None;schedule=[];grad_checked=False
    for epoch in range(1,(1 if stage=='pilot' else cfg['epochs'])+1):
        order=sorted(seqs);random.Random(cfg['seed']+epoch).shuffle(order)
        model.train();ce_sum=rank_sum=0.
        for n,key in enumerate(tqdm(order,desc=f'{stage}/{arm}/seed{cfg["seed"]}/epoch{epoch}',mininterval=5)):
            seq=seqs[key];optimizer.zero_grad(set_to_none=True)
            z=torch.stack([score(model,encs[r['key']]) for r in seq])
            target=torch.tensor([r['label'] for r in seq],device='cuda')
            ce=F.cross_entropy(z,target,reduction='sum')
            raw=z[:,1]-z[:,0];h=memory_logits(raw,model.head.temperature,mode,cfg['alpha'])
            hinges=[F.relu(cfg['rank_margin']+h[p['early']]-h[p['late']]) for p in allpairs[key]]
            rank=torch.stack(hinges).sum() if hinges else raw.sum()*0
            # Mean over optimizer trajectories equals global mean step CE + lambda mean pair loss.
            loss=ce*len(seqs)/len(training)+weight*rank*len(seqs)/npairs
            if not torch.isfinite(loss):raise ValueError('Nonfinite trajectory loss')
            loss.backward()
            if not grad_checked:
                norms={kind:sum(p.grad.detach().float().square().sum().item() for name,p in model.named_parameters()
                    if p.grad is not None and match(name))**.5 for kind,match in
                    {'head':lambda x:x.startswith('head.'),'lora':lambda x:'lora_' in x}.items()}
                assert all(v>0 for v in norms.values())
                assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
                write_json(folder/'gradient_audit.json',norms);grad_checked=True
            torch.nn.utils.clip_grad_norm_(params,cfg['gradient_clip'],error_if_nonfinite=True)
            optimizer.step();ce_sum+=ce.item();rank_sum+=rank.item()
            schedule.append({'epoch':epoch,'trajectory':key,'keys':[r['key'] for r in seq]})
            if (n+1)%8==0:write_json(folder/'status.json',{'phase':'training','epoch':epoch,
                'trajectories':n+1,'total':len(order),'mean_step_ce':ce_sum/sum(len(seqs[k]) for k in order[:n+1])})
        predictions,val=predict(model,tok,validation,cfg,folder,f'validation_epoch{epoch}')
        if val['rejections']:raise ValueError('Synthetic validation rejection')
        memory=describe_memory(validation,predictions,allpairs,mode,model.head.temperature,cfg['alpha'],folder/f'validation_memory_epoch{epoch}.jsonl')
        write_json(folder/f'epoch{epoch}.json',{'train_ce':ce_sum/len(training),
            'train_rank':rank_sum/npairs,'validation':val,'memory_diagnostics':memory})
        if val['mean_ce']<best:
            best=val['mean_ce'];selected=epoch;save_checkpoint(cp,tok,model,folder/'checkpoint',cfg,arm,epoch)
    write_json(folder/'schedule.json',schedule)
    after=frozen_hash(model);assert after==env['frozen_initial']
    write_json(folder/'frozen_audit.json',{'passed':True,'hash':after})
    del optimizer,encs,model;gc.collect();torch.cuda.empty_cache()
    cp2,tok2,reloaded=load(folder/'checkpoint')
    predictions,_=predict(reloaded,tok2,validation[:4],cfg,folder,'reload_probe')
    expected={r['key']:r for r in read_jsonl(folder/f'validation_epoch{selected}.jsonl')}
    error=max(abs(x-y) for r in predictions for x,y in zip(r['logits'],expected[r['key']]['logits']))
    assert error<=1e-5 and reloaded.head.temperature==env['temperature']
    write_json(folder/'checkpoint_audit.json',{'selected_epoch':selected,'validation_ce':best,'reload_error':error,
        'adapter_sha256':cp2.weights_sha256(),'head_sha256':sha256(cp2.file('head.pt')),
        'max_cuda_allocated':torch.cuda.max_memory_allocated()})
    del reloaded;gc.collect();torch.cuda.empty_cache()


def evaluate_arm(root,arm):
    import torch
    cfg=read_json(root/'protocol.json');folder=root/'full'/arm
    cp,tok,model=load(folder/'checkpoint');mode,_=cfg['arms'][arm]
    rows=read_json(root/'records.json');pairs=read_json(root/'pairs.json')
    write_json(folder/'evaluation_environment.json',model_environment(cp,model))
    for split in ('test','external_test'):
        subset=[r for r in rows if r['split']==split]
        predictions,_=predict(model,tok,subset,cfg,folder,split)
        if split=='test':
            memory=describe_memory(subset,predictions,pairs,mode,model.head.temperature,cfg['alpha'],folder/'test_memory.jsonl')
            write_json(folder/'test_memory.json',memory)
    del model;gc.collect();torch.cuda.empty_cache()


def run(root,stage):
    cfg=read_json(root/'protocol.json')
    if stage=='full' and read_json(root/'pilot_gate.json')['status']!='passed':raise ValueError('Pilot gate closed')
    (root/stage).mkdir(exist_ok=False)
    audit_math(root)
    def launch(phase):
        workers=[]
        for gpu,arm in enumerate(cfg['arms']):
            log=(root/stage/f'{arm}_{phase}.log').open('w')
            p=subprocess.Popen([sys.executable,'-u','-m',__package__+'.risk_memory',phase,
                '--root',str(root),'--stage',stage,'--arm',arm],
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu)),stdout=log,stderr=subprocess.STDOUT)
            log.close();workers.append((arm,p))
        write_json(root/stage/f'{phase}_workers.json',{a:p.pid for a,p in workers})
        failed=[(a,p.wait()) for a,p in workers]
        if any(code for _,code in failed):raise RuntimeError(f'{phase} failures: {failed}')
    launch('train')
    envs=[read_json(root/stage/a/'environment.json') for a in cfg['arms']]
    assert all(e==envs[0] for e in envs)
    assert len({sha256(root/stage/a/'schedule.json') for a in cfg['arms']})==1
    assert all(read_json(root/stage/a/'frozen_audit.json')['passed'] for a in cfg['arms'])
    write_json(root/stage/'matched_audit.json',{'passed':True})
    if stage=='pilot':write_json(root/'pilot_gate.json',{'status':'awaiting_review','technical_passed':True})
    else:
        write_json(root/stage/'frozen_before_evaluation.json',{
            a:read_json(root/stage/a/'checkpoint_audit.json') for a in cfg['arms']})
        launch('evaluate')
    write_json(root/stage/'status.json',{'phase':'complete'})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=('prepare','audit','train','evaluate','run'))
    p.add_argument('--root',type=Path,required=True);p.add_argument('--reference',type=Path)
    p.add_argument('--stage',choices=('pilot','full'));p.add_argument('--arm',choices=tuple(ARMS))
    a=p.parse_args()
    if a.phase=='prepare':return prepare(a.root,a.reference)
    cfg=read_json(a.root/'protocol.json')
    import kev
    from kev.api import SystemOneRequest,to_record
    assert (Path(kev.__file__).parent.parent/'SOURCE_COMMIT').read_text().strip()==cfg['source_commit']
    assert cfg['question']==QUESTION
    _,mapping=to_record(SystemOneRequest.model_validate(request('mapping')))
    assert mapping[0]['keys']==['false','true']
    assert cfg['records_sha256']==sha256(a.root/'records.json')
    assert cfg['pairs_sha256']==sha256(a.root/'pairs.json')
    assert cfg['code_sha256']=={p.name:sha256(p) for p in Path(__file__).parent.glob('*.py')}
    if a.phase=='audit':audit_math(a.root)
    elif a.phase=='train':train(a.root,a.stage,a.arm)
    elif a.phase=='evaluate':evaluate_arm(a.root,a.arm)
    else:run(a.root,a.stage)


if __name__=='__main__':main()
