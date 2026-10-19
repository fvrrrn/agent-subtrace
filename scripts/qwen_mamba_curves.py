"""Six fixed token-curve arms on the original controlled Kev training split."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

from qwen_mamba_pilot import BASE, REVISION, parameter_hash, sha, write

CURVES = ('exponential', 'cosine', 'cosine_memory')
ARMS = [f'{curve}_{suffix}' for curve in CURVES for suffix in ('plain', 'mono')]
SEED = 20261017


def read(path):
    return json.loads(Path(path).read_text())


def prepare(root, reference):
    root.mkdir(parents=True, exist_ok=False)
    assert sha(reference/'records.json') == read(reference/'protocol.json')['records_sha256']
    shutil.copy2(reference/'records.json', root/'records.json')
    refs = root/'kev_reference';refs.mkdir()
    full = reference/'full'
    for split in ('test', 'external_test'):
        shutil.copy2(full/f'baseline_{split}.jsonl', refs/f'original_{split}.jsonl')
        for arm in ('noul', 'unsafe_up', 'bidirectional'):
            for seed in (20261017, 20261018, 20261019):
                shutil.copy2(full/f'{arm}_{seed}'/f'{split}.jsonl', refs/f'{arm}_{seed}_{split}.jsonl')
    write(root/'protocol.json', {
        'base':BASE, 'revision':REVISION, 'seed':SEED, 'arms':ARMS,
        'records_sha256':sha(root/'records.json'),
        'source_sha256':{p.name:sha(p) for p in (Path(__file__), Path(__file__).with_name('qwen_mamba_pilot.py'))},
        'reference_sha256':{p.name:sha(p) for p in refs.iterdir()},
        'epochs':8, 'lr':3e-4, 'weight_decay':.01, 'batch_trajectories':4,
        'train_limit':8192, 'eval_limit':65536, 'beta':.8, 'kappa':math.log(5), 'mono_weight':.1,
        'architecture':'Original frozen Qwen3.5-0.8B-Base token features, no Kev or pilot weights. Same fresh 696985-parameter head as StepGuard pilot: LN/1024->256, residual Mamba2(state64,expand2,headdim64,chunk128,fused), LN/linear. Shared exact initialization for all six arms.',
        'rendering':'One chronological sequence per synthetic trajectory, tool definitions and original user request once, role-tagged observations and actions. Independently tokenized segments guarantee stable prefix boundaries. At each action endpoint only its observed history and current proposed action are visible. Full-sequence extraction uses causal Qwen; future perturbation and prefix re-encoding audited. External history strings remain verbatim; they are not split using guessed labels.',
        'supervision':'Only annotated action-content tokens receive curve BCE; context/role-marker/tool/user tokens have zero supervised weight but carry state and gradients. Synthetic labels are original step labels; no labels rendered. Curve state starts at zero for each trajectory and holds across unannotated observations. No reset between independent branches inside a trajectory.',
        'curves':{'exponential':'a+(y-a)*(1-exp(-ln(5)*u))', 'cosine':'a+(y-a)*(1-cos(pi*u))/2', 'cosine_memory':'a+0.8*(y-a)*(1-cos(pi*u))/2'},
        'monotonicity':'mean relu(p[t-1]-p[t]) over action-token transitions whose action label is unsafe, including first action token vs preceding role-marker token. Non-decreasing, no margin, no ranking across separate unsafe actions. Coefficient .1; plain coefficient 0. Same calculation in all arms.',
        'loss_normalization':'Curve BCE averaged within each action, then across dataset actions. Monotonic loss averaged within each unsafe action, then across dataset unsafe actions. Whole-trajectory mini-batches use fixed global normalizers; no token-length domination or detached Mamba state.',
        'selection':'Eight fixed epochs; checkpoint minimizes hard binary endpoint BCE on synthetic validation, never curve fit or StepGuard. All six checkpoints frozen before final synthetic test/StepGuard. No sweep beyond the six specified conditions.',
        'comparison':'Identical original synthetic splits (1776/488/3904 steps), external StepGuard test 3017 actions; external_dev excluded. Compare all common successful keys to saved Kev original and three trained arms, all three old seeds reported separately. Mamba has one seed. No training on StepGuard and no pilot checkpoint transfer.',
        'limitations':['Prior StepGuard pilot trained on a subset of this previously viewed pool, but its weights are not reused. This is exploratory, not untouched confirmation.',
            'Soft targets use full action label and length, not genuine causal token gold. Identical action prefixes can have different future-dependent targets.',
            'Safe-action decay is a stipulated alarm prior, not verified threat remediation. Endpoint classification and curve compliance must be reported separately.',
            'One scalar risk and a recurrent state do not establish separation of attack vectors or interpretability.',
            'Qwen already contextualizes tokens; no isolated demonstration that Mamba memory is responsible.',
            'Same data as Kev, different architecture, rendering, training objective and epoch count; no equal-compute claim.']})


def serial(value):
    return value if isinstance(value,str) else json.dumps(value,ensure_ascii=False,separators=(',',':'))


def documents(rows):
    grouped=defaultdict(list)
    for row in rows:
        if row['split'] in ('train','validation','test'):grouped[row['trajectory']].append(row)
        elif row['split']=='external_test':grouped[row['key']].append(row)
    output=[]
    from graph_safety.step_training import QUESTION
    for key,seq in sorted(grouped.items()):
        seq=sorted(seq,key=lambda r:r.get('position',0))
        first=json.loads(seq[0]['state']);last=json.loads(seq[-1]['state'])
        chunks=[(QUESTION['instructions']+'\n[TOOL INTERFACES]\n'+serial(first['tool_interfaces'])+
            '\n[USER REQUEST]\n'+serial(first['user_request'])+'\n',None)]
        if seq[0]['split']=='external_test':
            chunks.append(('[OBSERVED HISTORY]\n'+serial(first['observed_history'])+'\n',None))
            chunks.append(('[AGENT ACTION]\n',None))
            chunks.append((serial(first['proposed_action']),seq[0]))
        else:
            events=last['observed_history']+[{'role':'agent','action':last['proposed_action']}]
            by_position={}
            for r in seq:
                s=json.loads(r['state']);n=len(s['observed_history'])
                assert s['tool_interfaces']==first['tool_interfaces'] and s['user_request']==first['user_request']
                assert events[:n]==s['observed_history']
                assert events[n]=={'role':'agent','action':s['proposed_action']}
                assert n not in by_position
                by_position[n]=r
            assert all(events[p]['role']=='agent' for p in by_position)
            for pos,event in enumerate(events):
                if event['role']=='agent':
                    chunks.append(('[AGENT ACTION]\n',None))
                    chunks.append((serial(event['action']),by_position.get(pos)))
                    chunks.append(('\n',None))
                else:chunks.append(('['+event['role'].upper()+']\n'+serial({k:v for k,v in event.items() if k!='role'})+'\n',None))
            if chunks[-1]==('\n',None):chunks.pop()
        output.append({'key':key,'split':seq[0]['split'],'chunks':chunks})
    return output


def make_head():
    import torch.nn as nn
    from mamba_ssm import Mamba2
    class Head(nn.Module):
        def __init__(self):
            super().__init__()
            self.project=nn.Sequential(nn.LayerNorm(1024),nn.Linear(1024,256))
            self.mamba=Mamba2(d_model=256,d_state=64,expand=2,headdim=64,chunk_size=128,use_mem_eff_path=True)
            self.readout=nn.Sequential(nn.LayerNorm(256),nn.Linear(256,1))
        def forward(self,x):
            x=self.project(x)
            return self.readout(x+self.mamba(x)).squeeze(-1)
    return Head().cuda()


def curve_targets(spans, curve, device):
    import torch
    previous=0.;result=[]
    for span in spans:
        u=torch.arange(1,span['end']-span['start']+1,device=device,dtype=torch.float32)/(span['end']-span['start'])
        if curve=='exponential':fraction=1-torch.exp(-math.log(5)*u)
        else:fraction=(1-torch.cos(math.pi*u))/2*(.8 if curve=='cosine_memory' else 1.)
        targets=previous+(span['label']-previous)*fraction
        result.append(targets)
        previous=span['label']+(previous-span['label'])*(.2 if curve!='cosine' else 0.)
    return result


def cache(root):
    import torch
    import transformers
    import mamba_ssm
    from transformers import AutoTokenizer,Qwen3_5TextModel
    from tqdm.auto import tqdm
    cfg=read(root/'protocol.json');rows=read(root/'records.json')
    folder=root/'features';folder.mkdir(exist_ok=False)
    tokenizer=AutoTokenizer.from_pretrained(BASE,revision=REVISION,local_files_only=True)
    backbone=Qwen3_5TextModel.from_pretrained(BASE,revision=REVISION,local_files_only=True,
        dtype=torch.bfloat16,attn_implementation='sdpa').cuda().eval().requires_grad_(False)
    before=parameter_hash(backbone);index=[];rejects=[];audits=[];started=time.monotonic()
    write(root/'environment.json',{'torch':torch.__version__,'transformers':transformers.__version__,
        'mamba':mamba_ssm.__version__,'gpu':torch.cuda.get_device_name(),'base_hash':before,'base_revision':REVISION})
    for doc in tqdm(documents(rows),desc='Qwen causal token cache',mininterval=5):
        ids=[];spans=[];texts=[]
        for text,row in doc['chunks']:
            start=len(ids);ids.extend(tokenizer.encode(text,add_special_tokens=False));texts.append(text)
            if row:
                assert len(ids)>start
                spans.append({'start':start,'end':len(ids),'key':row['key'],'label':row['label'],
                    'group':row['group'],'source_state_sha256':hashlib.sha256(row['state'].encode()).hexdigest(),
                    'prefix_ids_sha256':hashlib.sha256(json.dumps(ids).encode()).hexdigest()})
        limit=cfg['train_limit'] if doc['split']=='train' else cfg['eval_limit']
        if len(ids)>limit:
            rejects.append({'trajectory':doc['key'],'split':doc['split'],'tokens':len(ids),'keys':[s['key'] for s in spans],'reason':'context_overflow'});continue
        tensor=torch.tensor([ids],device='cuda')
        with torch.inference_mode():hidden=backbone(input_ids=tensor,use_cache=False).last_hidden_state[0]
        assert torch.isfinite(hidden).all()
        if doc['split']=='train' and len(audits)<2 and len(spans)>1:
            cut=spans[0]['end']
            with torch.inference_mode():
                short=backbone(input_ids=tensor[:,:cut],use_cache=False).last_hidden_state[0]
            rel=(hidden[:cut].float()-short.float()).norm()/short.float().norm()
            similarity=torch.nn.functional.cosine_similarity(hidden[cut-1].float(),short[-1].float(),dim=0)
            audits.append({'key':doc['key'],'cut':cut,'relative_l2':rel.item(),'endpoint_cosine':similarity.item()})
            assert rel<.03 and similarity>.999
        filename=hashlib.sha256(doc['key'].encode()).hexdigest()+'.pt'
        torch.save({'hidden':hidden.cpu().clone(),'input_ids':torch.tensor(ids),'text':''.join(texts)},folder/filename)
        index.append({'key':doc['key'],'split':doc['split'],'tokens':len(ids),'spans':spans,
            'file':filename,'sha256':sha(folder/filename)})
    assert parameter_hash(backbone)==before
    assert all(p.grad is None and not p.requires_grad for p in backbone.parameters())
    write(root/'index.json',index);write(root/'rejections.json',rejects)
    write(root/'encoder_audit.json',{'frozen_unchanged':True,'prefix_audits':audits,
        'seconds':time.monotonic()-started,'actions':dict(Counter(d['split'] for d in index for s in d['spans'])),
        'max_tokens':max(d['tokens'] for d in index),'no_truncation':True})
    del backbone,hidden,tensor;torch.cuda.empty_cache()
    torch.manual_seed(SEED);model=make_head();torch.save(model.state_dict(),root/'initial_head.pt')
    x=torch.randn(1,256,1024,device='cuda',dtype=torch.bfloat16)
    with torch.autocast('cuda',dtype=torch.bfloat16):z=model(x)
    z[0,127].float().square().backward()
    norms={n:p.grad.float().norm().item() for n,p in model.named_parameters() if p.grad is not None}
    assert any(v>0 for n,v in norms.items() if n.startswith('mamba.'))
    assert all(math.isfinite(v) for v in norms.values())
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        altered=x.clone();altered[:,128:]=torch.randn_like(altered[:,128:]);other=model(altered)
    error=(z[:,:128]-other[:,:128]).abs().max().item();assert error<.02
    toy=[{'start':0,'end':5,'label':y} for y in (0,1,0,0,1)]
    endpoints={c:[float(t[-1]) for t in curve_targets(toy,c,'cpu')] for c in CURVES}
    assert max(abs(a-b) for a,b in zip(endpoints['cosine_memory'],[0,.8,.16,.032,.8064]))<1e-6
    assert endpoints['cosine']==[0,1,0,0,1]
    assert max(abs(a-b) for a,b in zip(endpoints['exponential'],endpoints['cosine_memory']))<1e-6
    write(root/'gate.json',{'status':'passed','future_perturbation_error':error,'gradient_norms':norms,
        'curve_endpoint_audit':endpoints,'initial_head_sha256':sha(root/'initial_head.pt'),
        'trainable_parameters':sum(p.numel() for p in model.parameters()),'encoder_prefix_audited':True})


def load_features(root,docs):
    import torch
    result={}
    for d in docs:
        p=root/'features'/d['file'];assert sha(p)==d['sha256']
        result[d['key']]=torch.load(p,weights_only=True,map_location='cpu')['hidden']
    return result


def batch(root,docs,features=None):
    import torch
    n=((max(d['tokens'] for d in docs)+127)//128)*128
    x=torch.zeros(len(docs),n,1024,device='cuda',dtype=torch.bfloat16)
    for i,d in enumerate(docs):
        h=features[d['key']] if features is not None else torch.load(root/'features'/d['file'],weights_only=True,map_location='cpu')['hidden']
        x[i,:d['tokens']]=h.to('cuda')
    return x


def predict(root,model,docs,folder,name,features=None,curve=None,token_traces=False):
    import torch
    import torch.nn.functional as F
    from graph_safety.metrics import evaluate
    from tqdm.auto import tqdm
    cfg=read(root/'protocol.json');model.eval();out=[];total=0.;curve_sum=0.;pairs=0;falls=0;curve_actions=0
    traces=(folder/f'{name}_tokens.jsonl').open('w') if token_traces else None
    with torch.no_grad():
        for start in tqdm(range(0,len(docs),cfg['batch_trajectories']),desc=f'{folder.name}/{name}',mininterval=5):
            group=docs[start:start+cfg['batch_trajectories']];x=batch(root,group,features)
            with torch.autocast('cuda',dtype=torch.bfloat16):logits=model(x)
            for i,d in enumerate(group):
                targets=curve_targets(d['spans'],curve,'cuda') if curve and d['split']!='external_test' else None
                for j,s in enumerate(d['spans']):
                    z=logits[i,s['end']-1].float();p=z.sigmoid().item()
                    total+=F.binary_cross_entropy_with_logits(z,torch.tensor(float(s['label']),device='cuda')).item()
                    out.append({'key':s['key'],'label':s['label'],'score':p,'logit':z.item(),
                        'source_state_sha256':s['source_state_sha256'],'prefix_ids_sha256':s['prefix_ids_sha256']})
                    if targets is not None:
                        action=logits[i,s['start']:s['end']].float().sigmoid()
                        curve_sum+=(action-targets[j]).square().mean().item();curve_actions+=1
                        if s['label']:
                            previous=logits[i,s['start']-1:s['end']-1].float().sigmoid()
                            pairs+=len(action);falls+=int((action<previous).sum())
                if traces:traces.write(json.dumps({'trajectory':d['key'],'scores':logits[i,:d['tokens']].float().sigmoid().cpu().tolist(),'spans':d['spans']})+'\n')
    if traces:traces.close()
    metrics=evaluate([r['label'] for r in out],[r['score'] for r in out]);metrics.pop('pr_curve')
    metrics['endpoint_bce']=total/len(out)
    metrics['curve_diagnostics']={'action_mean_mse':curve_sum/curve_actions if curve_actions else None,
        'unsafe_token_transitions':pairs,'decreasing_transitions':falls,'not_token_accuracy':True}
    write(folder/f'{name}.json',metrics)
    (folder/f'{name}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in out))
    return metrics,out


def train(root,arm):
    import torch
    import torch.nn.functional as F
    from tqdm.auto import tqdm
    cfg=read(root/'protocol.json');assert read(root/'gate.json')['status']=='passed'
    folder=root/'arms'/arm;folder.mkdir(parents=True,exist_ok=False)
    curve,suffix=arm.rsplit('_',1);weight=cfg['mono_weight'] if suffix=='mono' else 0.
    docs=read(root/'index.json');training=[d for d in docs if d['split']=='train'];validation=[d for d in docs if d['split']=='validation']
    features=load_features(root,training+validation)
    torch.manual_seed(SEED);random.seed(SEED);model=make_head()
    model.load_state_dict(torch.load(root/'initial_head.pt',weights_only=True));initial_hash=parameter_hash(model)
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'],fused=True)
    actions=sum(len(d['spans']) for d in training);unsafe=sum(s['label'] for d in training for s in d['spans'])
    target_cache={d['key']:curve_targets(d['spans'],curve,'cuda') for d in training}
    best=float('inf');selected=None;history=[];schedule=[];started=time.monotonic();gradient_checked=False
    for epoch in range(1,cfg['epochs']+1):
        model.train();order=list(training);random.Random(SEED+epoch).shuffle(order);ce_sum=mono_sum=0.
        for offset in tqdm(range(0,len(order),cfg['batch_trajectories']),desc=f'{arm} epoch{epoch}/{cfg["epochs"]}',mininterval=5):
            group=order[offset:offset+cfg['batch_trajectories']];x=batch(root,group,features)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):logits=model(x)
            ce=[];mono=[]
            for i,d in enumerate(group):
                for j,s in enumerate(d['spans']):
                    z=logits[i,s['start']:s['end']].float()
                    ce.append(F.binary_cross_entropy_with_logits(z,target_cache[d['key']][j]))
                    if s['label']:
                        prior=logits[i,s['start']-1:s['end']-1].float().sigmoid()
                        mono.append(F.relu(prior-z.sigmoid()).mean())
            c=torch.stack(ce).sum();m=torch.stack(mono).sum() if mono else logits.sum()*0
            loss=len(training)/len(group)*(c/actions+weight*m/unsafe)
            assert torch.isfinite(loss)
            loss.backward()
            if not gradient_checked:
                norm=sum(p.grad.float().square().sum().item() for p in model.mamba.parameters() if p.grad is not None)**.5
                assert norm>0 and math.isfinite(norm)
                write(folder/'gradient_audit.json',{'mamba_gradient_norm':norm});gradient_checked=True
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
            ce_sum+=c.item();mono_sum+=m.item();schedule.append({'epoch':epoch,'trajectories':[d['key'] for d in group]})
        metrics,_=predict(root,model,validation,folder,f'validation_epoch{epoch}',features,curve)
        history.append({'epoch':epoch,'train_curve_bce':ce_sum/actions,'train_monotonic_loss':mono_sum/unsafe,'validation':metrics})
        write(folder/'history.json',history)
        if metrics['endpoint_bce']<best:
            best=metrics['endpoint_bce'];selected=epoch
            torch.save(model.state_dict(),folder/'checkpoint.pt')
    model.load_state_dict(torch.load(folder/'checkpoint.pt',weights_only=True))
    _,probes=predict(root,model,validation[:cfg['batch_trajectories']],folder,'reload_probe',features,curve)
    expected={r['key']:r for r in map(json.loads,(folder/f'validation_epoch{selected}.jsonl').read_text().splitlines())}
    error=max(abs(r['logit']-expected[r['key']]['logit']) for r in probes)
    # Repeat exactly the original first validation batch, including padding.
    assert error==0
    write(folder/'schedule.json',schedule)
    write(folder/'selection.json',{'epoch':selected,'validation_bce':best,'checkpoint_sha256':sha(folder/'checkpoint.pt'),
        'initial_parameter_hash':initial_hash,'initial_file_sha256':sha(root/'initial_head.pt'),
        'reload_max_logit_error':error,'seconds':time.monotonic()-started,'curve':curve,'mono_weight':weight})


def evaluate_arm(root,arm):
    import torch
    folder=root/'arms'/arm;selection=read(folder/'selection.json')
    assert selection['checkpoint_sha256']==sha(folder/'checkpoint.pt')
    assert (root/'all_checkpoints_frozen_before_test.json').exists()
    model=make_head();model.load_state_dict(torch.load(folder/'checkpoint.pt',weights_only=True))
    docs=read(root/'index.json');curve=arm.rsplit('_',1)[0]
    for split in ('test','external_test'):
        predict(root,model,[d for d in docs if d['split']==split],folder,split,curve=curve,token_traces=split=='test')
    write(folder/'status.json',{'phase':'complete'})


def compare(root):
    from graph_safety.metrics import evaluate
    original={r['key']:r for r in read(root/'records.json')}
    results={}
    for split in ('test','external_test'):
        collections={}
        for arm in ARMS:
            collections['mamba_'+arm]={r['key']:r for r in map(json.loads,(root/'arms'/arm/f'{split}.jsonl').read_text().splitlines())}
        for p in (root/'kev_reference').glob(f'*_{split}.jsonl'):
            # test glob also matches external_test; filter explicitly.
            if split=='test' and p.stem.endswith('_external_test'):continue
            collections['kev_'+p.stem[:-len('_'+split)]]={r['key']:r for r in map(json.loads,p.read_text().splitlines()) if r.get('status','complete')=='complete'}
        keys=sorted(set.intersection(*(set(v) for v in collections.values())))
        assert keys
        labels=[collections['mamba_'+ARMS[0]][k]['label'] for k in keys]
        metrics={}
        for name,rr in collections.items():
            assert [rr[k]['label'] for k in keys]==labels
            if name.startswith('mamba_'):
                assert all(rr[k]['source_state_sha256']==hashlib.sha256(original[k]['state'].encode()).hexdigest() for k in keys)
                assert all(rr[k]['prefix_ids_sha256']==collections['mamba_'+ARMS[0]][k]['prefix_ids_sha256'] for k in keys)
            m=evaluate(labels,[rr[k]['score'] for k in keys]);m.pop('pr_curve');metrics[name]=m
        results[split]={'common_keys':keys,'n':len(keys),'models':metrics,
            'missing_from_common':{name:sorted(set(rr)-set(keys)) for name,rr in collections.items()}}
    write(root/'comparison.json',{'splits':results,'rejections':read(root/'rejections.json'),
        'selection':{a:read(root/'arms'/a/'selection.json') for a in ARMS},'limits':read(root/'protocol.json')['limitations']})
    print(json.dumps({split:{name:{k:m[k] for k in ('average_precision','roc_auc')} for name,m in data['models'].items()} for split,data in results.items()},indent=2),flush=True)


def run(root):
    cfg=read(root/'protocol.json');assert sha(root/'records.json')==cfg['records_sha256']
    root.joinpath('arms').mkdir(exist_ok=False)
    cache(root)
    def stage(phase):
        workers=[]
        for gpu,curve in enumerate(CURVES):
            log=(root/f'{curve}_{phase}.log').open('w')
            p=subprocess.Popen([sys.executable,'-u',__file__,'queue','--root',str(root),'--curve',curve,'--stage',phase],
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu)),stdout=log,stderr=subprocess.STDOUT)
            log.close();workers.append((curve,p))
        write(root/f'{phase}_workers.json',{c:p.pid for c,p in workers})
        codes={c:p.wait() for c,p in workers}
        if any(codes.values()):raise RuntimeError(f'{phase}: {codes}')
    stage('train')
    selection={a:read(root/'arms'/a/'selection.json') for a in ARMS}
    assert len({s['initial_parameter_hash'] for s in selection.values()})==1
    assert len({sha(root/'arms'/a/'schedule.json') for a in ARMS})==1
    write(root/'all_checkpoints_frozen_before_test.json',selection)
    stage('evaluate');compare(root);write(root/'status.json',{'phase':'complete'})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=('prepare','run','queue','train','evaluate','compare'))
    p.add_argument('--root',type=Path,required=True);p.add_argument('--reference',type=Path)
    p.add_argument('--curve',choices=CURVES);p.add_argument('--arm',choices=ARMS);p.add_argument('--stage',choices=('train','evaluate'))
    a=p.parse_args()
    if a.phase=='prepare':return prepare(a.root,a.reference)
    cfg=read(a.root/'protocol.json')
    assert cfg['source_sha256']=={p.name:sha(p) for p in (Path(__file__),Path(__file__).with_name('qwen_mamba_pilot.py'))}
    assert cfg['records_sha256']==sha(a.root/'records.json')
    assert cfg['reference_sha256']=={p.name:sha(p) for p in (a.root/'kev_reference').iterdir()}
    if a.phase=='queue':
        for suffix in ('plain','mono'):
            subprocess.run([sys.executable,'-u',__file__,a.stage,'--root',str(a.root),'--arm',f'{a.curve}_{suffix}'],check=True)
    elif a.phase=='run':run(a.root)
    elif a.phase=='train':train(a.root,a.arm)
    elif a.phase=='evaluate':evaluate_arm(a.root,a.arm)
    else:compare(a.root)


if __name__=='__main__':main()
