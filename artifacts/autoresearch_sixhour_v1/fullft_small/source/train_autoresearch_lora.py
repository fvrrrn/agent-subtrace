"""Budgeted whole-dialog LoRA comparison with simultaneous token/event supervision."""
import argparse
import json
import math
from pathlib import Path
import random
import time

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

import train_whole_dialogs as original
from train_curve_event_heads import Head, profile
from train_autoresearch_heads import event_logit, seed
from prepare_whole_dialogs import BASE, REVISION, digest, write
from graph_safety.metrics import evaluate


def read(path):
    return json.loads(path.read_text())


class BudgetStop(Exception):
    pass


def check_budget(args):
    # Leave time for checkpoint persistence and the frozen-weight hash audit.
    if time.time() >= args.deadline_unix - 180:
        raise BudgetStop('deadline safety reserve reached (180 seconds)')


def event(z, null, span, readout):
    return event_logit(z, null, span, readout)


def losses(z, null, row, args):
    tokens, events = [], []
    for span in row['spans']:
        values = z[span['start']:span['end']]
        tokens.append(F.binary_cross_entropy_with_logits(values,
            span['label']*profile(len(values), z.device, args.curve)))
        e = event(z, null, span, args.readout)
        events.append(F.binary_cross_entropy_with_logits(e, e.new_tensor(float(span['label']))))
    return torch.stack(tokens).mean(), torch.stack(events).mean()


def forward(model, head, row):
    hidden = original.encode(model, row)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        z, null = head(hidden)
    return z[0, :row['tokens']].float(), null[0, :row['tokens']].float() if null is not None else None


def validate(model, head, rows, root, name, args):
    model.eval(); head.eval()
    labels, scores, ces, curves = [], [], [], []
    folder = root / name
    folder.mkdir(exist_ok=False)
    (folder / 'tokens').mkdir()
    with torch.no_grad(), (folder / 'predictions.jsonl').open('w') as stream:
        for row in tqdm(rows, desc=name, mininterval=5):
            check_budget(args)
            z, null = forward(model, head, row)
            c, _ = losses(z, null, row, args)
            curves.append(c.item())
            torch.save({'key': row['key'], 'ids_sha256': row['ids_sha256'], 'raw_logits': z.cpu(),
                'null_logits': null.cpu() if null is not None else None, 'spans': row['spans']},
                folder / 'tokens' / (row['key']+'.pt'))
            for span in row['spans']:
                e = event(z, null, span, args.readout)
                score = e.sigmoid().item()
                labels.append(span['label']); scores.append(score)
                ces.append(F.binary_cross_entropy_with_logits(e,e.new_tensor(float(span['label']))).item())
                stream.write(json.dumps({'key': span['key'], 'dialog': row['key'], 'label': span['label'],
                    'score': score, 'event_logit': e.item(), 'state_sha256': span['state_sha256'],
                    'ids_sha256': row['ids_sha256'], 'sources': row['sources']})+'\n')
    check_budget(args)
    metrics = evaluate(labels, scores)
    pr = metrics.pop('pr_curve')
    metrics['event_bce'] = sum(ces)/len(ces)
    metrics['token_bce'] = sum(curves)/len(curves)
    for fpr in (.01, .05):
        metrics[f'recall_at_fpr_{fpr}'] = max((p['recall'] for p in pr
            if p['recall'] is not None and p.get('specificity') is not None and 1-p['specificity'] <= fpr+1e-12), default=0.)
    write(folder / 'metrics.json',metrics)
    return metrics


def main(args):
    args.root.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    model = head = None
    frozen_before = None
    history = []; best = None
    epoch = 0; dialogs_done = 0; applied_dialogs = 0
    status = {'status':'initializing'}
    try:
        check_budget(args)
        assert args.epochs > 0 and args.event_weight > 0 and math.isfinite(args.event_weight)
        rows, audit = original.load_data(args.data)
        splits = read(args.splits)
        if 'splits' in splits:
            splits = splits['splits']
        assert args.train_split in ('train_small','train_full')
        bykey = {r['key']: r for r in rows}
        training = [bykey[k] for k in splits[args.train_split]]
        validation = [bykey[k] for k in splits['dev_select']]
        assert training and validation
        assert len({r['key'] for r in training}) == len(training)
        assert len({r['key'] for r in validation}) == len(validation)
        assert not ({r['group'] for r in training} & {r['group'] for r in validation})
        for name in ('dev_confirm','original_validation'):
            assert not ({r['group'] for r in training} & {bykey[k]['group'] for k in splits[name]})
        for row in training+validation:
            assert digest(row['ids']) == row['ids_sha256'] and len(row['ids']) == row['tokens'] <= 16384
            assert row['spans']
            previous_end = 0
            for span in row['spans']:
                assert previous_end <= span['start'] < span['end'] <= row['tokens']
                assert span['label'] in (0, 1)
                previous_end = span['end']
        check_budget(args)
        seed(original.SEED)
        head = Head('curve_null' if args.readout == 'null' else 'curve_linear_softmax', args.attention).cuda()
        model = original.backbone(True)
        trainable = {name:p for name,p in model.named_parameters() if p.requires_grad}
        assert trainable and all('lora_' in name for name in trainable), list(trainable)
        assert all(p.requires_grad for name,p in model.named_parameters() if 'lora_' in name)
        frozen_before = original.frozen_hash(model)
        protocol = {**vars(args), 'data_sha256': audit['dialogs_sha256'], 'split_sha256': original.file_hash(args.splits),
            'source_sha256': original.file_hash(Path(__file__)), 'base': BASE, 'revision': REVISION,
            'dependency_sha256': {Path(m.__file__).name:original.file_hash(Path(m.__file__))
                for m in (original, __import__('train_curve_event_heads'), __import__('prepare_whole_dialogs'), __import__('train_autoresearch_heads'))},
            'template_sha256': audit['template_sha256'], 'seed': original.SEED, 'frozen_weights_sha256': frozen_before,
            'trainable_backbone_names':list(trainable),
            'lora': {'rank':16,'alpha':32,'dropout':0.,'target':'all-linear','gradient_checkpointing':'non-reentrant'},
            'torch':str(torch.__version__),'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(),
            'train_dialogs':len(training),'dev_dialogs':len(validation),'no_truncation':True,
            'deadline_reserve_seconds':180, 'stress_input':'longest training dialog only',
            'objective':'mean_action(token_curve_BCE+event_weight*hard_event_BCE), accumulation4 whole dialogs'}
        write(args.root / 'protocol.json', {k:str(v) if isinstance(v,Path) else v for k,v in protocol.items()})
        check_budget(args)
        probe = next(r for r in training if r['tokens'] < 4096 and any(s['label'] for s in r['spans']))
        model.eval(); head.eval()
        cut = min(probe['spans'][0]['end'], probe['tokens']//2)
        assert 0 < cut < probe['tokens']
        changed = dict(probe)
        # Change only future input IDs, never using labels or held-out inputs.
        vocab_size = model.get_input_embeddings().num_embeddings
        changed['ids'] = probe['ids'][:cut]+[(i+1)%vocab_size for i in probe['ids'][cut:]]
        with torch.no_grad():
            z,n = forward(model,head,probe)
            zz,nn = forward(model,head,changed)
            causal_error = (z[:cut]-zz[:cut]).abs().max().item()
            null_causal_error = (n[:cut]-nn[:cut]).abs().max().item() if n is not None else 0.
            assert causal_error == null_causal_error == 0
            del zz,nn
            check_budget(args)
            from peft import get_peft_model_state_dict, set_peft_model_state_dict
            gate_folder = args.root/'gate_reload'; gate_folder.mkdir()
            torch.save(head.state_dict(),gate_folder/'head.pt')
            model.save_pretrained(gate_folder/'adapter',safe_serialization=False)
            adapter = torch.load(gate_folder/'adapter'/'adapter_model.bin',map_location='cpu',weights_only=True)
            head.load_state_dict(torch.load(gate_folder/'head.pt',map_location='cpu',weights_only=True),strict=True)
            result = set_peft_model_state_dict(model,adapter)
            assert not result.unexpected_keys
            assert not any('lora_' in k for k in result.missing_keys)
            loaded_adapter = get_peft_model_state_dict(model)
            assert adapter.keys() == loaded_adapter.keys()
            assert all(torch.equal(v,loaded_adapter[k].cpu()) for k,v in adapter.items())
            zz,nn = forward(model,head,probe)
            reload_error = (z-zz).abs().max().item()
            null_reload_error = (n-nn).abs().max().item() if n is not None else 0.
            assert reload_error == null_reload_error == 0
            del z,n,zz,nn,adapter,loaded_adapter
        check_budget(args)
        model.train(); head.train()
        z,n = forward(model,head,probe)
        z.retain_grad()
        if n is not None:
            n.retain_grad()
        c,e = losses(z,n,probe,args)
        cg = torch.autograd.grad(c,head.readout.weight,retain_graph=True)[0].float().norm().item()
        eg = torch.autograd.grad(e,head.readout.weight,retain_graph=True)[0].float().norm().item()
        assert math.isfinite(cg) and math.isfinite(eg) and cg>0 and eg>0
        (c+args.event_weight*e).backward()
        mask = torch.zeros_like(z,dtype=torch.bool)
        endpoints = torch.zeros_like(mask)
        for span in probe['spans']:
            mask[span['start']:span['end']] = True
            endpoints[span['end']-1] = True
        assert torch.count_nonzero(z.grad[~mask]) == 0
        if n is not None:
            assert torch.count_nonzero(n.grad[~endpoints]) == 0
        grads={name:p.grad.float().norm().item() for name,p in trainable.items() if p.grad is not None}
        assert grads.keys() == trainable.keys()
        assert all(math.isfinite(v) for v in grads.values()) and any(v>0 for v in grads.values())
        assert any(v>0 for name,v in grads.items() if 'lora_B' in name)
        assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
        head_grads = {name:p.grad.float().norm().item() for name,p in head.named_parameters() if p.grad is not None}
        assert all(math.isfinite(v) for v in head_grads.values())
        for i in (0,1):
            assert any(v>0 for name,v in head_grads.items() if name.startswith(f'blocks.{i}.'))
        assert head_grads['projection.weight'] > 0 and head_grads['readout.weight'] > 0
        if args.attention:
            assert head_grads['attention_qkv.weight'] > 0 and head_grads['attention_output.weight'] > 0
        if n is not None:
            assert head_grads['null_readout.weight'] > 0
        model.zero_grad(set_to_none=True);head.zero_grad(set_to_none=True)
        del z,n,c,e
        check_budget(args)
        longest=max(training,key=lambda r:r['tokens'])
        z,n=forward(model,head,longest);c,e=losses(z,n,longest,args)
        (c+args.event_weight*e).backward()
        assert all(torch.isfinite(p.grad).all() for p in list(model.parameters())+list(head.parameters()) if p.grad is not None)
        model.zero_grad(set_to_none=True);head.zero_grad(set_to_none=True)
        del z,n,c,e
        write(args.root/'gate.json',{'status':'passed','token_gradient':cg,'event_gradient':eg,
            'lora_gradient_norms':grads,'head_gradient_norms':head_grads,
            'causal_suffix_error':causal_error,'null_causal_suffix_error':null_causal_error,
            'reload_error':reload_error,'null_reload_error':null_reload_error,'loss_support_passed':True,
            'longest_backward_tokens':longest['tokens'],'stress_split':'training','optimizer_steps':0})
        opt=torch.optim.AdamW([{'params':head.parameters(),'lr':3e-4},
            {'params':[p for p in model.parameters() if p.requires_grad],'lr':args.lora_lr}],weight_decay=.01,fused=True)
        for epoch in range(1,args.epochs+1):
            dialogs_done = applied_dialogs = 0
            check_budget(args)
            model.train();head.train();order=list(training)
            random.Random(original.SEED+epoch).shuffle(order)
            opt.zero_grad(set_to_none=True);tc=te=0.
            for i,row in enumerate(tqdm(order,desc=f'lora {epoch}/{args.epochs}',mininterval=5)):
                check_budget(args)
                z,n=forward(model,head,row);c,e=losses(z,n,row,args)
                window=min(4,len(order)-(i//4)*4)
                ((c+args.event_weight*e)/window).backward()
                tc+=c.detach().item();te+=e.detach().item()
                if (i+1)%4==0 or i+1==len(order):
                    torch.nn.utils.clip_grad_norm_(list(head.parameters())+[p for p in model.parameters() if p.requires_grad],1.,error_if_nonfinite=True)
                    opt.step();opt.zero_grad(set_to_none=True)
                    applied_dialogs = i+1
                dialogs_done = i+1
                del z,n,c,e
                check_budget(args)
            folder=args.root/f'epoch{epoch}';folder.mkdir()
            torch.save(head.state_dict(),folder/'head.pt');model.save_pretrained(folder/'adapter')
            torch.save({'optimizer':opt.state_dict(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),
                'python_rng':random.getstate(),'epoch':epoch,'training_epoch_complete':True},folder/'training_state.pt')
            write(folder/'status.json',{'status':'trained_pending_validation','epoch':epoch})
            metrics=validate(model,head,validation,args.root,f'dev_epoch{epoch}',args)
            entry={'epoch':epoch,'token_bce':tc/len(order),'event_bce':te/len(order),'validation':metrics,'seconds':time.monotonic()-started}
            history.append(entry);write(args.root/'history.json',history)
            write(folder/'status.json',{'status':'complete','epoch':epoch})
            key=(metrics['average_precision'],-metrics['event_bce'])
            if best is None or key>best:
                best=key;write(args.root/'selection.json',{'epoch':epoch,'criterion':'dev_select AP,ties lower event BCE',
                    'head_sha256':original.file_hash(folder/'head.pt'),
                    'adapter_sha256':{p.name:original.file_hash(p) for p in sorted((folder/'adapter').iterdir()) if p.is_file()},
                    'metrics':metrics})
        status = {'status':'complete','epochs':len(history)}
    except BudgetStop as error:
        status = {'status':'budget_stopped','reason':str(error),'epochs':len(history),
            'partial_epoch':epoch,'dialogs_processed':dialogs_done,'dialogs_applied':applied_dialogs,
            'pending_dialogs_not_applied':dialogs_done-applied_dialogs,'diagnostic_resumable':False}
        write(args.root/'budget_stop.json',status)
        if model is not None and head is not None:
            # Diagnostic only: pending accumulation gradients are intentionally not resumable.
            torch.save({'head':head.state_dict(),**status},args.root/'deadline_partial.pt')
            model.save_pretrained(args.root/'deadline_adapter')
    except Exception as error:
        status = {'status':'failed','error':repr(error),'epochs':len(history)}
        write(args.root/'failure.json',{'error':repr(error),'seconds':time.monotonic()-started,'no_silent_exclusions':True})
        raise
    finally:
        try:
            if frozen_before is not None:
                frozen_after = original.frozen_hash(model)
                unchanged = frozen_after == frozen_before
                write(args.root/'frozen_audit.json',{'before':frozen_before,'after':frozen_after,'unchanged':unchanged})
                status['frozen_weights_unchanged'] = unchanged
                assert unchanged, 'frozen backbone weights changed'
            else:
                status['frozen_weights_unchanged'] = None
        except Exception as error:
            status.update(status='failed',frozen_audit_error=repr(error))
            raise
        finally:
            status['seconds'] = time.monotonic()-started
            if model is not None:
                status['peak_cuda_bytes'] = torch.cuda.max_memory_allocated()
            write(args.root/'status.json',status)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--splits',type=Path,required=True)
    parser.add_argument('--train-split',choices=('train_small','train_full'),default='train_small')
    parser.add_argument('--readout',choices=('endpoint','null','linear_softmax','multiscale','leaky'),default='linear_softmax')
    parser.add_argument('--curve',choices=('broad','linear','step'),default='broad')
    parser.add_argument('--event-weight',type=float,default=.1)
    parser.add_argument('--lora-lr',type=float,default=1e-5)
    parser.add_argument('--epochs',type=int,default=2)
    parser.add_argument('--attention',action='store_true')
    parser.add_argument('--deadline-unix',type=float,required=True)
    main(parser.parse_args())
