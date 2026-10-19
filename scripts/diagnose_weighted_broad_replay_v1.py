"""Bounded same-workflow replay diagnosis; always exits before training."""
import hashlib
import json
import sys
from pathlib import Path
import torch
from tqdm import tqdm
import train_weighted_broad_lora_v1 as w


def state_hash(head):
    h = hashlib.sha256()
    for name,value in head.state_dict().items():
        h.update(name.encode()); h.update(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def diff(a,b):
    a,b=a.detach().cpu(),b.detach().cpu()
    return {'equal':torch.equal(a,b),'max_error':(a.float()-b.float()).abs().max().item(),
            'mismatches':int((a!=b).sum()),'shape':list(a.shape)}


def layout(x):
    return {'shape':list(x.shape),'stride':list(x.stride()),'dtype':str(x.dtype),
            'offset':x.storage_offset(),'pointer_mod256':x.data_ptr()%256}


def autotune_state():
    from triton.runtime.autotuner import Autotuner
    result = {}
    for name,module in list(sys.modules.items()):
        if not name.startswith('mamba_ssm.'):
            continue
        for symbol,value in vars(module).items():
            if isinstance(value,Autotuner) and value.cache:
                result[name+'.'+symbol] = {str(k):str(v) for k,v in value.cache.items()}
    return result


def main():
    original_main = w.trainer.main
    def diagnostic_main(args):
        report = {'head_hashes':{},'environment':w.weighted.instruct.sound.environment()}
        prepare = args.prepare_head
        def prepare_head(head):
            prepare(head)
            report['head_hashes']['before_backbone'] = state_hash(head)
        args.prepare_head = prepare_head
        def gate(model,head,training,validation):
            model.eval();head.eval()
            report['head_hashes']['after_backbone'] = state_hash(head)
            report['precision'] = {'matmul_precision':torch.get_float32_matmul_precision(),
                'tf32_matmul':torch.backends.cuda.matmul.allow_tf32,'tf32_cudnn':torch.backends.cudnn.allow_tf32,
                'cudnn_benchmark':torch.backends.cudnn.benchmark,'deterministic':torch.are_deterministic_algorithms_enabled(),
                'model_training':model.training,'head_training':head.training}
            report['rows'] = []
            with torch.no_grad():
                for row in tqdm(validation,desc='Same-process strict replay diagnosis',unit='dialog'):
                    ref = torch.load(w.PARENT/'dev_select_epoch2/tokens'/(row['key']+'.pt'),weights_only=True)['raw_token_logits']
                    z,_ = w.trainer.forward(model,head,row)
                    entry = {'key':row['key'],'raw':diff(z,ref)}
                    report['rows'].append(entry)
                    if torch.equal(z.cpu(),ref):
                        continue
                    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
                        cached = torch.load(w.ROOT/'runs/frozen-instruct-4b-v1/cache/features'/(row['key']+'.pt'),weights_only=True)
                    online = w.trainer.original.encode(model,row)
                    with model.disable_adapter():
                        disabled = w.trainer.original.encode(model,row)
                    n = row['tokens']
                    x = torch.zeros_like(online);x[0,:n] = cached['hidden'].cuda()
                    padded = torch.nn.functional.pad(online[:,:n],(0,0,0,online.shape[1]-n)).contiguous()
                    entry.update(hidden_vs_cache=diff(online[0,:n],cached['hidden']),
                        zero_lora_vs_disabled=diff(online,disabled), padded_vs_cache=diff(padded,x),
                        layouts={'online':layout(online),'padded':layout(padded),'cache':layout(x)},
                        endpoint_initial=diff(z[[s['end']-1 for s in row['spans']]],ref[[s['end']-1 for s in row['spans']]]))
                    torch.save({'row':row,'online':online.cpu(),'cache':x.cpu(),'actual':z.cpu(),'reference':ref,
                        'head':{k:v.cpu() for k,v in head.state_dict().items()}},args.root/'failure_capture.pt')
                    runs = []
                    first = None
                    for repeat in tqdm(range(10),desc='Identical-input repeated forwards',unit='repeat'):
                        with torch.autocast('cuda',dtype=torch.bfloat16):
                            a,_=head(padded); b,_=head(x)
                        a=a[0,:n].float().cpu(); b=b[0,:n].float().cpu()
                        if first is None: first=a.clone()
                        idx=[s['end']-1 for s in row['spans']]
                        runs.append({'repeat':repeat,'online_vs_ref':diff(a,ref),'cache_vs_ref':diff(b,ref),
                            'online_vs_cache':diff(a,b),'vs_first':diff(a,first),
                            'endpoints_vs_ref':diff(a[idx],ref[idx])})
                    entry['repeats']=runs
                    # Locate the first diverging head submodule for identical input values.
                    captures = {}
                    handles=[]
                    for name,module in head.named_modules():
                        if name in ('input_norm','projection','norms.0','blocks.0','norms.1','blocks.1','final_norm','readout'):
                            handles.append(module.register_forward_hook(lambda m,i,o,name=name: captures.setdefault(name,[]).append(o.detach().cpu().clone())))
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        head(padded);head(x)
                    for handle in handles:handle.remove()
                    entry['module_diffs']={name:diff(v[0],v[1]) for name,v in captures.items()}
                    torch.save({name:v[0] for name,v in captures.items()},args.root/'modules.pt')
                    report['autotune'] = autotune_state()
                    break
            report['head_hashes']['after_replays']=state_hash(head)
            report['autotune'] = autotune_state()
            w.write(args.root/'diagnostic.json',report)
            print(json.dumps({'dialogs':len(report['rows']),'mismatched':sum(not r['raw']['equal'] for r in report['rows']),
                              'autotune':report['autotune']},indent=2),flush=True)
            raise RuntimeError('DIAGNOSTIC_COMPLETE_NO_TRAINING')
        args.warmstart_gate=gate
        return original_main(args)
    w.trainer.main=diagnostic_main
    w.main()


if __name__=='__main__':
    main()
