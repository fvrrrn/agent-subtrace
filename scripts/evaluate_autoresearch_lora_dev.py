"""Locked LoRA checkpoint evaluation; no fitting or training on confirmation data."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import time

import torch

import train_whole_dialogs as original
from train_curve_event_heads import Head
from train_autoresearch_lora import validate
from prepare_whole_dialogs import digest, write
from train_autoresearch_heads import seed, environment


def read(path):
    return json.loads(path.read_text())


def main(args):
    assert time.time()+360<args.deadline_unix, 'Insufficient setup/evaluation reserve'
    if args.split!='dev_select':assert args.lock is not None, 'Confirmation requires final lock'
    selection=read(args.train_root/'selection.json')
    protocol=read(args.train_root/'protocol.json')
    locked=None
    if args.lock is not None:
        lock=read(args.lock);locked=next(e for e in lock['selected'] if e['name']==args.name)
        assert Path(locked['root'])==args.train_root and locked['mode']==args.mode
        assert locked['selection']==selection
        assert original.file_hash(Path(locked['training_protocol']['path']))==locked['training_protocol']['sha256']
    assert read(args.train_root/'status.json')['status']=='complete'
    checkpoint=args.train_root/f'epoch{selection["epoch"]}'
    expected_head=selection['head_sha256'] if args.mode=='lora' else selection['checkpoint_hashes']['head.pt']
    assert original.file_hash(checkpoint/'head.pt')==expected_head
    if locked is not None:
        assert locked['epoch']==selection['epoch'] and locked['head']['sha256']==expected_head
        assert Path(locked['head']['path'])==checkpoint/'head.pt'
    config=SimpleNamespace(**protocol)
    config.deadline_unix=args.deadline_unix
    rows,audit=original.load_data(Path(protocol['data']))
    assert audit['dialogs_sha256']==protocol['data_sha256']
    splits=read(Path(protocol['splits']))
    split_hash=protocol.get('split_sha256',protocol.get('splits_sha256'))
    assert original.file_hash(Path(protocol['splits']))==split_hash
    index={r['key']:r for r in rows}
    selected=[index[k] for k in splits[args.split]]
    assert all(digest(r['ids'])==r['ids_sha256'] for r in selected)
    seed(protocol['seed'])
    head=Head('curve_null' if config.readout=='null' else 'curve_linear_softmax',getattr(config,'attention',False)).cuda()
    head.load_state_dict(torch.load(checkpoint/'head.pt',weights_only=True,map_location='cpu'))
    model=original.backbone(False)
    forward_fn=None
    if args.mode=='lora':
        adapter_hashes={str(p.relative_to(checkpoint/'adapter')):original.file_hash(p)
                        for p in (checkpoint/'adapter').rglob('*') if p.is_file()}
        assert adapter_hashes==selection['adapter_sha256']
        if locked is not None:assert adapter_hashes==locked['adapter']['files_sha256']
        from peft import PeftModel
        model=PeftModel.from_pretrained(model,checkpoint/'adapter',is_trainable=False)
    else:
        assert original.file_hash(checkpoint/'backbone.pt')==selection['checkpoint_hashes']['backbone.pt']
        if locked is not None:assert selection['checkpoint_hashes']['backbone.pt']==locked['backbone']['sha256']
        model.float()
        model.load_state_dict(torch.load(checkpoint/'backbone.pt',map_location='cpu',weights_only=True),strict=True)
        # Match FSDP's mixed-precision forward: BF16 parameter copies, FP32 buffers.
        # The immutable FP32 master checkpoint remains the source of truth.
        for parameter in model.parameters():parameter.data=parameter.data.to(torch.bfloat16)
        from train_autoresearch_fullft import forward as forward_fn
    head.eval().requires_grad_(False);model.eval().requires_grad_(False)
    hashes={str(p.relative_to(checkpoint)):original.file_hash(p) for p in checkpoint.rglob('*') if p.is_file()
            and p.name not in ('training_state.pt','hashes.json','status.json') and not p.name.startswith('training_rank')}
    started=time.monotonic()
    output=args.train_root/'evaluations';output.mkdir(exist_ok=True)
    try:
        metrics=validate(model,head,selected,output,args.split,config,forward_fn=forward_fn)
        assert hashes=={str(p.relative_to(checkpoint)):original.file_hash(p) for p in checkpoint.rglob('*') if p.is_file()
            and p.name not in ('training_state.pt','hashes.json','status.json') and not p.name.startswith('training_rank')}
        write(output/args.split/'evaluation_protocol.json',{'checkpoint_hashes':hashes,'selection':selection,
            'split':args.split,'selection_allowed':False,'data_sha256':audit['dialogs_sha256'],
            'environment':environment(),'locked_manifest':str(args.lock) if args.lock is not None else None,
            'fullft_forward_precision':'FP32 checkpoint loaded first; BF16 parameter copies with FP32 buffers and BF16 autocast' if args.mode=='fullft' else None,
            'splits_sha256':split_hash,'seconds':time.monotonic()-started,
            'peak_cuda_bytes':torch.cuda.max_memory_allocated(),'threshold_fit':False})
        print(json.dumps(metrics,indent=2),flush=True)
    except Exception as error:
        write(output/(args.split+'_failure.json'),{'error':repr(error),'seconds':time.monotonic()-started})
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-root',type=Path,required=True)
    parser.add_argument('--mode',choices=('lora','fullft'),default='lora')
    parser.add_argument('--lock',type=Path)
    parser.add_argument('--name')
    parser.add_argument('--split',choices=('dev_select','dev_confirm','original_validation'),required=True)
    parser.add_argument('--deadline-unix',type=float,required=True)
    main(parser.parse_args())
