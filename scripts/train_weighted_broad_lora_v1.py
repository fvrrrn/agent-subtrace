"""Eight locked continuation conditions; shared selected head, online Instruct."""
import argparse
import io
import json
import math
from pathlib import Path
from types import SimpleNamespace
import torch
from tqdm import tqdm
import train_autoresearch_lora as trainer
import train_streaming_weighted_multilayer_v1 as weighted
from prepare_whole_dialogs import write

ROOT = Path('/var/local/danefedov/experiments/agent-safety')
PARENT = ROOT / 'runs/streaming-weighted-multilayer-v1/last_broad/fit'
DATA = ROOT / 'runs/frozen-instruct-4b-v1/data'
BASE = 'Qwen/Qwen3.5-4B'
REVISION = '851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'
HEAD_SHA = '7f3d0d3950313f2f7b39ab1b7a92b4933c3f072b885f0025ebfe42701a9b2137'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--adaptation', choices=('lora', 'frozen'), required=True)
    parser.add_argument('--curve', choices=('broad', 'step'), required=True)
    parser.add_argument('--weighting', choices=('adaptive', 'uniform'), required=True)
    cli = parser.parse_args()
    args = SimpleNamespace(root=cli.root, data=DATA, splits=DATA/'splits.json', train_split='train_small',
        readout='endpoint', curve=cli.curve, event_weight=.1, lora_lr=1e-5, lora_last_layers=0,
        epochs=2, attention=False, deadline_unix=None, frozen=cli.adaptation=='frozen',
        weighting=cli.weighting, warmstart_sha256=HEAD_SHA, head_lr=3e-4)
    torch.set_num_threads(4)
    from weighted_lora_kernel_lock import apply
    kernel_manifest = Path(__file__).with_name('WEIGHTED_LORA_KERNELS.json')
    args.kernel_manifest_sha256 = trainer.original.file_hash(kernel_manifest)
    args.native_forward_kernel_configs = apply(kernel_manifest)
    trainer.BASE = trainer.original.BASE = BASE
    trainer.REVISION = trainer.original.REVISION = REVISION
    trainer.check_budget = lambda args: None
    def forward(model, head, row):
        hidden = trainer.original.encode(model, row)
        # Match frozen-cache Mamba input exactly: padding contains zeros, not
        # backbone outputs at masked positions. Retain full gradients on real tokens.
        hidden = torch.nn.functional.pad(hidden[:, :row['tokens']],
            (0,0,0,hidden.shape[1]-row['tokens'])).contiguous()
        with torch.autocast('cuda',dtype=torch.bfloat16):
            z,n = head(hidden)
        return z[0,:row['tokens']].float(), None
    trainer.forward = forward
    original_backbone = trainer.original.backbone
    # Frozen cells retain disabled-gradient zero adapters for identical kernels.
    def backbone(lora):
        model = original_backbone(True)
        if args.frozen:
            model.requires_grad_(False)
        return model
    trainer.original.backbone = backbone
    if args.frozen:
        args.lora_lr = 0.
    original_losses = trainer.losses
    trainer.losses = lambda z,n,row,a: (weighted.loss_parts(z,n,row['spans'],a)
        if a.weighting == 'adaptive' else original_losses(z,n,row,a))
    selection = trainer.read(PARENT/'selection.json')
    assert selection['epoch'] == 2 and selection['head_sha256'] == HEAD_SHA
    assert trainer.original.file_hash(PARENT/'head_epoch2.pt') == HEAD_SHA
    rows, audit = trainer.original.load_data(DATA)
    splits = trainer.read(args.splits)
    assert {k:len(v) for k,v in splits.items()} == {'train_small':2009, 'dev_select':301}
    assert len(rows) == len({r['key'] for r in rows}) == 2310
    assert set(r['key'] for r in rows) == set(splits['train_small']) | set(splits['dev_select'])
    cfg = json.loads(json.dumps({k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}))
    buffer = io.BytesIO(); torch.save(cfg,buffer); buffer.seek(0)
    assert torch.load(buffer,weights_only=True) == cfg
    for row in tqdm(rows, desc='CPU identities', unit='dialog'):
        assert trainer.digest(row['ids']) == row['ids_sha256']
    def prepare_head(head):
        head.load_state_dict(torch.load(PARENT/'head_epoch2.pt', map_location='cpu', weights_only=True), strict=True)
    args.prepare_head = prepare_head
    def warmstart_gate(model, head, training, validation):
        model.eval(); head.eval()
        # Exact online zero-adapter replay against every saved selected dev token.
        with torch.no_grad():
            for row in tqdm(validation, desc='Exact warm-start replay', unit='dialog'):
                saved = torch.load(PARENT/'dev_select_epoch2/tokens'/(row['key']+'.pt'), weights_only=True)
                assert saved['ids_sha256'] == row['ids_sha256']
                z,_ = trainer.forward(model,head,row)
                if not torch.equal(z.cpu(),saved['raw_token_logits']):
                    torch.save({'key':row['key'],'actual':z.cpu(),'reference':saved['raw_token_logits']},
                               args.root/'warmstart_mismatch.pt')
                    raise AssertionError(f'Exact replay mismatch: {row["key"]}')
        write(args.root/'warmstart_gate.json', {'status':'passed','dialogs':len(validation),
            'actions':sum(len(r['spans']) for r in validation),'raw_error':0,'head_sha256':HEAD_SHA})
        print('WARMSTART_PASSED', flush=True)
    args.warmstart_gate = warmstart_gate
    def gradient_gate(model, head, token, event):
        watched = [('head.'+n,p) for n,p in head.named_parameters()]
        watched += [('backbone.'+n,p) for n,p in model.named_parameters() if p.requires_grad]
        report = {}
        for kind,loss in [('token',token),('event',event)]:
            grads = torch.autograd.grad(loss,[p for _,p in watched],retain_graph=True,allow_unused=True)
            norms = {n:g.float().norm().item() for (n,_),g in zip(watched,grads) if g is not None}
            assert all(math.isfinite(v) for v in norms.values())
            for prefix in ('head.blocks.0.','head.blocks.1.','head.readout.','head.projection.'):
                assert any(v>0 for n,v in norms.items() if n.startswith(prefix))
            if not args.frozen:
                assert any(v>0 for n,v in norms.items() if 'lora_B' in n)
            report[kind] = norms
        write(args.root/'separate_loss_gradients.json',report)
    args.gradient_gate = gradient_gate
    trainer.main(args)
    if args.frozen:
        from safetensors.torch import load_file
        for epoch in (1,2):
            state = load_file(str(args.root/f'epoch{epoch}/adapter/adapter_model.safetensors'))
            assert all(torch.count_nonzero(v)==0 for n,v in state.items() if 'lora_B' in n)
        write(args.root/'zero_adapter_audit.json', {'status':'passed','epochs':2,'adapter_lr':0})


if __name__ == '__main__':
    main()
