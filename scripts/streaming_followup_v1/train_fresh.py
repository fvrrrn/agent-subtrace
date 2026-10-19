"""Fresh shared head initialization; unchanged native architecture and trainer gates."""
import argparse
import json
from pathlib import Path
import sys
import torch
import train_weighted_broad_lora_v1 as parent
from prepare_whole_dialogs import write


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--lock',type=Path,required=True)
    parser.add_argument('--adaptation',required=True)
    parser.add_argument('--curve',required=True)
    cli=parser.parse_args(); lock=json.loads(cli.lock.read_text())
    assert [cli.adaptation,cli.curve] in lock['cells']
    original_main=parent.trainer.main

    def run(args):
        args.event_weight=lock['event_weight']
        args.warmstart_sha256=None
        args.initialization='fresh shared seeded head; fresh zero-output LoRA adapters'
        args.fresh_lock_sha256=parent.trainer.original.file_hash(cli.lock)

        def prepare(head):
            hashes=parent.weighted.base.curve_heads.initialization(head)
            shared=cli.lock.parent/'shared_initialization.json'
            if shared.exists():
                assert json.loads(shared.read_text())==hashes
            else:
                with shared.open('x') as stream:
                    json.dump(hashes,stream,indent=2)
            torch.save(head.state_dict(),args.root/'initial_head.pt')
            write(args.root/'initialization.json',hashes)

        def fresh_gate(model,head,training,validation):
            # No trained-head reference applies to fresh init. The inherited
            # causal/reload and both-loss gates below verify this initialization.
            values=[p for n,p in model.named_parameters() if 'lora_B' in n]
            assert values and all(torch.count_nonzero(p)==0 for p in values)
            write(args.root/'fresh_init_gate.json',{'status':'passed','zero_lora_B':True,
                'initialization':parent.weighted.base.curve_heads.initialization(head),
                'old_trained_head_replay':'not applicable; no weights loaded from warm-start'})
        args.prepare_head=prepare
        args.warmstart_gate=fresh_gate
        original_main(args)
    parent.trainer.main=run
    sys.argv=[sys.argv[0],'--root',str(cli.root),'--adaptation',cli.adaptation,'--curve',cli.curve,'--weighting','uniform']
    parent.main()


if __name__=='__main__':
    main()
