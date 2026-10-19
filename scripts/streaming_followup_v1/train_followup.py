"""Reuse the immutable eight-cell trainer, changing only the locked objective."""
import argparse
import json
from pathlib import Path
import sys

import torch
import torch.nn.functional as F
import train_weighted_broad_lora_v1 as parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lock', type=Path, required=True)
    parser.add_argument('--cell', required=True)
    parser.add_argument('--root', type=Path, required=True)
    cli = parser.parse_args()
    lock = json.loads(cli.lock.read_text())
    cell = next(c for c in lock['cells'] if c['name'] == cli.cell)
    original_main = parent.trainer.main

    def run(args):
        args.event_weight = cell['event_weight']
        args.positive_weight = cell['positive_weight']
        args.followup_lock_sha256 = parent.trainer.original.file_hash(cli.lock)

        def losses(z, null, row, args):
            spans = row['spans']
            valid = torch.cat([z[s['start']:s['end']].float().sigmoid() for s in spans])
            beta = (valid.std(unbiased=False)/.35).clamp(0, 1) if len(valid) >= 2 else valid.new_zeros(())
            tokens, events = [], []
            for span in spans:
                values = z[span['start']:span['end']].float()
                target = float(span['label']) * parent.weighted.base.curve_heads.profile(len(values), z.device, args.curve)
                pw = values.new_tensor(args.positive_weight)
                loss = F.binary_cross_entropy_with_logits(values, target, pos_weight=pw, reduction='none')
                if args.weighting == 'adaptive':
                    alpha = torch.softmax(beta * values.sigmoid(), dim=0).detach()
                    tokens.append((alpha * loss).sum())
                else:
                    tokens.append(loss.mean())
                events.append(F.binary_cross_entropy_with_logits(values[-1], values.new_tensor(float(span['label'])), pos_weight=pw))
            return torch.stack(tokens).mean(), torch.stack(events).mean()

        parent.trainer.losses = losses
        original_main(args)

    parent.trainer.main = run
    config = lock['winner_configuration']
    sys.argv = [sys.argv[0], '--root', str(cli.root), '--adaptation', config['adaptation'],
                '--curve', config['curve'], '--weighting', config['weighting']]
    parent.main()


if __name__ == '__main__':
    main()
