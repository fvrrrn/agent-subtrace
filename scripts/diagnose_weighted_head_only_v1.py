"""Replay captured identical head/input, recording actual Triton choices."""
import argparse
from pathlib import Path
import torch
from tqdm import tqdm
import train_weighted_broad_lora_v1 as w
from diagnose_weighted_broad_replay_v1 import diff,state_hash,autotune_state

parser=argparse.ArgumentParser();parser.add_argument('--capture',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
torch.set_num_threads(4);w.trainer.seed(w.trainer.original.SEED)
saved=torch.load(args.capture/'failure_capture.pt',weights_only=True)
head=w.trainer.Head('curve_linear_softmax',False).cuda().eval()
head.load_state_dict(saved['head'],strict=True)
x=saved['cache'].cuda();ref=saved['reference'];n=len(ref)
report={'head_sha256':state_hash(head),'repeats':[]}
with torch.no_grad():
 for repeat in tqdm(range(10),desc='Captured head-only repeated replay'):
  with torch.autocast('cuda',dtype=torch.bfloat16):z,_=head(x)
  report['repeats'].append(diff(z[0,:n].float(),ref))
 report['autotune']=autotune_state()
 captures={};handles=[]
 def capture(name):
  def hook(m,i,o):captures[name]=o.detach().cpu().clone()
  return hook
 for name,module in head.named_modules():
  if name in ('input_norm','projection','norms.0','blocks.0','norms.1','blocks.1','final_norm','readout'):
   handles.append(module.register_forward_hook(capture(name)))
 with torch.autocast('cuda',dtype=torch.bfloat16):head(x)
 for h in handles:h.remove()
 torch.save(captures,args.output/'modules.pt')
 prior=args.capture/'modules.pt'
 if prior.exists():
  before=torch.load(prior,weights_only=True)
  report['module_diff_vs_full_process']={k:diff(v,before[k]) for k,v in captures.items()}
w.write(args.output/'diagnostic.json',report)
print(report,flush=True)
