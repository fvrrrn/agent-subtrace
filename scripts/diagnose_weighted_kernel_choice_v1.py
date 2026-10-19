"""Controlled A/B/A native-kernel replay on identical captured weights and input."""
import json
from pathlib import Path
import torch
from tqdm import tqdm
import train_weighted_broad_lora_v1 as w
from diagnose_weighted_broad_replay_v1 import diff,state_hash
from weighted_lora_kernel_lock import apply

root=w.ROOT/'runs'
capture=root/'weighted-broad-replay-diagnostic-v3'
output=root/'weighted-broad-kernel-choice-diagnostic-v1';output.mkdir(exist_ok=False)
torch.set_num_threads(4);w.trainer.seed(w.trainer.original.SEED)
saved=torch.load(capture/'failure_capture.pt',weights_only=True)
head=w.trainer.Head('curve_linear_softmax',False).cuda().eval();head.load_state_dict(saved['head'])
x=saved['cache'].cuda();n=len(saved['reference']);idx=[s['end']-1 for s in saved['row']['spans']]
report={'head_before':state_hash(head),'runs':[]};first=None
with torch.no_grad():
 for name,record in tqdm([('A',capture/'diagnostic.json'),
                          ('B',root/'weighted-broad-headonly-diagnostic-v1/diagnostic.json'),
                          ('A_repeat',capture/'diagnostic.json')],desc='Controlled kernel A/B/A'):
  configs=apply(record)
  with torch.autocast('cuda',dtype=torch.bfloat16):z,_=head(x)
  z=z[0,:n].float().cpu()
  if first is None:first=z.clone()
  report['runs'].append({'name':name,'configs':configs,'vs_original_reference':diff(z,saved['reference']),
      'vs_failed_capture':diff(z,saved['actual']),'vs_A':diff(z,first),
      'endpoints_vs_original':diff(z[idx],saved['reference'][idx])})
  torch.save(z,output/(name+'.pt'))
report['head_after']=state_hash(head)
w.write(output/'diagnostic.json',report)
print(json.dumps(report,indent=2),flush=True)
