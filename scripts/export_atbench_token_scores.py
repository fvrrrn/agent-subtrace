"""One fixed trajectory: real cached-Qwen Mamba token readouts, no training."""
import argparse
import hashlib
import json
from pathlib import Path

import train_action_models as training


if __name__=='__main__':
    import torch
    from transformers import AutoTokenizer
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--model',default='mamba_linear')
    a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    key='atbench:0:1'
    rows=[r for r in training.read(a.root/'records.json') if r['case_key']==key]
    assert len(rows)==3
    item=next(m for m in training.read(a.root/'models.json') if m['name']==a.model)
    head=training.make_head(1).eval().requires_grad_(False)
    checkpoint=a.root/'locked'/a.model/'selected_head.pt'
    assert training.sha(checkpoint)==item['selected']['head_sha256']
    head.load_state_dict(torch.load(checkpoint,weights_only=True,map_location='cuda'))
    expected={r['key']:r for r in map(json.loads,(a.root/'predictions'/a.model/'actions.jsonl').read_text().splitlines()) if r['case_key']==key}
    tok=AutoTokenizer.from_pretrained(training.BASE,revision=training.REVISION,local_files_only=True)
    from evaluate_action_tsharm import args_for
    args=args_for(a.root,online=item['kind']=='lora')
    model=None
    if item['kind']=='lora':
        from peft import PeftModel
        model=PeftModel.from_pretrained(training.backbone(args),a.root/'locked'/a.model/'adapter',is_trainable=False)
        model.eval().requires_grad_(False)
    events=[];error=0.
    with torch.inference_mode():
        for batch in training.batches(rows,args.batch_size,args.token_budget):
            z=training.token_forward(args,model,head,batch).float()
            for i,r in enumerate(batch):
                scores=z[i,:r['tokens']].sigmoid().cpu().tolist()
                ids=r['ids']
                event={'id':r['event_id'],'kind':r['kind'],'action_start':r['start'],
                    'action':json.loads(r['state'])['proposed_action'],
                    'context_tokens':r['start'],'total_tokens':r['tokens'],
                    'action_token_ids':ids[r['start']:],
                    'action_token_pieces':[tok.decode([v]) for v in ids[r['start']:]],
                    'action_scores':scores[r['start']:],'full_prefix_scores':scores,
                    'raw_action_logodds':z[i,r['start']:r['tokens']].cpu().tolist(),
                    'saved_endpoint':expected[r['key']]['score'],
                    'state_sha256':hashlib.sha256(r['state'].encode()).hexdigest()}
                assert event['state_sha256']==expected[r['key']]['state_sha256']
                error=max(error,abs(scores[-1]-event['saved_endpoint']))
                events.append(event)
    assert error<1e-6
    events.sort(key=lambda r:int(r['id'][1:]))
    result={'case_key':key,'source_id':1,'model':a.model,'selected':item['selected'],
            'trajectory_label':1,'per_action_gold':None,'endpoint_max_error':error,
            'interpretation':'Sigmoid readout at each token, not calibrated probability or token safety gold. Each action replays its own full prefix; states are reset between actions. Plot concatenates only action-token readouts, not one continuous streaming state.',
            'events':events}
    training.write(a.output/'token_scores.json',result)
    print(json.dumps({'case':key,'endpoint_max_error':error,
        'events':[{'id':e['id'],'action_tokens':len(e['action_scores']),
                   'endpoint':e['action_scores'][-1],'max':max(e['action_scores'])} for e in events]},indent=2))
