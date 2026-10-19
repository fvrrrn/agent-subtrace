"""Four localhost teacher replicas; paired native MTP / target-only speed comparison."""
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.request

ROOT=Path('/var/local/danefedov/experiments/agent-safety')


def main():
    run=ROOT/'runs/corpus-v1/serving-v3';run.mkdir(exist_ok=False)
    py=ROOT/'envs/vllm-0.31/bin/python'
    workers=[]
    for gpu in range(4):
        model='Qwen--Qwen3.6-35B-A3B-FP8' if gpu<2 else 'Qwen--Qwen3.8-27B-FP8'
        path=ROOT/'models/generative'/model;meta=json.loads((path/'DOWNLOAD_MANIFEST.json').read_text())
        mtp=gpu%2==1;port=8110+gpu
        cmd=[str(py),'-m','vllm.entrypoints.cli.main','serve',str(path),'--served-model-name',model,
            '--host','127.0.0.1','--port',str(port),'--language-model-only','--max-model-len','32768',
            '--max-num-seqs','32','--max-num-batched-tokens','8192','--gpu-memory-utilization','0.80',
            '--enable-prefix-caching','--reasoning-parser','qwen3','--generation-config','vllm',
            '--seed','20261018']
        if mtp:cmd+=['--speculative-config',json.dumps({'method':'mtp','num_speculative_tokens':2})]
        env=dict(os.environ,PATH=str(py.parent)+':'+os.environ['PATH'],CUDA_VISIBLE_DEVICES=str(gpu),HF_HUB_OFFLINE='1',HF_HUB_DISABLE_IMPLICIT_TOKEN='1',
            HF_HOME=str(ROOT/'models/generative-hf-cache'),XDG_CACHE_HOME=str(ROOT/'envs/runtime-cache'),
            TRITON_CACHE_DIR=str(run/f'triton_gpu{gpu}'),OMP_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',
            VLLM_USAGE_STATS_ENABLED='0',DO_NOT_TRACK='1')
        with (run/f'gpu{gpu}.log').open('w') as log:p=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        workers.append({'gpu':gpu,'model':model,'revision':meta['revision'],'mtp':mtp,'port':port,'pid':p.pid,'command':cmd})
        (run/'launch.json').write_text(json.dumps(workers,indent=2))
    def ready(w):
        deadline=time.monotonic()+1200
        while time.monotonic()<deadline:
            try:
                with urllib.request.urlopen(f'http://127.0.0.1:{w["port"]}/health',timeout=3) as r:
                    if r.status==200:return dict(w,status='ready')
            except Exception:pass
            try:os.kill(w['pid'],0)
            except ProcessLookupError:return dict(w,status='failed')
            proc=Path(f'/proc/{w["pid"]}/stat')
            if proc.exists() and proc.read_text().split(') ',1)[1].startswith('Z'):return dict(w,status='failed')
            time.sleep(3)
        return dict(w,status='timeout')
    with concurrent.futures.ThreadPoolExecutor(4) as pool:result=list(pool.map(ready,workers))
    (run/'readiness.json').write_text(json.dumps(result,indent=2))
    print(json.dumps([{k:w[k] for k in ('gpu','model','mtp','pid','status')} for w in result],indent=2),flush=True)


if __name__=='__main__':main()
