"""Replicate the independently checked reasoning teacher pair across four GPUs."""
import concurrent.futures
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import time
import urllib.request

ROOT=Path('/var/local/danefedov/experiments/agent-safety')


def main():
    folder=ROOT/'runs/corpus-v1/serving-strong';folder.mkdir(exist_ok=False)
    old=json.loads((ROOT/'runs/corpus-v1/serving-v3/readiness.json').read_text())
    guard=json.loads((ROOT/'runs/corpus-v1/serving-gptoss/readiness.json').read_text())
    qwen=next(w for w in old if w['gpu']==3)
    for w in (guard,qwen):
        with urllib.request.urlopen(f'http://127.0.0.1:{w["port"]}/health',timeout=10) as r:assert r.status==200
    workers=[guard,qwen];python=ROOT/'envs/vllm-0.31/bin/python'
    retired=next(w for w in old if w['gpu']==1);pid=retired['pid'];proc=Path(f'/proc/{pid}')
    if proc.exists():
        assert proc.stat().st_uid==os.getuid() and 'vllm.entrypoints.cli.main' in (proc/'cmdline').read_bytes().decode()
        assert os.getpgid(pid)==pid
        fd=os.pidfd_open(pid);os.killpg(pid,signal.SIGTERM);select.select([fd],[],[],90);os.close(fd)
    for gpu,template,port in ((1,qwen,8111),(2,guard,8115)):
        model=ROOT/'models/generative'/('Qwen--Qwen3.8-27B-FP8' if gpu==1 else 'openai--gpt-oss-safeguard-20b')
        cmd=[str(python),'-m','vllm.entrypoints.cli.main','serve',str(model),'--served-model-name',template['model'],
            '--host','127.0.0.1','--port',str(port),'--max-model-len','32768','--max-num-seqs','32',
            '--max-num-batched-tokens','8192','--gpu-memory-utilization','0.80' if gpu==1 else '0.75',
            '--enable-prefix-caching','--generation-config','vllm']
        if gpu==1:cmd+=['--language-model-only','--reasoning-parser','qwen3','--speculative-config','{"method":"mtp","num_speculative_tokens":2}']
        env=dict(os.environ,PATH=str(python.parent)+':'+os.environ['PATH'],CUDA_VISIBLE_DEVICES=str(gpu),HF_HUB_OFFLINE='1',
            HF_HUB_DISABLE_IMPLICIT_TOKEN='1',XDG_CACHE_HOME=str(ROOT/'envs/runtime-cache'),TRITON_CACHE_DIR=str(folder/f'triton_gpu{gpu}'),
            OMP_NUM_THREADS='4',VLLM_USAGE_STATS_ENABLED='0',DO_NOT_TRACK='1')
        with (folder/f'gpu{gpu}.log').open('w') as log:p=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        workers.append({'gpu':gpu,'model':template['model'],'revision':template['revision'],'port':port,'pid':p.pid,
            'mtp':gpu==1,'command':cmd,'status':'starting'})
    (folder/'launch.json').write_text(json.dumps(workers,indent=2))
    def ready(w):
        if w['status']=='ready':return w
        deadline=time.monotonic()+1200
        while time.monotonic()<deadline:
            try:
                with urllib.request.urlopen(f'http://127.0.0.1:{w["port"]}/health',timeout=3) as r:
                    if r.status==200:return dict(w,status='ready')
            except Exception:pass
            path=Path(f'/proc/{w["pid"]}/stat')
            if not path.exists() or path.read_text().split(') ',1)[1].startswith('Z'):return dict(w,status='failed')
            time.sleep(3)
        return dict(w,status='timeout')
    with concurrent.futures.ThreadPoolExecutor(4) as pool:result=list(pool.map(ready,workers))
    (folder/'readiness.json').write_text(json.dumps(result,indent=2))
    print(json.dumps([{k:w[k] for k in ('gpu','model','port','pid','status')} for w in result],indent=2),flush=True)


if __name__=='__main__':main()
