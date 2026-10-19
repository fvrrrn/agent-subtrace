"""Detached finite fresh-factorial launcher; viewer cannot break training pipes."""
import json
import os
from pathlib import Path
import subprocess
import sys

BASE=Path('/var/local/danefedov/experiments/agent-safety')
SOURCE=Path(__file__).parent
env=dict(os.environ,CUDA_VISIBLE_DEVICES='3',CUBLAS_WORKSPACE_CONFIG=':4096:8',
    HF_HOME=str(BASE/'models/huggingface'),HF_HUB_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1',
    PYTHONPATH=str(SOURCE/'src')+':'+str(BASE/'archive/legacy-shm/danefedov-qwen-mamba-v1/deps'))
with (SOURCE.parent/'scheduler.log').open('xb') as log:
    child=subprocess.Popen([sys.executable,'-u',str(SOURCE/'launch_fresh.py')],cwd=SOURCE,env=env,
        stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
record={'pid':child.pid,'source':str(SOURCE),'log':log.name,'stdin':'DEVNULL','stdout_stderr':'regular_file'}
with (SOURCE.parent/'detached.json').open('x') as stream:
    json.dump(record,stream,indent=2)
print(json.dumps(record),flush=True)
