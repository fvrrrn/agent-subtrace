"""Launch once: no stdout/SSH pipe is inherited by the scheduler."""
import json
import os
from pathlib import Path
import subprocess
import sys

BASE = Path('/var/local/danefedov/experiments/agent-safety')
SOURCE = Path(__file__).parent
LAUNCH = SOURCE.parent
env = dict(os.environ, CUDA_VISIBLE_DEVICES='3', CUBLAS_WORKSPACE_CONFIG=':4096:8',
    HF_HOME=str(BASE/'models/huggingface'), HF_HUB_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1',
    PYTHONPATH=str(SOURCE/'src')+':'+str(BASE/'archive/legacy-shm/danefedov-qwen-mamba-v1/deps'))
with (LAUNCH/'scheduler.log').open('xb') as log:
    child = subprocess.Popen([sys.executable,'-u',str(SOURCE/'recover_queue.py')], cwd=SOURCE,
        stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, env=env,
        start_new_session=True, close_fds=True)
record = {'pid':child.pid,'log':str(LAUNCH/'scheduler.log'),'source':str(SOURCE),
    'stdin':'DEVNULL','stdout_stderr':'regular_file','start_new_session':True}
with (LAUNCH/'detached.json').open('x') as stream:
    json.dump(record, stream, indent=2)
print(json.dumps(record), flush=True)
