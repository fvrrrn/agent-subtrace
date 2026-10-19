"""Separate live tqdm viewer, completion wait through pidfd."""
import json
import os
from pathlib import Path
import select
import subprocess

BASE=Path('/var/local/danefedov/experiments/agent-safety/runs')
launch=BASE/'fresh-factorial-v1-launch'; run=BASE/'fresh-factorial-v1'
pid=json.loads((launch/'detached.json').read_text())['pid']
fd=os.pidfd_open(pid)
logs=[launch/'scheduler.log']+[run/(a+'_'+c+'.log') for a in ('lora','frozen') for c in ('step','broad')]
tail=subprocess.Popen(['tail','-n','6','-F',*[str(p) for p in logs]],stderr=subprocess.DEVNULL)
try:
    poll=select.poll(); poll.register(fd,select.POLLIN); poll.poll()
finally:
    tail.terminate(); tail.wait(); os.close(fd)
print((run/'status.json').read_text(),flush=True)
