"""Independent log viewer; pidfd completion wait never controls training."""
import itertools
import json
import os
from pathlib import Path
import select
import subprocess

BASE = Path('/var/local/danefedov/experiments/agent-safety/runs')
launch = BASE/'weighted-broad-recovery-v1-launch'
pid = json.loads((launch/'detached.json').read_text())['pid']
fd = os.pidfd_open(pid)
paths = [launch/'scheduler.log', BASE/'weighted-broad-recovery-v1/followup_scheduler.log']
paths += [BASE/'weighted-broad-recovery-v1'/('_'.join(c)+'.log') for c in
    list(itertools.product(('lora','frozen'), ('broad','step'), ('adaptive','uniform')))[2:]]
paths += [BASE/'streaming-followup-v2'/(c+'.log') for c in ('off_event1','on_event01','on_event1')]
viewer = subprocess.Popen(['tail','-n','8','-F',*[str(p) for p in paths]], stderr=subprocess.DEVNULL)
try:
    wait = select.poll(); wait.register(fd, select.POLLIN); wait.poll()
finally:
    viewer.terminate(); viewer.wait(); os.close(fd)
print((BASE/'weighted-broad-recovery-v1/status.json').read_text(), flush=True)
