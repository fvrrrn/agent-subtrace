"""Inotify/pidfd evidence for the first fresh-cell optimizer steps."""
import ctypes
import json
import os
from pathlib import Path
import re
import select

ROOT=Path('/var/local/danefedov/experiments/agent-safety/runs/fresh-factorial-v1')
pid=json.loads((ROOT.parent/'fresh-factorial-v1-launch/detached.json').read_text())['pid']
pidfd=os.pidfd_open(pid)
libc=ctypes.CDLL(None,use_errno=True)
notify=libc.inotify_init1(os.O_NONBLOCK)
assert notify>=0 and libc.inotify_add_watch(notify,str(ROOT).encode(),0x2|0x8|0x100|0x80)>=0
poll=select.poll();poll.register(notify,select.POLLIN);poll.register(pidfd,select.POLLIN)
log=ROOT/'lora_step.log'
while True:
    text=log.read_text(errors='replace') if log.exists() else ''
    counts=[int(n) for n in re.findall(r'(\d+)/2009',text)]
    if counts and max(counts)>=8:
        status=json.loads((ROOT/'status.json').read_text()); child=status['child_pid']
        report={'scheduler_pid':pid,'child_pid':child,'completed_dialogs':max(counts),
            'optimizer_steps_lower_bound':max(counts)//4,
            'scheduler_fds':{str(i):os.readlink(f'/proc/{pid}/fd/{i}') for i in (0,1,2)},
            'child_fds':{str(i):os.readlink(f'/proc/{child}/fd/{i}') for i in (0,1,2)},
            'gate':json.loads((ROOT/'lora_step/gate.json').read_text()),
            'fresh_init_gate':json.loads((ROOT/'lora_step/fresh_init_gate.json').read_text())}
        with (ROOT/'first_steps_evidence.json').open('x') as stream:json.dump(report,stream,indent=2)
        print(json.dumps({k:v for k,v in report.items() if k!='gate'},indent=2),flush=True)
        break
    events=poll.poll()
    if any(fd==pidfd for fd,_ in events):raise RuntimeError((ROOT/'status.json').read_text())
    if any(fd==notify for fd,_ in events):os.read(notify,65536)
os.close(notify);os.close(pidfd)
