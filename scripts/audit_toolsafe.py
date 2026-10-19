"""Download pinned published step annotations and audit available supervision."""
import collections
import hashlib
import json
from pathlib import Path
import urllib.request

REV = '46358fa424a927a895c6c8322f99032c4eb5155e'
ROOT = Path(__file__).resolve().parents[1] / 'artifacts/kev08_step_training_v1'
FILES = ['agentharm-traj/benign_steps.json', 'agentharm-traj/harmful_steps.json',
         'agentdojo-traj/banking.json', 'agentdojo-traj/slack.json',
         'agentdojo-traj/travel.json', 'agentdojo-traj/workspace.json',
         'asb-traj/test/DPI_attack_success.json', 'asb-traj/test/OPI_attack_success.json',
         'asb-traj/test/atttack_failure.json']

ROOT.mkdir(parents=True, exist_ok=True)
audit = []
for name in FILES:
    path = ROOT / 'raw' / name
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        url = f'https://media.githubusercontent.com/media/MurrayTom/ToolSafe/{REV}/TS-Bench/{name}'
        path.write_bytes(urllib.request.urlopen(url, timeout=120).read())
    data = json.loads(path.read_text())
    row = {'file': name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
           'count': len(data), 'keys': list(data[0]),
           'scores': dict(collections.Counter(str(r.get('score')) for r in data))}
    audit.append(row)
    print(json.dumps(row), flush=True)
(ROOT / 'source_audit.json').write_text(json.dumps({'revision': REV, 'files': audit}, indent=2))
