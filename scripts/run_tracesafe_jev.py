"""Independent, bounded CPU/API runner for the TraceSafe addendum."""
import json,subprocess,time
from pathlib import Path
ROOT=Path('/workspace/tracesafe-eval-control');deadline=time.time()+3*3600
for attempt in range(3):
    try:p=subprocess.run(['/workspace/.venv/bin/python','-m','finance_decisions.tracesafe_eval','--role','jev'],timeout=max(1,deadline-time.time()))
    except subprocess.TimeoutExpired:
        (ROOT/'jev-terminal.json').write_text(json.dumps({'status':'timeout','time':time.time()}));raise SystemExit(1)
    if p.returncode==0:break
    if attempt<2:time.sleep(30*(attempt+1))
(ROOT/'jev-terminal.json').write_text(json.dumps({'status':'complete' if p.returncode==0 else 'failed','returncode':p.returncode,'time':time.time()}))
