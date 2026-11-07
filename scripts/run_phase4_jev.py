"""Cache heldout Jev decisions independently; never expose them to training."""
import json,subprocess,time
from pathlib import Path
ROOT=Path('/workspace/phase4-control');deadline=time.time()+8*3600
for attempt in range(3):
 try:p=subprocess.run(['/workspace/.venv/bin/python','-m','finance_decisions.phase4_eval','--role','jev'],timeout=max(1,deadline-time.time()))
 except subprocess.TimeoutExpired:
  (ROOT/'jev-terminal.json').write_text(json.dumps({'status':'timeout','time':time.time()}));raise SystemExit(1)
 if p.returncode==0:break
 if attempt<2:time.sleep(30*(attempt+1))
(ROOT/'jev-terminal.json').write_text(json.dumps({'status':'complete' if p.returncode==0 else 'failed','returncode':p.returncode,'time':time.time()}))
