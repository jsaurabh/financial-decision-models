"""Queue a bounded quality continuation, then evaluate the saved result."""
import os
import subprocess
import tarfile
from pathlib import Path

root = Path('/content/financial-decision-models')
with tarfile.open('/content/finance-harden.tar.gz') as archive:
    archive.extractall(root, filter='data')
token = Path('/content/.finance-hf-token')
environment = dict(os.environ)
environment['HF_TOKEN'] = token.read_text().strip(); token.unlink()
environment.update(PYTHONUNBUFFERED='1', TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='8')
driver = r'''
import json,time,subprocess,sys
from pathlib import Path
root=Path('/content/financial-decision-models')
status=root/'hardening-status.json'
status.write_text(json.dumps({'status':'waiting_for_v1_evaluation','time':time.time()}))
for attempt in range(240):
 if (root/'quality-status.json').exists(): break
 time.sleep(5)
else: raise TimeoutError('V1 quality evaluation did not complete')
commands=[
 [sys.executable,'-m','finance_decisions.harden_quality'],
 [sys.executable,'-m','finance_decisions.evaluate_quality','--artifact','runs/gemma4-finance-v2/release',
  '--out','results/gemma4-finance-v2','--hf-repo','jsaurabh/gemma4-e2b-finance-decisions-v2'],
]
for stage,cmd in enumerate(commands):
 status.write_text(json.dumps({'status':'running','stage':stage,'time':time.time()}))
 result=subprocess.run(cmd,cwd=root)
 if result.returncode:
  status.write_text(json.dumps({'status':'failed','stage':stage,'returncode':result.returncode,'time':time.time()}))
  sys.exit(result.returncode)
status.write_text(json.dumps({'status':'complete','time':time.time()}))
'''
(root / '_harden_driver.py').write_text(driver)
with (root / 'hardening.log').open('w') as log:
    p = subprocess.Popen([os.sys.executable, str(root / '_harden_driver.py')], cwd=root,
                         env=environment, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
print({'pid': p.pid, 'status': 'quality_continuation_queued'})
