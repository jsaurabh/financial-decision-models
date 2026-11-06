"""Queue paraphrase hardening after V2; keep fresh-probe labels out of training."""
import os
import subprocess
import tarfile
from pathlib import Path

root = Path('/content/financial-decision-models')
with tarfile.open('/content/finance-robustness.tar.gz') as archive:
    archive.extractall(root, filter='data')
token = Path('/content/.finance-hf-token')
environment = dict(os.environ)
environment['HF_TOKEN'] = token.read_text().strip(); token.unlink()
environment.update(PYTHONUNBUFFERED='1', TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='8')
driver = r'''
import json,time,subprocess,sys
from pathlib import Path
root=Path('/content/financial-decision-models')
status=root/'robustness-status.json'
status.write_text(json.dumps({'status':'waiting_for_v2','time':time.time()}))
for attempt in range(240):
 prior=json.loads((root/'hardening-status.json').read_text())
 if prior['status']=='complete': break
 if prior['status']=='failed': raise RuntimeError('V2 failed; inspect hardening.log')
 time.sleep(5)
else: raise TimeoutError('V2 did not complete within 20 minutes')
commands=[
 [sys.executable,'-m','finance_decisions.harden_quality','--starting-artifact','runs/gemma4-finance-v2/selected',
  '--out','runs/gemma4-finance-v3','--hf-repo','jsaurabh/gemma4-e2b-finance-decisions-v3','--augment','--steps','700'],
 [sys.executable,'-m','finance_decisions.evaluate_quality','--artifact','runs/gemma4-finance-v3/release',
  '--out','results/gemma4-finance-v3','--hf-repo','jsaurabh/gemma4-e2b-finance-decisions-v3','--fresh-probe'],
]
for stage,cmd in enumerate(commands):
 status.write_text(json.dumps({'status':'running','stage':stage,'time':time.time()}))
 result=subprocess.run(cmd,cwd=root)
 if result.returncode:
  status.write_text(json.dumps({'status':'failed','stage':stage,'returncode':result.returncode,'time':time.time()}))
  sys.exit(result.returncode)
status.write_text(json.dumps({'status':'complete','time':time.time()}))
'''
(root / '_robustness_driver.py').write_text(driver)
with (root / 'robustness.log').open('w') as log:
    p = subprocess.Popen([os.sys.executable, str(root / '_robustness_driver.py')], cwd=root,
                         env=environment, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
print({'pid': p.pid, 'status': 'paraphrase_quality_continuation_queued'})
