"""Smoke-test the saved release after quality evaluation completes."""
import os
import subprocess
import tarfile
from pathlib import Path

root = Path('/content/financial-decision-models')
with tarfile.open('/content/finance-api.tar.gz') as archive:
    archive.extractall(root, filter='data')
environment = dict(os.environ)
environment.update(PYTHONUNBUFFERED='1', TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='8',
                   DECISION_ARTIFACT=str(root / 'runs/gemma4-finance-v3/release'), DECISION_DEVICE='cuda')
driver = r'''
import json,time,subprocess,sys
from pathlib import Path
root=Path('/content/financial-decision-models')
for attempt in range(240):
 state=json.loads((root/'robustness-status.json').read_text())
 if state['status']=='complete':break
 if state['status']=='failed':raise RuntimeError('Quality evaluation failed; inspect robustness.log')
 time.sleep(5)
else:raise TimeoutError('V3 did not complete in 20 minutes')
subprocess.run([sys.executable,'-m','scripts.api_smoke'],cwd=root,check=True)
'''
(root / '_api_driver.py').write_text(driver)
with (root / 'api-smoke.log').open('w') as log:
    p = subprocess.Popen([os.sys.executable, str(root / '_api_driver.py')], cwd=root,
                         env=environment, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
print({'pid': p.pid, 'status': 'saved_artifact_api_smoke_queued'})
