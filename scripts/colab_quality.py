"""Start quality evaluation after the bounded pilot finishes on the same G4 VM."""
import os
import subprocess
import tarfile
from pathlib import Path

root = Path('/content/financial-decision-models')
with tarfile.open('/content/finance-quality.tar.gz') as archive:
    archive.extractall(root, filter='data')
token = Path('/content/.finance-hf-token')
environment = dict(os.environ)
environment['HF_TOKEN'] = token.read_text().strip(); token.unlink()
environment.update(PYTHONUNBUFFERED='1', TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='8')
driver = r'''
import json,time,subprocess,sys
from pathlib import Path
root=Path('/content/financial-decision-models')
for attempt in range(240):
 status=json.loads((root/'job-status.json').read_text())
 if status['status']=='complete': break
 if status['status']=='failed': raise RuntimeError('Pilot failed; inspect training.log')
 time.sleep(5)
else: raise TimeoutError('Pilot has not completed in 20 minutes')
subprocess.run([sys.executable,'-m','finance_decisions.evaluate_quality'],cwd=root,check=True)
(root/'quality-status.json').write_text(json.dumps({'status':'complete','time':time.time()}))
'''
(root / '_quality_driver.py').write_text(driver)
with (root / 'quality.log').open('w') as log:
    p = subprocess.Popen([os.sys.executable, str(root / '_quality_driver.py')], cwd=root,
                         env=environment, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
print({'pid': p.pid, 'status': 'quality_evaluation_queued'})
