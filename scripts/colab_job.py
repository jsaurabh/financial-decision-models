"""Run in the provisioned Colab G4 kernel after uploading source.tar.gz.

The credential file is transferred separately, never embedded in notebook code.
"""
import os
import subprocess
import tarfile
from pathlib import Path

root = Path('/content/financial-decision-models')
root.mkdir(exist_ok=True)
with tarfile.open('/content/finance-source.tar.gz') as archive:
    archive.extractall(root, filter='data')
token_path = Path('/content/.finance-hf-token')
environment = dict(os.environ)
environment['HF_TOKEN'] = token_path.read_text().strip()
token_path.unlink()
environment['TOKENIZERS_PARALLELISM'] = 'false'
environment['PYTHONUNBUFFERED'] = '1'
environment['OMP_NUM_THREADS'] = '8'
driver = r'''
import json,subprocess,sys,time
from pathlib import Path
root=Path('/content/financial-decision-models')
status=root/'job-status.json'
commands=[
 [sys.executable,'-m','pip','uninstall','-y','torchao'],
 [sys.executable,'-m','pip','install','-q','-r','requirements.txt'],
 [sys.executable,'-m','finance_decisions.train','--steps','700'],
]
for stage,cmd in enumerate(commands):
 status.write_text(json.dumps({'stage':stage,'command':cmd,'status':'running','time':time.time()}))
 result=subprocess.run(cmd,cwd=root)
 if result.returncode:
  status.write_text(json.dumps({'stage':stage,'status':'failed','returncode':result.returncode,'time':time.time()}))
  sys.exit(result.returncode)
status.write_text(json.dumps({'status':'complete','time':time.time()}))
'''
(root / '_driver.py').write_text(driver)
with (root / 'training.log').open('w') as log:
    process = subprocess.Popen([os.sys.executable, str(root / '_driver.py')], cwd=root,
                               env=environment, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
print({'pid': process.pid, 'log': str(root / 'training.log')})
