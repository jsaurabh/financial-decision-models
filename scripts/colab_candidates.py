"""Train the shared candidate scorer and evaluate on the sealed public holdout."""
import os
import subprocess
import tarfile
from pathlib import Path

root=Path('/content/financial-decision-models')
with tarfile.open('/content/finance-candidates.tar.gz') as archive:
    archive.extractall(root,filter='data')
token=Path('/content/.finance-hf-token')
environment=dict(os.environ);environment['HF_TOKEN']=token.read_text().strip();token.unlink()
environment.update(PYTHONUNBUFFERED='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='8',
    DECISION_ARTIFACT=str(root/'runs/gemma4-finance-v4/release'),DECISION_DEVICE='cuda',
    DECISION_MODEL_ID='jsaurabh/gemma4-e2b-finance-decisions-v4',DECISION_RESULT_DIR='results/gemma4-finance-v4')
driver=r'''
import json,subprocess,sys,time
from pathlib import Path
root=Path('/content/financial-decision-models');status=root/'candidate-status.json'
commands=[
 [sys.executable,'-m','finance_decisions.harden_quality','--starting-artifact','runs/gemma4-finance-v3/selected',
  '--out','runs/gemma4-finance-v4','--hf-repo','jsaurabh/gemma4-e2b-finance-decisions-v4','--augment','--candidate-head','--steps','700'],
 [sys.executable,'-m','finance_decisions.evaluate_quality','--artifact','runs/gemma4-finance-v4/release',
  '--out','results/gemma4-finance-v4','--hf-repo','jsaurabh/gemma4-e2b-finance-decisions-v4','--fresh-probe','--public-holdout'],
 [sys.executable,'-m','scripts.api_smoke'],
]
for stage,cmd in enumerate(commands):
 status.write_text(json.dumps({'status':'running','stage':stage,'time':time.time()}))
 result=subprocess.run(cmd,cwd=root)
 if result.returncode:
  status.write_text(json.dumps({'status':'failed','stage':stage,'returncode':result.returncode,'time':time.time()}));sys.exit(result.returncode)
status.write_text(json.dumps({'status':'complete','time':time.time()}))
'''
(root/'_candidate_driver.py').write_text(driver)
with (root/'candidates.log').open('w') as log:
    p=subprocess.Popen([os.sys.executable,str(root/'_candidate_driver.py')],cwd=root,env=environment,
        stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print({'pid':p.pid,'status':'candidate_head_training_started'})
