"""Start the frozen matched experiment in a detached Colab G4 worker."""
import os
import subprocess
import tarfile
from pathlib import Path

root=Path('/content/financial-decision-models');root.mkdir(exist_ok=True)
with tarfile.open('/content/finance-phase2.tar.gz') as archive:archive.extractall(root,filter='data')
token=Path('/content/.finance-hf-token');env=dict(os.environ);env['HF_TOKEN']=token.read_text().strip();token.unlink()
env.update(PYTHONUNBUFFERED='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='8')
driver=r'''
import json,subprocess,sys,time,traceback
from pathlib import Path
root=Path('/content/financial-decision-models');status=root/'phase2-status.json'
commands=[
 [sys.executable,'-m','pip','uninstall','-y','torchao'],
 [sys.executable,'-m','pip','install','-r','requirements.txt'],
 [sys.executable,'-m','finance_decisions.phase2_train','--mode','independent-causal','--out','runs/phase2-causal','--hf-repo','jsaurabh/gemma4-finance-phase2-causal'],
 [sys.executable,'-m','finance_decisions.phase2_train','--mode','joint-bidirectional','--out','runs/phase2-joint','--hf-repo','jsaurabh/gemma4-finance-phase2-joint'],
 [sys.executable,'-m','finance_decisions.phase2_eval','--artifact','runs/phase2-causal/release','--out','results/phase2/causal','--hf-repo','jsaurabh/gemma4-finance-phase2-causal'],
 [sys.executable,'-m','finance_decisions.phase2_eval','--artifact','runs/phase2-joint/release','--out','results/phase2/joint','--hf-repo','jsaurabh/gemma4-finance-phase2-joint'],
 [sys.executable,'-m','finance_decisions.phase2_eval','--v4','--out','results/phase2/v4','--hf-repo','jsaurabh/gemma4-finance-phase2-causal'],
]
for stage,cmd in enumerate(commands):
 status.write_text(json.dumps({'status':'running','stage':stage,'command':cmd,'time':time.time()}))
 result=subprocess.run(cmd,cwd=root)
 if result.returncode:
  status.write_text(json.dumps({'status':'failed','stage':stage,'returncode':result.returncode,'time':time.time()}));sys.exit(result.returncode)
status.write_text(json.dumps({'status':'gemma_complete','time':time.time()}))
'''
(root/'_phase2_driver.py').write_text(driver)
with (root/'phase2.log').open('w') as log:
 p=subprocess.Popen([os.sys.executable,str(root/'_phase2_driver.py')],cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print({'pid':p.pid,'status':'phase2_started'})
