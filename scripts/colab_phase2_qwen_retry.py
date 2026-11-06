"""Standalone G4 Qwen evaluation; Gemma results survive a Colab preemption."""
import os,subprocess,tarfile
from pathlib import Path
root=Path('/content/financial-decision-models');root.mkdir(exist_ok=True)
with tarfile.open('/content/finance-phase2-qwen-retry.tar.gz') as a:a.extractall(root,filter='data')
token=Path('/content/.finance-hf-token');env=dict(os.environ);env['HF_TOKEN']=token.read_text().strip();token.unlink()
env.update(PYTHONUNBUFFERED='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='8')
driver=r'''
import json,subprocess,sys,time,shutil
from pathlib import Path
import psutil
root=Path('/content/financial-decision-models');status=root/'qwen-retry-status.json'
(root/'hardware.json').write_text(json.dumps({'host_ram':psutil.virtual_memory()._asdict(),'disk':list(shutil.disk_usage('/content')),'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total','--format=csv,noheader'],text=True)},indent=2))
commands=[
 [sys.executable,'-m','pip','uninstall','-y','torchao'],
 [sys.executable,'-m','pip','install','uv'],
 [sys.executable,'-m','uv','venv','--seed','--system-site-packages','--python',sys.executable,'/content/qwen-env'],
 ['/content/qwen-env/bin/python','-m','pip','install','-r','references/qwen27b/requirements.txt'],
 ['/content/qwen-env/bin/python','-m','finance_decisions.phase2_eval','--qwen-source',str(root/'references/qwen27b'),
  '--out','results/phase2/qwen','--hf-repo','jsaurabh/gemma4-finance-phase2-causal'],
]
for stage,cmd in enumerate(commands):
 status.write_text(json.dumps({'status':'running','stage':stage,'time':time.time()}))
 result=subprocess.run(cmd,cwd=root)
 if result.returncode:
  status.write_text(json.dumps({'status':'failed','stage':stage,'returncode':result.returncode,'time':time.time()}));sys.exit(result.returncode)
status.write_text(json.dumps({'status':'complete','time':time.time()}))
'''
(root/'_qwen_retry.py').write_text(driver)
with (root/'qwen-retry.log').open('w') as log:
 p=subprocess.Popen([os.sys.executable,str(root/'_qwen_retry.py')],cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print({'pid':p.pid,'status':'qwen_retry_started'})
