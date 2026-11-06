"""Prepare a separate Qwen runtime and pinned downloads without using the GPU."""
import os
import subprocess
import tarfile
from pathlib import Path
root=Path('/content/financial-decision-models')
with tarfile.open('/content/finance-qwen-reference.tar.gz') as a:a.extractall(root,filter='data')
token=Path('/content/.finance-hf-token');env=dict(os.environ);env['HF_TOKEN']=token.read_text().strip();token.unlink()
env.update(PYTHONUNBUFFERED='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='8')
driver=r'''
import json,subprocess,sys,time
from pathlib import Path
root=Path('/content/financial-decision-models');status=root/'qwen-prepare-status.json'
commands=[
 [sys.executable,'-m','venv','--system-site-packages','/content/qwen-env'],
 ['/content/qwen-env/bin/python','-m','pip','install','-r','references/qwen27b/requirements.txt'],
 ['/content/qwen-env/bin/python','-c',"from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3.8-27B',revision='1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0'); snapshot_download('jsaurabh/qwen-decision-27b-noncausal-lora',revision='b2eefb929a883136b3bd56ce70999fdac656ee8a')"],
]
for stage,cmd in enumerate(commands):
 status.write_text(json.dumps({'status':'running','stage':stage,'time':time.time()}))
 result=subprocess.run(cmd,cwd=root)
 if result.returncode:
  status.write_text(json.dumps({'status':'failed','stage':stage,'returncode':result.returncode,'time':time.time()}));sys.exit(result.returncode)
status.write_text(json.dumps({'status':'ready','time':time.time()}))
while True:
 state=json.loads((root/'phase2-status.json').read_text())
 if state['status']=='failed':
  status.write_text(json.dumps({'status':'blocked_by_gemma_failure','time':time.time()}));sys.exit(1)
 if state['status']=='gemma_complete':break
 time.sleep(15)
status.write_text(json.dumps({'status':'evaluating','time':time.time()}))
cmd=['/content/qwen-env/bin/python','-m','finance_decisions.phase2_eval','--qwen-source',str(root/'references/qwen27b'),
 '--out','results/phase2/qwen','--hf-repo','jsaurabh/gemma4-finance-phase2-causal']
result=subprocess.run(cmd,cwd=root)
status.write_text(json.dumps({'status':'complete' if result.returncode==0 else 'failed','stage':'evaluation','returncode':result.returncode,'time':time.time()}))
'''
(root/'_qwen_prepare.py').write_text(driver)
with (root/'qwen-prepare.log').open('w') as log:
 p=subprocess.Popen([os.sys.executable,str(root/'_qwen_prepare.py')],cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print({'pid':p.pid,'status':'qwen_prepare_started'})
