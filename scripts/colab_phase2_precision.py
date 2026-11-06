"""Wait for Qwen to finish, then investigate observed order instability."""
import os,subprocess
from pathlib import Path
root=Path('/content/financial-decision-models')
token=Path('/content/.finance-hf-token');env=dict(os.environ);env['HF_TOKEN']=token.read_text().strip();token.unlink()
env.update(PYTHONUNBUFFERED='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='8')
driver=r'''
import json,subprocess,sys,time
from pathlib import Path
root=Path('/content/financial-decision-models');status=root/'precision-status.json'
while True:
 state=json.loads((root/'qwen-retry-status.json').read_text())
 if state['status']=='failed':sys.exit('Qwen evaluation failed')
 if state['status']=='complete':break
 time.sleep(15)
for stage,cmd in enumerate([[sys.executable,'-m','pip','install','-r','requirements.txt'],[sys.executable,'-m','finance_decisions.phase2_precision_audit']]):
 status.write_text(json.dumps({'status':'running','stage':stage,'time':time.time()}))
 result=subprocess.run(cmd,cwd=root)
 if result.returncode:
  status.write_text(json.dumps({'status':'failed','stage':stage,'time':time.time()}));sys.exit(result.returncode)
status.write_text(json.dumps({'status':'complete','time':time.time()}))
'''
(root/'_precision_driver.py').write_text(driver)
with (root/'precision.log').open('w') as log:
 p=subprocess.Popen([os.sys.executable,str(root/'_precision_driver.py')],cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print({'pid':p.pid,'status':'precision_audit_scheduled'})
