"""Detached phase4 fitting, then separately unpacked frozen test evaluation."""
import os,subprocess
from pathlib import Path
role=Path('/content/phase4-role.txt').read_text().strip();assert role in ['gemma','qwen']
secret=Path('/content/.phase4-hf-token');env=dict(os.environ);env['HF_TOKEN']=secret.read_text().strip();secret.unlink()
env.update(PYTHONUNBUFFERED='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='8',PHASE4_FAMILY=role,HF_HUB_DOWNLOAD_TIMEOUT='90')
driver=r'''
import hashlib,json,os,subprocess,sys,tarfile,time,traceback
from pathlib import Path
role=os.environ['PHASE4_FAMILY'];root=Path('/content/phase4-job');status=Path('/content/phase4-status.json')
def mark(state,**kw):
 p=status.with_suffix('.tmp');p.write_text(json.dumps({'role':role,'status':state,'time':time.time(),**kw}));p.replace(status)
try:
 mark('validating_sealed_job');ready=json.loads(Path('/content/phase4-ready.json').read_text());root.mkdir(exist_ok=True)
 for name in ['phase4-code.tar.gz','phase4-training.tar.gz','phase4-tests.tar.gz']:
  assert hashlib.sha256((Path('/content')/name).read_bytes()).hexdigest()==ready['sha256'][name]
 for name in ['phase4-code.tar.gz','phase4-training.tar.gz']:
  with tarfile.open(Path('/content')/name) as archive:archive.extractall(root,filter='data')
 assert not (root/'data/private/phase4/core_test.jsonl').exists()
 assert not (root/'data/private/phase4/transfer_test.jsonl').exists()
 mark('installing')
 subprocess.run([sys.executable,'-m','pip','uninstall','-y','torchao'],check=True)
 subprocess.run([sys.executable,'-m','pip','install','-r',str(root/'requirements-phase4.txt')],check=True)
 subprocess.run([sys.executable,'-c','from transformers import Gemma4ForConditionalGeneration,Qwen3_5ForConditionalGeneration; import torch,torchvision; print({"torch":torch.__version__,"torchvision":torchvision.__version__})'],check=True)
 for stage,module in [('training','finance_decisions.phase4_train'),('evaluation','finance_decisions.phase4_eval')]:
  if stage=='evaluation':
   summary=json.loads((root/'runs'/('phase4-'+role)/'summary.json').read_text());assert summary['status']=='training_complete'
   # First access to test rows happens AFTER selection/calibration/release.
   with tarfile.open('/content/phase4-tests.tar.gz') as archive:archive.extractall(root,filter='data')
  mark('starting_'+stage)
  for attempt in range(3):
   args=['--family',role] if stage=='training' else ['--role',role]
   result=subprocess.run([sys.executable,'-m',module,*args],cwd=root)
   if result.returncode==0:break
   if attempt==2:raise RuntimeError(stage+' failed after bounded local retries')
   mark(stage+'_retry_wait',attempt=attempt+1);time.sleep(30*(attempt+1))
 mark('pipeline_complete')
except Exception as exc:
 traceback.print_exc();mark('failed',error_type=type(exc).__name__)
finally:
 try:
  from huggingface_hub import HfApi
  HfApi().upload_file(repo_id='jsaurabh/financial-decision-phase4',repo_type='dataset',path_or_fileobj=str(status),path_in_repo='runtime/'+role+'.json')
 except Exception:pass
'''
path=Path('/content/phase4-driver.py');path.write_text(driver)
with Path('/content/phase4-worker.log').open('w') as log:
 p=subprocess.Popen([os.sys.executable,str(path)],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print({'role':role,'pid':p.pid,'status':'sealed_aligned_training_pipeline_started'})
