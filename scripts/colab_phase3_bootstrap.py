"""Owned Colab training worker: pinned install, sealed inputs, bounded retries."""
import os
import subprocess
from pathlib import Path

secret=Path('/content/.phase3-hf-token');env=dict(os.environ)
env['HF_TOKEN']=secret.read_text().strip();secret.unlink()
env.update(PYTHONUNBUFFERED='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='8')
driver=r'''
import hashlib,json,os,subprocess,sys,tarfile,time,traceback
from pathlib import Path
from huggingface_hub import HfApi,snapshot_download
status=Path('/content/phase3-status.json');root=Path('/content/phase3')
def mark(state,**kw):
 p=status.with_suffix('.tmp');p.write_text(json.dumps({'status':state,'time':time.time(),**kw}));p.replace(status)
try:
 mark('installing')
 subprocess.run([sys.executable,'-m','pip','uninstall','-y','torchao'],check=True)
 subprocess.run([sys.executable,'-m','pip','install','-r','/content/phase3-requirements.txt'],check=True)
 subprocess.run([sys.executable,'-c','from transformers import Gemma4ForConditionalGeneration; import torch,torchvision; print({"torch":torch.__version__,"torchvision":torchvision.__version__})'],check=True)
 mark('prefetching_weights')
 snapshot_download('google/gemma-4-E2B-it',revision='4adf15748de9e1a0097b6274cee9c15836c41c37')
 mark('waiting_for_sealed_training_job');deadline=time.time()+3600
 while not Path('/content/phase3-ready.json').exists():
  if time.time()>deadline:raise TimeoutError('Training corpus did not arrive within one hour')
  time.sleep(5)
 ready=json.loads(Path('/content/phase3-ready.json').read_text())
 root.mkdir(exist_ok=True)
 for filename in ['phase3-code.tar.gz','phase3-data.tar.gz']:
  path=Path('/content')/filename
  assert hashlib.sha256(path.read_bytes()).hexdigest()==ready['sha256'][filename]
  with tarfile.open(path) as archive:archive.extractall(root,filter='data')
 assert hashlib.sha256((root/'data/phase3-manifest.json').read_bytes()).hexdigest()==ready['manifest_sha256']
 mark('starting_training')
 for attempt in range(3):
  result=subprocess.run([sys.executable,'-m','finance_decisions.phase3_train','--resume'],cwd=root)
  if result.returncode==0:break
  if attempt==2:raise RuntimeError(f'Trainer exited {result.returncode} after bounded local retries')
  mark('retry_wait',attempt=attempt+1);time.sleep(30*(attempt+1))
 mark('complete')
except Exception as exc:
 traceback.print_exc();mark('failed',error_type=type(exc).__name__,error=str(exc))
finally:
 try:
  HfApi().upload_file(repo_id='jsaurabh/financial-decision-phase3-pilot',repo_type='dataset',path_or_fileobj=str(status),path_in_repo='runtime/status.json')
 except Exception:pass
'''
path=Path('/content/phase3-driver.py');path.write_text(driver)
with Path('/content/phase3-worker.log').open('w') as log:
    worker=subprocess.Popen([os.sys.executable,str(path)],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print({'pid':worker.pid,'status':'phase3_training_worker_started'})
