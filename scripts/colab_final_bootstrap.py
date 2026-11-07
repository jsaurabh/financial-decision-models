"""Pinned, evaluation-only GPU worker; data and code arrive as sealed packages."""
import os
import subprocess
from pathlib import Path

secret=Path('/content/.final-hf-token');env=dict(os.environ)
env['HF_TOKEN']=secret.read_text().strip();secret.unlink()
env.update(PYTHONUNBUFFERED='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='8')
driver=r'''
import hashlib,json,subprocess,sys,tarfile,time,traceback
from pathlib import Path
from huggingface_hub import HfApi,snapshot_download
root=Path('/content/final-eval');status=Path('/content/final-status.json')
def mark(state,**kw):
 p=status.with_suffix('.tmp');p.write_text(json.dumps({'status':state,'time':time.time(),**kw}));p.replace(status)
try:
 mark('installing')
 subprocess.run([sys.executable,'-m','pip','uninstall','-y','torchao'],check=True)
 subprocess.run([sys.executable,'-m','pip','install','-r','/content/final-requirements.txt'],check=True)
 subprocess.run([sys.executable,'-c','from transformers import Gemma4ForConditionalGeneration; import torch,torchvision; print({"torch":torch.__version__,"torchvision":torchvision.__version__})'],check=True)
 mark('prefetching_weights')
 snapshot_download('google/gemma-4-E2B-it',revision='4adf15748de9e1a0097b6274cee9c15836c41c37')
 snapshot_download('jsaurabh/gemma4-finance-phase3-pilot',revision='1eec2a39d4d6a3106d5175b73539293ae3259cf0',allow_patterns=['run/release/**'])
 mark('waiting_for_sealed_final_job');deadline=time.time()+3600
 while not Path('/content/final-ready.json').exists():
  if time.time()>deadline:raise TimeoutError('No sealed final-evaluation corpus within one hour')
  time.sleep(5)
 ready=json.loads(Path('/content/final-ready.json').read_text());root.mkdir(exist_ok=True)
 for name in ['final-code.tar.gz','final-data.tar.gz']:
  path=Path('/content')/name
  assert hashlib.sha256(path.read_bytes()).hexdigest()==ready['sha256'][name]
  with tarfile.open(path) as archive:archive.extractall(root,filter='data')
 mark('starting_evaluation')
 for attempt in range(3):
  result=subprocess.run([sys.executable,'-m','finance_decisions.final_eval','--role','gemma'],cwd=root)
  if result.returncode==0:break
  if attempt==2:raise RuntimeError(f'Evaluator exited {result.returncode} after bounded local retries')
  mark('retry_wait',attempt=attempt+1);time.sleep(30*(attempt+1))
 mark('complete')
except Exception as exc:
 traceback.print_exc();mark('failed',error_type=type(exc).__name__,error=str(exc))
finally:
 try:HfApi().upload_file(repo_id='jsaurabh/financial-decision-final-eval',repo_type='dataset',path_or_fileobj=str(status),path_in_repo='runtime/gemma.json')
 except Exception:pass
'''
path=Path('/content/final-driver.py');path.write_text(driver)
with Path('/content/final-worker.log').open('w') as log:
    worker=subprocess.Popen([os.sys.executable,str(path)],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print({'pid':worker.pid,'status':'frozen_final_evaluation_worker_started'})
