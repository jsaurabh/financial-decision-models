"""Run only the sealed TraceSafe addendum on an isolated, bounded GPU."""
import os,subprocess
from pathlib import Path

secret=Path('/content/.tracesafe-hf-token');env=dict(os.environ)
env['HF_TOKEN']=secret.read_text().strip();secret.unlink()
env.update(PYTHONUNBUFFERED='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='8')
driver=r'''
import hashlib,json,subprocess,sys,tarfile,time,traceback
from pathlib import Path
root=Path('/content/tracesafe-eval');status=Path('/content/final-status.json')
def mark(state,**kw):
 p=status.with_suffix('.tmp');p.write_text(json.dumps({'status':state,'time':time.time(),**kw}));p.replace(status)
try:
 mark('validating_sealed_job')
 ready=json.loads(Path('/content/tracesafe-ready.json').read_text());root.mkdir(exist_ok=True)
 for name in ['tracesafe-code.tar.gz','tracesafe-data.tar.gz']:
  path=Path('/content')/name
  assert hashlib.sha256(path.read_bytes()).hexdigest()==ready['sha256'][name]
  with tarfile.open(path) as archive:archive.extractall(root,filter='data')
 mark('installing')
 subprocess.run([sys.executable,'-m','pip','uninstall','-y','torchao'],check=True)
 subprocess.run([sys.executable,'-m','pip','install','-r','/content/tracesafe-requirements.txt'],check=True)
 subprocess.run([sys.executable,'-c','from transformers import Gemma4ForConditionalGeneration; import torch,torchvision; print({"torch":torch.__version__,"torchvision":torchvision.__version__})'],check=True)
 mark('starting_evaluation')
 for attempt in range(3):
  result=subprocess.run([sys.executable,'-m','finance_decisions.tracesafe_eval','--role','gemma'],cwd=root)
  if result.returncode==0:break
  if attempt==2:raise RuntimeError('Evaluator failed after bounded local retries')
  mark('retry_wait',attempt=attempt+1);time.sleep(30*(attempt+1))
 mark('complete')
except Exception as exc:
 traceback.print_exc();mark('failed',error_type=type(exc).__name__)
finally:
 try:
  from huggingface_hub import HfApi
  HfApi().upload_file(repo_id='jsaurabh/financial-decision-tracesafe-eval',repo_type='dataset',path_or_fileobj=str(status),path_in_repo='runtime/gemma.json')
 except Exception:pass
'''
path=Path('/content/tracesafe-driver.py');path.write_text(driver)
with Path('/content/final-worker.log').open('w') as log:
    worker=subprocess.Popen([os.sys.executable,str(path)],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print({'pid':worker.pid,'status':'sealed_tracesafe_evaluation_started'})
