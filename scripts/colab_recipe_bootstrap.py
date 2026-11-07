"""Run one detached Gemma E2B diagnostic pipeline, with no test archive."""
import os
import subprocess
from pathlib import Path

token=Path('/content/.recipe-hf-token')
env=dict(os.environ);env['HF_TOKEN']=token.read_text().strip();token.unlink()
env.update(PYTHONUNBUFFERED='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='8',
           HF_HUB_DOWNLOAD_TIMEOUT='90')
driver=r'''
import hashlib,json,os,subprocess,sys,tarfile,time,traceback
from pathlib import Path
root=Path('/content/recipe-job');status=Path('/content/recipe-status.json')
def mark(state,**kw):
 p=status.with_suffix('.tmp');p.write_text(json.dumps({'status':state,'time':time.time(),**kw}));p.replace(status)
try:
 mark('validating_sealed_job');ready=json.loads(Path('/content/recipe-ready.json').read_text());root.mkdir(exist_ok=True)
 for name,expected in ready['sha256'].items():
  assert hashlib.sha256((Path('/content')/name).read_bytes()).hexdigest()==expected
  with tarfile.open(Path('/content')/name) as archive:archive.extractall(root,filter='data')
 assert not (root/'data/private/phase4').exists()
 assert set(p.stem for p in (root/'data/private/recipe-audit').glob('*.jsonl'))=={'train','development','memorization','controls'}
 mark('installing')
 subprocess.run([sys.executable,'-m','pip','uninstall','-y','torchao'],check=True)
 subprocess.run([sys.executable,'-m','pip','install','-r',str(root/'requirements-recipe-audit.txt')],check=True)
 subprocess.run([sys.executable,'-c','from transformers import Gemma4ForConditionalGeneration; import torch,torchvision; print({"torch":torch.__version__,"torchvision":torchvision.__version__})'],check=True)
 for attempt in range(3):
  mark('starting_pipeline',attempt=attempt)
  result=subprocess.run([sys.executable,'-m','finance_decisions.recipe_audit_train'],cwd=root)
  if result.returncode==0:break
  if attempt==2:raise RuntimeError('Bounded recipe diagnostic failed after retries')
  mark('retry_wait',attempt=attempt+1);time.sleep(30*(attempt+1))
 mark('pipeline_complete')
except Exception as exc:
 traceback.print_exc();mark('failed',error_type=type(exc).__name__)
finally:
 try:
  from huggingface_hub import HfApi
  HfApi().upload_file(repo_id='jsaurabh/financial-decision-gemma-recipe-audit-data',repo_type='dataset',path_or_fileobj=str(status),path_in_repo='runtime/status.json')
 except Exception:pass
'''
Path('/content/recipe-driver.py').write_text(driver)
with Path('/content/recipe-worker.log').open('w') as log:
    process=subprocess.Popen([os.sys.executable,'/content/recipe-driver.py'],env=env,
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
Path('/content/recipe-driver.pid').write_text(str(process.pid))
print({'pid':process.pid,'status':'gemma_only_recipe_diagnostic_started'})
