"""Background public-eval worker; role is uploaded as /content/public-role.txt."""
import os,subprocess
from pathlib import Path
role=Path('/content/public-role.txt').read_text().strip()
secret=Path('/content/.public-hf-token');env=dict(os.environ);env['HF_TOKEN']=secret.read_text().strip();secret.unlink()
env.update(PYTHONUNBUFFERED='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='8',PUBLIC_EVAL_ROLE=role)
# Only this project's code/manifest is accepted from a private dataset repo.
driver=r'''
import hashlib,json,os,subprocess,sys,tarfile,time,traceback
from pathlib import Path
from huggingface_hub import HfApi,hf_hub_download,snapshot_download
role=os.environ['PUBLIC_EVAL_ROLE'];root=Path('/content/public-eval');root.mkdir(exist_ok=True)
status=Path('/content/public-status.json');api=HfApi();repo='jsaurabh/financial-decision-public-eval'
def mark(state,**kw):
 tmp=status.with_suffix('.tmp');tmp.write_text(json.dumps({'status':state,'role':role,'time':time.time(),**kw}));tmp.replace(status)
try:
 mark('installing')
 req=Path('/content/public-requirements.txt')
 subprocess.run([sys.executable,'-m','pip','uninstall','-y','torchao'],check=True)
 subprocess.run([sys.executable,'-m','pip','install','-r',str(req)],check=True)
 mark('prefetching_weights')
 if role=='qwen':
  snapshot_download('Qwen/Qwen3.8-27B',revision='1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0')
  snapshot_download('jsaurabh/qwen-decision-27b-noncausal-lora',revision='b2eefb929a883136b3bd56ce70999fdac656ee8a')
 else:
  snapshot_download('google/gemma-4-E2B-it',revision='4adf15748de9e1a0097b6274cee9c15836c41c37')
  for repo_id,rev in [('jsaurabh/gemma4-finance-phase2-causal','6b4188a4790bba0beaef19bb48e4b9b18232138d'),('jsaurabh/gemma4-finance-phase2-joint','99b0773bf266fbc4b19189fe451f5f4088021f48')]:
   snapshot_download(repo_id,revision=rev,allow_patterns=['decision_config.json','readout.safetensors','backbone/**','tokenizer/**'])
 mark('waiting_for_sealed_job');deadline=time.time()+3600
 while True:
  try:
   info=api.dataset_info(repo)
   assert info.private
   revision=info.sha
   manifest=json.loads(Path(hf_hub_download(repo,'job.json',repo_type='dataset',revision=revision)).read_text())
   if manifest.get('ready'):break
  except Exception as exc:
   if time.time()>deadline:raise TimeoutError('No sealed job arrived within one hour') from exc
  time.sleep(30)
 path=hf_hub_download(repo,'job.tar.gz',repo_type='dataset',revision=revision)
 assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==manifest['archive_sha256']
 with tarfile.open(path) as archive:archive.extractall(root,filter='data')
 mark('evaluating',job_revision=revision,manifest_sha256=manifest['data_manifest_sha256'])
 result=subprocess.run([sys.executable,'-m','finance_decisions.public_eval','--role',role],cwd=root)
 if result.returncode:raise RuntimeError(f'Evaluator exited {result.returncode}')
 mark('complete',job_revision=revision)
except Exception as exc:
 traceback.print_exc();mark('failed',error_type=type(exc).__name__,error=str(exc))
finally:
 try:
  api.upload_file(repo_id=repo,repo_type='dataset',path_or_fileobj=str(status),path_in_repo=f'progress/{role}.json')
 except Exception:pass
'''
p=Path('/content/public-driver.py');p.write_text(driver)
with Path('/content/public-worker.log').open('w') as log:
 worker=subprocess.Popen([os.sys.executable,str(p)],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print({'pid':worker.pid,'role':role,'status':'background_worker_started'})
