"""Isolated training lifecycle: bounded recovery, release and aggregate capture."""
import json
import os
import subprocess
import time
from pathlib import Path

PROJECT=Path('/workspace/financial-decision-models');ROOT=Path('/workspace/phase3-control');ROOT.mkdir(exist_ok=True)
CLI=['/workspace/.venv/bin/python',str(PROJECT/'scripts/colab_public_cli.py'),'phase3']
SESSION='phase3-gemma-train'
start_file=ROOT/'started.json'
if not start_file.exists():start_file.write_text(json.dumps({'time':time.time()}))
started=json.loads(start_file.read_text())['time'];misses=0;heartbeat=0
retry_file=ROOT/'recoveries.json';retries=json.loads(retry_file.read_text())['count'] if retry_file.exists() else 0


def cli(*args,timeout=180):
    try:
        p=subprocess.run(CLI+list(args),capture_output=True,text=True,timeout=timeout)
        return p.returncode,p.stdout
    except subprocess.TimeoutExpired:return 124,'CLI timeout'


def save(state):
    state.update(controller_time=time.time(),elapsed_s=time.time()-started,recoveries=retries)
    temp=ROOT/'status.tmp';temp.write_text(json.dumps(state,indent=2));temp.replace(ROOT/'status.json')


def recover():
    global retries
    retries+=1;retry_file.write_text(json.dumps({'count':retries}))
    cli('download','-s',SESSION,'/content/phase3-worker.log',str(ROOT/f'attempt-{retries}.log'))
    cli('stop','-s',SESSION)
    rc,msg=cli('new','-s',SESSION,'--gpu','G4')
    if rc:return False,msg
    uploads=[(PROJECT/'requirements-gemma-eval.txt','/content/phase3-requirements.txt'),
             (Path('/workspace/.credentials/phase3-hf-token'),'/content/.phase3-hf-token')]
    for src,dst in uploads:
        rc,msg=cli('upload','-s',SESSION,str(src),dst)
        if rc:return False,msg
    rc,msg=cli('exec','-s',SESSION,'-f',str(PROJECT/'scripts/colab_phase3_bootstrap.py'),'--timeout','60')
    if rc:return False,msg
    for name in ['phase3-code.tar.gz','phase3-data.tar.gz','phase3-ready.json']:
        rc,msg=cli('upload','-s',SESSION,str(ROOT/name),'/content/'+name,timeout=300)
        if rc:return False,msg
    return True,'Resuming identical sealed job from private model checkpoint'


while True:
    rc,msg=cli('download','-s',SESSION,'/content/phase3-status.json',str(ROOT/'worker-status.json'))
    if rc==0:
        state=json.loads((ROOT/'worker-status.json').read_text());misses=0
    else:
        misses+=1;state={'status':'unreachable','detail':msg[-200:]}
    if time.time()-heartbeat>120:
        rc,_=cli('exec','-s',SESSION,'-f',str(PROJECT/'scripts/colab_phase3_heartbeat.py'),'--timeout','45',timeout=80)
        heartbeat=time.time()
        if rc:state['heartbeat_error']=True
    expired=time.time()-started>4*3600
    if (state['status']=='failed' or misses>=3) and not expired and retries<2 and (ROOT/'phase3-ready.json').exists():
        save({'status':'recovering','attempt':retries+1})
        ok,detail=recover();misses=0;state={'status':'restarted' if ok else 'failed','detail':detail};save(state)
        if ok:time.sleep(30);continue
    terminal=state['status'] in ['complete','failed']
    if expired:state.update(status='timeout',error='Four-hour pilot deadline');terminal=True
    if misses>=3 and retries>=2:state.update(status='failed',error='Runtime unavailable after bounded recovery');terminal=True
    if terminal:
        cli('download','-s',SESSION,'/content/phase3-worker.log',str(ROOT/'worker.log'))
        rc,msg=cli('stop','-s',SESSION)
        if rc and ('not found' in msg.lower() or 'no session' in msg.lower()):rc=0
        state['stop_returncode']=rc;save(state)
        if rc==0:
            try:
                from huggingface_hub import HfApi,hf_hub_download
                repo='jsaurabh/gemma4-finance-phase3-pilot';api=HfApi();info=api.model_info(repo);assert info.private
                output=PROJECT/'results/phase3-pilot';output.mkdir(exist_ok=True)
                for filename in ['summary.json','initial-development.json','run_config.json','training.jsonl']:
                    try:
                        p=hf_hub_download(repo,'run/'+filename,revision=info.sha)
                        (output/filename).write_bytes(Path(p).read_bytes())
                    except Exception:pass
                (output/'runtime.json').write_text(json.dumps(state,indent=2))
                (output/'artifacts.json').write_text(json.dumps({'repo':repo,'revision':info.sha,'private':True},indent=2))
                (output/'README.md').write_text('# Phase-3 training pilot\n\nTraining-only pilot; no final benchmark evaluation was launched. See summary/runtime and the pinned private model artifact.\n')
                paths=[str(p.relative_to(PROJECT)) for p in output.iterdir() if p.is_file()]
                staged=set(subprocess.check_output(['git','diff','--cached','--name-only'],cwd=PROJECT,text=True).splitlines())
                branch=subprocess.check_output(['git','branch','--show-current'],cwd=PROJECT,text=True).strip()
                secrets=[os.environ[k].encode() for k in ['HF_TOKEN','GH_TOKEN','TYPESAFE_API_KEY','COLAB_ADC_JSON'] if os.environ.get(k)]
                assert not any(secret in (PROJECT/p).read_bytes() for p in paths for secret in secrets)
                if staged<=set(paths) and branch=='gemma4-finance':
                    subprocess.run(['git','add',*paths],cwd=PROJECT,check=True)
                    if subprocess.run(['git','diff','--cached','--quiet'],cwd=PROJECT).returncode:
                        subprocess.run(['git','commit','-m','Record phase-3 supervised pilot training outcome'],cwd=PROJECT,check=True)
                        subprocess.run(['git','push','origin','HEAD:gemma4-finance'],cwd=PROJECT,check=True)
            except Exception as exc:print(json.dumps({'report_error_type':type(exc).__name__}),flush=True)
            for filename in ['phase3-colab-adc.json','phase3-hf-token']:
                path=Path('/workspace/.credentials')/filename
                if path.exists():path.unlink()
            break
    save(state);time.sleep(40)
