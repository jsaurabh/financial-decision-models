"""Keep owned kernels active, recover bounded preemptions, release terminal GPUs."""
import json,os,subprocess,time
from pathlib import Path

ROOT=Path('/workspace/public-eval-control');ROOT.mkdir(exist_ok=True)
PROJECT=Path('/workspace/financial-decision-models')
CLI=['/workspace/.venv/bin/python',str(PROJECT/'scripts/colab_public_cli.py')]
START=ROOT/'watcher-start.json'
if not START.exists():START.write_text(json.dumps({'time':time.time()}))
started=json.loads(START.read_text())['time'];done={};misses={r:0 for r in ['gemma','qwen']};last_heartbeat={r:0 for r in misses}
RETRIES=ROOT/'retries.json';retries=json.loads(RETRIES.read_text()) if RETRIES.exists() else {'gemma':0,'qwen':0}
# One additional Qwen recovery after a confirmed checkpoint DNS failure and a
# heartbeat error that was incorrectly treated as proof of VM loss.
MAX_RECOVERIES={'gemma':2,'qwen':3}

def cli(role,*args,timeout=120):
    try:
        p=subprocess.run(CLI+[role,*args],capture_output=True,text=True,timeout=timeout)
        return p.returncode,p.stdout
    except subprocess.TimeoutExpired:return 124,'CLI timeout'


def save(states):
    obj={'time':time.time(),'elapsed_s':time.time()-started,'workers':states,'recoveries':retries}
    path=ROOT/'status.json';temp=ROOT/'status.tmp';temp.write_text(json.dumps(obj,indent=2));temp.replace(path)
    with (ROOT/'controller.jsonl').open('a') as f:f.write(json.dumps(obj)+'\n')


def recover(role):
    retries[role]+=1;RETRIES.write_text(json.dumps(retries));print(json.dumps({'recovering':role,'attempt':retries[role]}),flush=True)
    cli(role,'download','-s',f'public-eval-{role}','/content/public-worker.log',str(ROOT/f'{role}-attempt-{retries[role]}.log'))
    cli(role,'stop','-s',f'public-eval-{role}')
    rc,msg=cli(role,'new','-s',f'public-eval-{role}','--gpu','G4')
    if rc:return False,msg
    secret=Path('/workspace/.credentials/public-eval-hf-token')
    if not secret.exists():secret.write_text(os.environ['HF_TOKEN']);secret.chmod(0o600)
    requirements=PROJECT/'requirements-gemma-eval.txt' if role=='gemma' else Path('/workspace/public-eval-source/qwen-requirements.txt')
    uploads=[(Path(f'/workspace/public-eval-source/{role}-role.txt'),'/content/public-role.txt'),
             (requirements,'/content/public-requirements.txt'),(secret,'/content/.public-hf-token'),
             (Path('/workspace/public-eval-source/public-rows.jsonl.gz'),'/content/public-rows.jsonl.gz')]
    for src,dst in uploads:
        rc,msg=cli(role,'upload','-s',f'public-eval-{role}',str(src),dst,timeout=240)
        if rc:return False,msg
    rc,msg=cli(role,'exec','-s',f'public-eval-{role}','-f',str(PROJECT/'scripts/colab_public_warmup.py'),'--timeout','60')
    return rc==0,msg


for _ in range(1500):
    states={}
    for role in ['gemma','qwen']:
        if role in done:states[role]=done[role];continue
        rc,msg=cli(role,'download','-s',f'public-eval-{role}','/content/public-status.json',str(ROOT/f'{role}-status.json'))
        if rc==0:
            state=json.loads((ROOT/f'{role}-status.json').read_text());misses[role]=0
        else:
            misses[role]+=1;state={'status':'unreachable','detail':msg[-200:]}
        if time.time()-last_heartbeat[role]>120:
            hb,_=cli(role,'exec','-s',f'public-eval-{role}','-f',str(PROJECT/'scripts/colab_public_heartbeat.py'),'--timeout','45',timeout=80)
            last_heartbeat[role]=time.time()
            if hb:state['heartbeat_error']=True
        expired=time.time()-started>8*3600
        recoverable=state.get('status')=='failed' or misses[role]>=3
        if recoverable and not expired and retries[role]<MAX_RECOVERIES[role]:
            states[role]={'status':'recovering','attempt':retries[role]+1,'time':time.time()};save(states|done)
            ok,detail=recover(role);misses[role]=0;last_heartbeat[role]=time.time()
            state={'status':'restarted' if ok else 'failed','detail':detail[-200:],'role':role,'time':time.time()}
            if ok:states[role]=state;continue
        terminal=state.get('status') in ['complete','failed']
        if expired and not terminal:state.update(status='timeout',error='Eight-hour GPU evaluation deadline');terminal=True
        if misses[role]>=3 and retries[role]>=MAX_RECOVERIES[role]:state.update(status='failed',error='Runtime unavailable after bounded recovery');terminal=True
        if terminal:
            cli(role,'download','-s',f'public-eval-{role}','/content/public-worker.log',str(ROOT/f'{role}-worker.log'))
            rc,msg=cli(role,'stop','-s',f'public-eval-{role}')
            # A pruned local session is already released; verify via server listing.
            if rc and ('not found' in msg.lower() or 'no session' in msg.lower()):rc=0
            state['stop_returncode']=rc
            if rc==0:done[role]=state
        states[role]=state
    save(states)
    if len(done)==2:
        for name in ['public-eval-colab-adc.json','public-eval-hf-token']:
            p=Path('/workspace/.credentials')/name
            if p.exists():p.unlink()
        break
    time.sleep(40)
