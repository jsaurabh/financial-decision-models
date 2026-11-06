"""Bounded lifecycle for the isolated final-test GPU; never mutates a live cache."""
import json, os, subprocess, time
from pathlib import Path

PROJECT=Path('/workspace/financial-decision-models')
ROOT=Path('/workspace/final-eval-control'); ROOT.mkdir(exist_ok=True)
CLI=['/workspace/.venv/bin/python',str(PROJECT/'scripts/colab_public_cli.py'),'final']
SESSION='phase3-final-eval'

def cli(*args,timeout=180):
    try:
        p=subprocess.run(CLI+list(args),capture_output=True,text=True,timeout=timeout)
        return p.returncode,p.stdout[-1000:]
    except subprocess.TimeoutExpired:return 124,'CLI timeout'

def save(value):
    tmp=ROOT/'status.tmp';tmp.write_text(json.dumps(value,indent=2));tmp.replace(ROOT/'status.json')

def recover():
    rc,_=cli('stop','-s',SESSION)
    if rc:return False
    rc,_=cli('new','-s',SESSION,'--gpu','G4')
    if rc:return False
    uploads=[(PROJECT/'requirements-gemma-eval.txt','/content/final-requirements.txt'),
             (Path('/workspace/.credentials/final-hf-token'),'/content/.final-hf-token')]
    for name in ['final-code.tar.gz','final-data.tar.gz','final-ready.json']:
        if (ROOT/name).exists():uploads.append((ROOT/name,'/content/'+name))
    for src,dst in uploads:
        rc,_=cli('upload','-s',SESSION,str(src),dst,timeout=300)
        if rc:return False
    rc,_=cli('exec','-s',SESSION,'-f',str(PROJECT/'scripts/colab_final_bootstrap.py'),'--timeout','60')
    return rc==0

def main():
    import fcntl
    lock=(ROOT/'watcher.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    start=ROOT/'watcher-start.json'
    if not start.exists():start.write_text(json.dumps({'time':time.time()}))
    started=json.loads(start.read_text())['time'];misses=0;heartbeat=0;recoveries=0
    retry=ROOT/'recoveries.json'
    if retry.exists():recoveries=json.loads(retry.read_text())['count']
    while True:
        rc,detail=cli('download','-s',SESSION,'/content/final-status.json',str(ROOT/'worker-status.json'))
        if rc:misses+=1;state={'status':'unreachable','detail':detail}
        else:misses=0;state=json.loads((ROOT/'worker-status.json').read_text())
        elapsed=time.time()-started;expired=elapsed>6*3600
        # A heartbeat error alone is not evidence that a healthy worker was lost.
        if time.time()-heartbeat>120 and state.get('status') not in ['complete','failed']:
            cli('exec','-s',SESSION,'-f',str(PROJECT/'scripts/colab_public_heartbeat.py'),'--timeout','45',timeout=80)
            heartbeat=time.time()
        if misses>=3 and not expired and recoveries<2:
            recoveries+=1;retry.write_text(json.dumps({'count':recoveries}))
            save({'status':'recovering','recoveries':recoveries,'time':time.time(),'elapsed_s':elapsed})
            if recover():misses=0;continue
            state={'status':'failed','error':'VM recovery failed'}
        terminal=state.get('status') in ['complete','failed'] or expired or misses>=3
        if expired:state.update(status='timeout',error='Six-hour GPU deadline')
        if terminal:
            cli('download','-s',SESSION,'/content/final-worker.log',str(ROOT/'worker.log'))
            stop_rc,stop_detail=cli('stop','-s',SESSION)
            state.update(stop_returncode=stop_rc,stop_detail=stop_detail)
            if stop_rc==0:
                for name in ['final-colab-adc.json','final-hf-token']:
                    Path('/workspace/.credentials',name).unlink(missing_ok=True)
        save(state|{'recoveries':recoveries,'elapsed_s':elapsed,'time':time.time()})
        if terminal and state.get('stop_returncode')==0:break
        time.sleep(40)

if __name__=='__main__':main()
