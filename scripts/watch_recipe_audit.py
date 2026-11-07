"""Four-hour lifecycle; recover only a confirmed deleted runtime, once."""
import json
import subprocess
import time
from pathlib import Path

from colab_runtime_health import confirmed_runtime_loss

PROJECT=Path('/workspace/financial-decision-models');ROOT=Path('/workspace/recipe-audit-control')
CLI=['/workspace/.venv/bin/python',str(PROJECT/'scripts/colab_public_cli.py'),'gemma-recipe']
SESSION='gemma-recipe'


def call(*args,timeout=180):
    try:
        p=subprocess.run(CLI+list(args),capture_output=True,text=True,timeout=timeout)
        return p.returncode,p.stdout[-1500:]
    except subprocess.TimeoutExpired:return 124,'CLI timeout'


def stage():
    for source,target in [(Path('/workspace/.credentials/gemma-recipe-hf-token'),'/content/.recipe-hf-token')]+[
        (ROOT/name,'/content/'+name) for name in ['recipe-code.tar.gz','recipe-data.tar.gz','recipe-ready.json']]:
        rc,detail=call('upload','-s',SESSION,str(source),target,timeout=300)
        print(json.dumps({'staging':source.name,'returncode':rc}),flush=True)
        if rc:raise RuntimeError('Recipe staging failed: '+detail)
    rc,detail=call('exec','-s',SESSION,'-f',str(PROJECT/'scripts/colab_recipe_bootstrap.py'),
                   '--timeout','60',timeout=100)
    if rc:raise RuntimeError('Recipe bootstrap failed: '+detail)
    print(detail,flush=True)


def main():
    import fcntl
    ROOT.mkdir(exist_ok=True);lock=(ROOT/'watcher.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    start=ROOT/'start.json'
    if not start.exists():
        rc,detail=call('new','-s',SESSION,'--gpu','G4',timeout=240)
        print(json.dumps({'allocation_returncode':rc,'detail':detail}),flush=True)
        if rc:raise RuntimeError('Colab G4 allocation failed')
        start.write_text(json.dumps({'time':time.time()}))
        try:stage()
        except Exception as exc:
            (ROOT/'staging-failure.json').write_text(json.dumps({'status':'failed','error':str(exc)[:300]}))
    started=json.loads(start.read_text())['time'];recoveries=0;misses=0;heartbeat=0
    recovered=ROOT/'recoveries.json'
    if recovered.exists():recoveries=json.loads(recovered.read_text())['count']
    while True:
        rc,detail=call('download','-s',SESSION,'/content/recipe-status.json',str(ROOT/'worker-status.json'))
        if rc:misses+=1;state={'status':'unreachable','detail':detail}
        else:misses=0;state=json.loads((ROOT/'worker-status.json').read_text())
        if (ROOT/'staging-failure.json').exists():state=json.loads((ROOT/'staging-failure.json').read_text())
        elapsed=time.time()-started;expired=elapsed>4*3600
        lost=misses>=3 and confirmed_runtime_loss(rc,detail)
        if lost and not expired and recoveries<1:
            recoveries+=1;recovered.write_text(json.dumps({'count':recoveries}))
            stop_rc,stop_detail=call('stop','-s',SESSION)
            if stop_rc==0 or confirmed_runtime_loss(stop_rc,stop_detail):
                new_rc,_=call('new','-s',SESSION,'--gpu','G4',timeout=240)
                if new_rc==0:
                    try:stage();misses=0;continue
                    except Exception:pass
            state={'status':'failed','error':'Confirmed runtime recovery failed'}
        if time.time()-heartbeat>120 and state.get('status') not in ['pipeline_complete','failed']:
            call('exec','-s',SESSION,'-f',str(PROJECT/'scripts/colab_recipe_heartbeat.py'),
                  '--timeout','45',timeout=80);heartbeat=time.time()
        terminal=state.get('status') in ['pipeline_complete','failed'] or expired or lost
        if expired:state.update(status='timeout',error='Four-hour GPU budget reached')
        if terminal:
            call('download','-s',SESSION,'/content/recipe-worker.log',str(ROOT/'worker.log'))
            stop_rc,stop_detail=call('stop','-s',SESSION)
            if confirmed_runtime_loss(stop_rc,stop_detail):stop_rc=0
            state.update(stop_returncode=stop_rc,stop_detail=stop_detail)
            if stop_rc==0:
                for name in ['gemma-recipe-colab-adc.json','gemma-recipe-hf-token']:
                    Path('/workspace/.credentials',name).unlink(missing_ok=True)
        state.update(time=time.time(),elapsed_s=elapsed,recoveries=recoveries)
        tmp=ROOT/'status.tmp';tmp.write_text(json.dumps(state,indent=2));tmp.replace(ROOT/'status.json')
        if terminal and state.get('stop_returncode')==0:break
        time.sleep(45)


if __name__=='__main__':main()
