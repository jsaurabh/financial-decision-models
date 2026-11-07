"""Independent three-hour lifecycle; releases GPU even after runner failure."""
import json,subprocess,time
from pathlib import Path
from colab_runtime_health import confirmed_runtime_loss

PROJECT=Path('/workspace/financial-decision-models')
ROOT=Path('/workspace/tracesafe-eval-control');ROOT.mkdir(exist_ok=True)
CLI=['/workspace/.venv/bin/python',str(PROJECT/'scripts/colab_public_cli.py'),'tracesafe']
SESSION='tracesafe-final-eval'

def cli(*args,timeout=150):
    try:
        p=subprocess.run(CLI+list(args),capture_output=True,text=True,timeout=timeout)
        return p.returncode,p.stdout[-1000:]
    except subprocess.TimeoutExpired:return 124,'CLI timeout'

def main():
    import fcntl
    lock=(ROOT/'watcher.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    start=ROOT/'watcher-start.json'
    if not start.exists():start.write_text(json.dumps({'time':time.time()}))
    started=json.loads(start.read_text())['time'];heartbeat=0;misses=0
    while True:
        rc,detail=cli('download','-s',SESSION,'/content/final-status.json',str(ROOT/'worker-status.json'))
        if rc:misses+=1;state={'status':'unreachable','detail':detail}
        else:misses=0;state=json.loads((ROOT/'worker-status.json').read_text())
        if time.time()-heartbeat>120 and state.get('status') not in ['complete','failed']:
            cli('exec','-s',SESSION,'-f',str(PROJECT/'scripts/colab_public_heartbeat.py'),'--timeout','45',timeout=80);heartbeat=time.time()
        expired=time.time()-started>3*3600
        lost=misses>=5 and confirmed_runtime_loss(rc,detail)
        terminal=state.get('status') in ['complete','failed'] or expired or lost
        if expired:state.update(status='timeout',error='Three-hour GPU deadline')
        if lost:state.update(status='failed',error='Confirmed runtime loss after five checks')
        if terminal:
            cli('download','-s',SESSION,'/content/final-worker.log',str(ROOT/'worker.log'))
            stop_rc,stop_detail=cli('stop','-s',SESSION);state.update(stop_returncode=stop_rc,stop_detail=stop_detail)
            if stop_rc==0:
                for name in ['tracesafe-colab-adc.json','tracesafe-hf-token']:Path('/workspace/.credentials',name).unlink(missing_ok=True)
        state.update(time=time.time(),elapsed_s=time.time()-started)
        temp=ROOT/'status.tmp';temp.write_text(json.dumps(state,indent=2));temp.replace(ROOT/'status.json')
        if terminal and state.get('stop_returncode')==0:break
        time.sleep(40)

if __name__=='__main__':main()
