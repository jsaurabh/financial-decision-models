"""Eight-hour owned G4 lifecycle; service outages do not prove runtime loss."""
import argparse,json,subprocess,time
from pathlib import Path
from colab_runtime_health import confirmed_runtime_loss

PROJECT=Path('/workspace/financial-decision-models');ROOT=Path('/workspace/phase4-control')

def main():
 import fcntl
 parser=argparse.ArgumentParser();parser.add_argument('--family',choices=['gemma','qwen'],required=True);args=parser.parse_args();family=args.family
 root=ROOT/family;root.mkdir(exist_ok=True);lock=(root/'watcher.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 cli=['/workspace/.venv/bin/python',str(PROJECT/'scripts/colab_public_cli.py'),'phase4-'+family];session='phase4-'+family
 def call(*arguments,timeout=180):
  try:
   p=subprocess.run(cli+list(arguments),capture_output=True,text=True,timeout=timeout);return p.returncode,p.stdout[-1500:]
  except subprocess.TimeoutExpired:return 124,'CLI timeout'
 start=root/'start.json'
 if not start.exists():start.write_text(json.dumps({'time':time.time()}))
 started=json.loads(start.read_text())['time'];misses=0;heartbeat=0
 retry=root/'recoveries.json';recoveries=json.loads(retry.read_text())['count'] if retry.exists() else 0
 def recover():
  rc,detail=call('stop','-s',session)
  if rc and not confirmed_runtime_loss(rc,detail):return False
  rc,_=call('new','-s',session,'--gpu','G4',timeout=240)
  if rc:return False
  uploads=[(root/'role.txt','/content/phase4-role.txt'),(Path('/workspace/.credentials')/('phase4-'+family+'-hf-token'),'/content/.phase4-hf-token')]
  uploads += [(ROOT/name,'/content/'+name) for name in ['phase4-code.tar.gz','phase4-training.tar.gz','phase4-tests.tar.gz','phase4-ready.json']]
  for source,target in uploads:
   rc,_=call('upload','-s',session,str(source),target,timeout=300)
   if rc:return False
  rc,_=call('exec','-s',session,'-f',str(PROJECT/'scripts/colab_phase4_bootstrap.py'),'--timeout','60',timeout=100)
  return rc==0
 while True:
  rc,detail=call('download','-s',session,'/content/phase4-status.json',str(root/'worker-status.json'))
  if rc:misses+=1;state={'status':'unreachable','detail':detail}
  else:misses=0;state=json.loads((root/'worker-status.json').read_text())
  if (root/'staging-failure.json').exists():state=json.loads((root/'staging-failure.json').read_text())
  elapsed=time.time()-started;expired=elapsed>8*3600;lost=misses>=3 and confirmed_runtime_loss(rc,detail)
  if time.time()-heartbeat>120 and state.get('status') not in ['pipeline_complete','failed']:
   call('exec','-s',session,'-f',str(PROJECT/'scripts/colab_public_heartbeat.py'),'--timeout','45',timeout=80);heartbeat=time.time()
  if lost and not expired and recoveries<1:
   recoveries+=1;retry.write_text(json.dumps({'count':recoveries}))
   if recover():misses=0;continue
   state={'status':'failed','error':'Confirmed runtime recovery failed'}
  terminal=state.get('status') in ['pipeline_complete','failed'] or expired or lost
  if expired:state.update(status='timeout',error='Eight-hour GPU deadline')
  if terminal:
   call('download','-s',session,'/content/phase4-worker.log',str(root/'worker.log'))
   stop_rc,stop_detail=call('stop','-s',session)
   if confirmed_runtime_loss(stop_rc,stop_detail):stop_rc=0
   state.update(stop_returncode=stop_rc,stop_detail=stop_detail)
   if stop_rc==0:
    for name in ['phase4-'+family+'-colab-adc.json','phase4-'+family+'-hf-token']:Path('/workspace/.credentials',name).unlink(missing_ok=True)
  state.update(time=time.time(),elapsed_s=elapsed,recoveries=recoveries);tmp=root/'status.tmp';tmp.write_text(json.dumps(state,indent=2));tmp.replace(root/'status.json')
  if terminal and state.get('stop_returncode')==0:break
  time.sleep(45)

if __name__=='__main__':main()
