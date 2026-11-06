"""Owned Colab worker watcher: persist heartbeat and release finished workers."""
import json,subprocess,time,os
from pathlib import Path
root=Path('/workspace/public-eval-control');root.mkdir(exist_ok=True)
start=time.time();done={};history=root/'controller.jsonl'
for loop in range(1000):
 states={}
 for role in ['gemma','qwen']:
  if role in done:states[role]=done[role];continue
  path=root/f'{role}-status.json'
  cmd=['/workspace/.venv/bin/python','/workspace/financial-decision-models/scripts/colab_public_cli.py',role,'download','-s',f'public-eval-{role}','/content/public-status.json',str(path)]
  try:
   result=subprocess.run(cmd,capture_output=True,text=True,timeout=90)
   state=json.loads(path.read_text()) if result.returncode==0 else {'status':'unreachable','detail':result.stdout[-250:]}
  except Exception as exc:state={'status':'unreachable','error_type':type(exc).__name__}
  terminal=state.get('status') in ['complete','failed']
  expired=time.time()-start>8*3600
  if expired and not terminal:state['status']='timeout';state['error']='Eight-hour GPU evaluation deadline'
  if terminal or expired:
   subprocess.run(['/workspace/.venv/bin/python','/workspace/financial-decision-models/scripts/colab_public_cli.py',role,'download','-s',f'public-eval-{role}','/content/public-worker.log',str(root/f'{role}-worker.log')],capture_output=True,text=True,timeout=90)
   stop=subprocess.run(['/workspace/.venv/bin/python','/workspace/financial-decision-models/scripts/colab_public_cli.py',role,'stop','-s',f'public-eval-{role}'],capture_output=True,text=True,timeout=90)
   state['stop_returncode']=stop.returncode
   if stop.returncode==0:done[role]=state
  states[role]=state
 snapshot={'time':time.time(),'elapsed_s':time.time()-start,'workers':states}
 (root/'status.json').write_text(json.dumps(snapshot,indent=2))
 with history.open('a') as f:f.write(json.dumps(snapshot)+'\n')
 if len(done)==2:
  for name in ['public-eval-colab-adc.json','public-eval-hf-token']:
   path=Path('/workspace/.credentials')/name
   if path.exists():path.unlink()
  break
 time.sleep(40)
