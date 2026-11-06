"""Bundle completed reports and runtime provenance; excludes weights and secrets."""
import json
import hashlib
import subprocess
import tarfile
from pathlib import Path
root=Path('/content/financial-decision-models')
meta=root/'results/phase2/runtime';meta.mkdir(parents=True,exist_ok=True)
(meta/'source-sha256.json').write_text(json.dumps({str(f.relative_to(root)):hashlib.sha256(f.read_bytes()).hexdigest()
 for folder in ['finance_decisions','references/qwen27b'] for f in (root/folder).rglob('*.py')},indent=2)+'\n')
for name,python in [('gemma','/usr/bin/python3'),('qwen','/content/qwen-env/bin/python')]:
 p=subprocess.run([python,'-m','pip','freeze'],capture_output=True,text=True)
 (meta/f'{name}-packages.txt').write_text(p.stdout)
for name in ['phase2-causal','phase2-joint']:
 for file in ['run_config.json','selection.json','training.jsonl']:
  src=root/'runs'/name/file
  if src.exists():
   dest=root/'results/phase2'/name.replace('phase2-','')/file
   dest.parent.mkdir(exist_ok=True,parents=True);dest.write_bytes(src.read_bytes())
with tarfile.open('/content/phase2-reports.tar.gz','w:gz') as tar:
 tar.add(root/'results/phase2',arcname='results/phase2')
 for name in ['phase2-status.json','qwen-prepare-status.json']:
  tar.add(root/name,arcname='results/phase2/runtime/'+name)
print({'status':'reports_bundled','bytes':Path('/content/phase2-reports.tar.gz').stat().st_size})
