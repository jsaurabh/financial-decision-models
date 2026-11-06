"""Collect the completed Qwen retry without overwriting Gemma runtime records."""
import hashlib,json,subprocess,tarfile
from pathlib import Path
root=Path('/content/financial-decision-models');meta=root/'results/phase2/runtime';meta.mkdir(parents=True,exist_ok=True)
python='/content/qwen-env/bin/python'
(meta/'qwen-packages.txt').write_text(subprocess.check_output([python,'-m','pip','freeze'],text=True))
code="import json,torch; print(json.dumps({'torch':torch.__version__,'cuda':torch.version.cuda,'matmul_allow_tf32':torch.backends.cuda.matmul.allow_tf32}))"
(meta/'qwen-inference-defaults.json').write_text(subprocess.check_output([python,'-c',code],text=True))
(meta/'qwen-source-sha256.json').write_text(json.dumps({str(f.relative_to(root)):hashlib.sha256(f.read_bytes()).hexdigest() for folder in ['finance_decisions','references/qwen27b'] for f in (root/folder).rglob('*.py')},indent=2)+'\n')
for src,dest in [('hardware.json','qwen-hardware.json'),('qwen-retry-status.json','qwen-retry-status.json')]:
 (meta/dest).write_bytes((root/src).read_bytes())
with tarfile.open('/content/qwen-phase2-reports.tar.gz','w:gz') as t:
 t.add(root/'results/phase2/qwen',arcname='results/phase2/qwen')
 t.add(meta,arcname='results/phase2/runtime')
print({'status':'qwen_reports_bundled','bytes':Path('/content/qwen-phase2-reports.tar.gz').stat().st_size})
