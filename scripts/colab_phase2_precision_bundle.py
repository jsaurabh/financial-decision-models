"""Preserve post-test diagnostic and exact runtime source."""
import hashlib,json,subprocess,tarfile,sys
from pathlib import Path
root=Path('/content/financial-decision-models');meta=root/'results/phase2/runtime';meta.mkdir(parents=True,exist_ok=True)
(meta/'precision-status.json').write_bytes((root/'precision-status.json').read_bytes())
(meta/'precision-packages.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
(meta/'precision-source-sha256.json').write_text(json.dumps({str(f.relative_to(root)):hashlib.sha256(f.read_bytes()).hexdigest() for f in (root/'finance_decisions').glob('*.py')},indent=2)+'\n')
with tarfile.open('/content/precision-reports.tar.gz','w:gz') as t:
 t.add(root/'results/phase2/precision-audit',arcname='results/phase2/precision-audit')
 for f in meta.glob('precision-*'):t.add(f,arcname='results/phase2/runtime/'+f.name)
print({'status':'precision_reports_bundled','bytes':Path('/content/precision-reports.tar.gz').stat().st_size})
