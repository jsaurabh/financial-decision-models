"""Bundle completed reports only; exclude raw training data, credentials and weights."""
import tarfile
from pathlib import Path
root=Path('/content/financial-decision-models')
with tarfile.open('/content/finance-reports.tar.gz','w:gz') as archive:
    archive.add(root/'results',arcname='results')
    for run in (root/'runs').glob('gemma4-finance-*'):
        for filename in ['run_config.json','training.jsonl','evaluation.json']:
            path=run/filename
            if path.exists():archive.add(path,arcname=f'results/{run.name}/training-{filename}')
        path=run/'release/decision_config.json'
        if path.exists():archive.add(path,arcname=f'results/{run.name}/decision_config.json')
print({'bundle':'/content/finance-reports.tar.gz','bytes':Path('/content/finance-reports.tar.gz').stat().st_size})
