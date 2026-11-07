from pathlib import Path
import subprocess
root = Path('/content/financial-decision-models')
print(subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.total,memory.used,utilization.gpu', '--format=csv,noheader']).decode())
print((root / 'job-status.json').read_text() if (root / 'job-status.json').exists() else 'starting')
path = root / 'training.log'
if path.exists():
    print('\n'.join(path.read_text(errors='replace').splitlines()[-12:]))
