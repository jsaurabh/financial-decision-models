"""Restart a failed preflight before any optimizer step; reuse the existing G4 VM."""
import os
import subprocess
from pathlib import Path
root = Path('/content/financial-decision-models')
# Colab's preinstalled torchao may be too old for current PEFT. It is unused here.
subprocess.run([os.sys.executable, '-m', 'pip', 'uninstall', '-y', 'torchao'], check=True)
# _driver is a child of the kernel; the secret is inherited from the original bootstrap.
# Recover HF_TOKEN from the kernel's existing environment only, never from process inspection.
environment = dict(os.environ)
token = Path('/content/.finance-hf-token')
if token.exists():
    environment['HF_TOKEN'] = token.read_text().strip(); token.unlink()
if not environment.get('HF_TOKEN'):
    raise RuntimeError('Stage the HF token file before restarting')
environment.update(TOKENIZERS_PARALLELISM='false', PYTHONUNBUFFERED='1', OMP_NUM_THREADS='8')
with (root / 'training.log').open('a') as log:
    p = subprocess.Popen([os.sys.executable, str(root / '_driver.py')], cwd=root, env=environment,
                         stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
print({'pid': p.pid, 'status': 'restarted'})
