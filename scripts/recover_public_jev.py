"""Recover matching Jev checkpoint records; re-request only missing decisions.

The collector now uses a separate cache. Never select recovered predictions by
accuracy: the earliest timestamp wins if a case exists in multiple snapshots.
"""
import hashlib
import json
import subprocess
from pathlib import Path

from huggingface_hub.constants import HF_HUB_CACHE

from finance_decisions.public_eval import checkpoint, data, digest, labels, read

root = Path('/workspace/public-eval-control')
out = Path('results/public-eval/jev')
backup = root/'jev-output-before-recovery.jsonl'
if not backup.exists():
    backup.write_bytes((out/'predictions.jsonl').read_bytes())
config = json.loads((out/'config.json').read_text())
rows, manifest = data()
assert config['manifest_sha256'] == manifest['rows_sha256']
expected = {r['id']: (digest(r), labels(r), r['target']) for r in rows}
snapshots = Path(HF_HUB_CACHE)/'datasets--jsaurabh--financial-decision-public-eval/snapshots'
paths = [backup, *sorted(snapshots.glob('*/runs/jev/predictions.jsonl'))]
records, blobs, receipts = {}, set(), []
for path in paths:
    if path != backup:
        old_config = path.parent/'config.json'
        if not old_config.exists() or json.loads(old_config.read_text()) != config:
            continue
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    if sha in blobs:
        continue
    blobs.add(sha)
    for record in read(path):
        key = record['id']
        assert key in expected and (record['input_sha256'], record['labels'], record['target']) == expected[key]
        if key not in records or record.get('timestamp_utc', '~') < records[key].get('timestamp_utc', '~'):
            records[key] = record
    receipts.append({'sha256': sha, 'unique_recovered': len(records)})
recovered = [records[r['id']] for r in rows if r['id'] in records]
temp = out/'recovered.tmp'
temp.write_text(''.join(json.dumps(r, allow_nan=False)+'\n' for r in recovered))
temp.replace(out/'predictions.jsonl')
receipt = {'recovered': len(recovered), 'missing': len(rows)-len(recovered), 'sources': receipts,
           'selection': 'Earliest API timestamp, never prediction correctness; canonical input/label/target validated.'}
(root/'jev-recovery.json').write_text(json.dumps(receipt, indent=2))
checkpoint(out, recovered, rows, config)
print(json.dumps({'recovered': len(recovered), 'missing': len(rows)-len(recovered)}), flush=True)
if len(recovered) != len(rows):
    subprocess.run(['/workspace/.venv/bin/python', '-m', 'finance_decisions.public_eval', '--role', 'jev'], check=True)
print(json.dumps({'status': 'complete'}), flush=True)
