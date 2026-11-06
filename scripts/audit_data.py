"""Local corpus audit: targets, split hashes, exact state overlap, and families."""
import hashlib
import json
from pathlib import Path

from finance_decisions.schema import options, render

manifest = json.loads(Path('data/manifest.json').read_text())
seen_ids, seen_families, seen_states = set(), set(), set()
result = {}
for split in ['test', 'calibration', 'train']:
    path = Path('data/private') / f'{split}.jsonl'
    assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest['splits'][split]['sha256']
    rows = [json.loads(x) for x in path.read_text().splitlines()]
    ids = {r['id'] for r in rows}; families = {r['family'] for r in rows}
    states = {hashlib.sha256(' '.join(r['state'].lower().split()).encode()).hexdigest() for r in rows}
    assert len(ids) == len(rows)
    assert not ids & seen_ids
    assert not families & seen_families
    assert not states & seen_states
    for r in rows:
        assert r['target'] in options(r['question'])[0]
        render(r['state'], r['question'])
    seen_ids |= ids; seen_families |= families; seen_states |= states
    result[split] = {'rows': len(rows), 'families': len(families), 'unique_normalized_states': len(states)}
gap = [json.loads(x) for x in Path('data/private/contrastive-banking-train.jsonl').read_text().splitlines()]
legacy = [json.loads(x) for x in Path('data/private/legacy-banking-128.jsonl').read_text().splitlines()]
assert not {r['family'] for r in gap} & {r['family'] for r in legacy}
result['contrastive_vs_legacy_family_overlap'] = 0
result['status'] = 'passed'
Path('data/audit.json').write_text(json.dumps(result, indent=2)); print(json.dumps(result, indent=2))
