"""Stage the earlier frozen banking test and verify exact historical inputs."""
import hashlib
import json
import subprocess
from pathlib import Path

source = Path('/workspace/qwen-rlcd-jev-repro')
old = json.loads((source / 'results/colab/autojev-27b-v4/three-way-comparison.json').read_text())
archive = {r['id']: r for r in [json.loads(x) for x in (source / 'data/jev_gap_curriculum_v1/eval.jsonl').read_text().splitlines()]}
rows = []
for comparison in old['comparisons']:
    raw = archive[comparison['id']]
    question = raw['input']['questions']['decision']
    state = raw['input']['state']
    digest = hashlib.sha256(json.dumps({'state': state, 'question': question}, sort_keys=True,
        separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()
    assert digest == comparison['jev']['input_sha256']
    rows.append({'id': raw['id'], 'family': raw['family'], 'domain': raw['family'].split('-')[1],
                 'state': state, 'question': question, 'target': raw['reference']['target'],
                 'historical': {k: comparison[k] for k in ['jev', 'our_banking_finetune', 'public_autojev']}})
out = Path('data/private/legacy-banking-128.jsonl')
out.write_text(''.join(json.dumps(r) + '\n' for r in rows))
manifest = {'source_repo': 'https://github.com/jsaurabh/qwen-rlcd-jev-repro',
            'source_revision': subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip(),
            'rows': len(rows), 'historical_input_hashes_match': True,
            'sha256': hashlib.sha256(out.read_bytes()).hexdigest(),
            'use': 'evaluation only; not used for this Gemma training or calibration',
            'historical_results': {k: old[k] for k in ['jev', 'our_banking_finetune', 'public_autojev']},
            'limits': 'Narrow synthetic rules; historical provider scores are not fresh live calls. No latency comparison is made.'}
Path('data/legacy-banking-manifest.json').write_text(json.dumps(manifest, indent=2))
print(json.dumps(manifest, indent=2))

training_path = source / 'data/jev_gap_curriculum_v1/train.jsonl'
training = []
for line in training_path.read_text().splitlines():
    raw = json.loads(line)
    training.append({'id': raw['id'], 'family': raw['family'], 'domain': 'banking_' + raw['family'].split('-')[1],
                     'state': raw['input']['state'], 'question': raw['input']['questions']['decision'],
                     'target': raw['reference']['target']})
assert not {r['family'] for r in training} & {r['family'] for r in rows}
Path('data/private/contrastive-banking-train.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in training))
Path('data/contrastive-banking-manifest.json').write_text(json.dumps({
    'source_repo': manifest['source_repo'], 'source_revision': manifest['source_revision'],
    'source_file': 'data/jev_gap_curriculum_v1/train.jsonl',
    'source_sha256': hashlib.sha256(training_path.read_bytes()).hexdigest(),
    'rows': len(training), 'split': 'upstream train only; no upstream eval families',
    'license': 'MIT', 'labels': 'deterministic rules; not human reviewed'}, indent=2))
