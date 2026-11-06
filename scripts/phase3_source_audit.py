"""Preliminary training-source inventory; no training or model API calls.

Raw sources stay in private working copies. Exact overlap checks here do not
replace the cross-release, historical-holdout and near-duplicate split audit.
"""
import concurrent.futures
import hashlib
import json
import zipfile
from collections import Counter
from pathlib import Path

from huggingface_hub import hf_hub_download

ROOT = Path('/workspace/public-eval-source')


def read(path):
    with Path(path).open() as f:
        return [json.loads(line) for line in f if line.strip()]


def content_hash(value):
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(' '.join(text.lower().split()).encode()).hexdigest()


def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    paths = json.loads((ROOT/'paths.json').read_text())
    manifest = json.loads(Path('data/public-eval-manifest.json').read_text())
    for key in ['contract-nli.zip', 'rag-responses.jsonl', 'rag-sources.jsonl']:
        assert fingerprint(paths[key]) == manifest['source_hashes'][key]['sha256']
    result = {'status': 'preliminary inventory, NOT a training-ready or leakage-free corpus',
              'excluded_eval_rows_sha256': manifest['rows_sha256'], 'training_started': False,
              'limits': ['Historical-holdout cross-release audit pending', 'Near-duplicate audit pending',
                         'License/usage audit and canonical decision conversion pending'], 'sources': {}}
    with zipfile.ZipFile(paths['contract-nli.zip']) as archive:
        splits = {s: json.loads(archive.read(f'contract-nli/{s}.json')) for s in ['train', 'dev', 'test']}
    test_hashes = {content_hash(d['text']) for d in splits['test']['documents']}
    contract = {}
    for split, data in splits.items():
        documents = data['documents']
        overlap = sum(content_hash(d['text']) in test_hashes for d in documents)
        contract[split] = {'documents': len(documents), 'unique_document_hashes': len({content_hash(d['text']) for d in documents}),
                           'decisions': len(documents)*len(data['labels']), 'documents_matching_test_exact': overlap}
    result['sources']['ContractNLI'] = {'revision': 'eced6528dd3c1d14d73f9a87df8f7bdbc03126f9', 'splits': contract,
        'archive_sha256': fingerprint(paths['contract-nli.zip']), 'split_unit': 'whole contract/document'}
    sources = {r['source_id']: r for r in read(paths['rag-sources.jsonl'])}
    responses = read(paths['rag-responses.jsonl'])
    context = {key: content_hash(row['prompt']) for key, row in sources.items()}
    test_contexts = {context[r['source_id']] for r in responses if r['split'] == 'test'}
    rag = {}
    for split in sorted({r['split'] for r in responses}):
        subset = [r for r in responses if r['split'] == split]
        eligible = [r for r in subset if context[r['source_id']] not in test_contexts]
        rag[split] = {'responses': len(subset), 'context_groups': len({context[r['source_id']] for r in subset}),
                      'responses_matching_test_context_exact': len(subset)-len(eligible),
                      'responses_outside_test_contexts': len(eligible),
                      'label_counts_outside_test_contexts': dict(Counter('unsupported' if r['labels'] else 'supported' for r in eligible))}
    result['sources']['RAGTruth'] = {'revision': 'c103204b9ce28d6bbad859304bf30de72b8ed8fe', 'splits': rag,
        'response_sha256': fingerprint(paths['rag-responses.jsonl']), 'split_unit': 'source prompt/context, all generated responses together'}
    jobs = [
        ('MurrayTom/TS-Bench', '19047a0c4a54f54cc1ddd4a2c597105d8453b819', 'data/train/train.jsonl'),
        ('MurrayTom/TS-Bench', '19047a0c4a54f54cc1ddd4a2c597105d8453b819', 'data/train/validation.jsonl'),
        ('nvidia/When2Call', '0582f7749df63a96fdc3070932e83e72396ace53', 'train/when2call_train_sft.jsonl'),
        ('nvidia/When2Call', '0582f7749df63a96fdc3070932e83e72396ace53', 'train/when2call_train_pref.jsonl')]
    def fetch(job):
        repo, revision, filename = job
        path = hf_hub_download(repo, filename, revision=revision, repo_type='dataset')
        rows = read(path)
        item = {'repo': repo, 'revision': revision, 'file': filename, 'sha256': fingerprint(path),
                'rows': len(rows), 'columns': sorted(rows[0]) if rows else [],
                'eligibility': 'canonical source-group and historical-overlap audit pending'}
        if repo == 'MurrayTom/TS-Bench':
            eval_rows = [r for key, p in paths.items() if key.startswith(repo+'/data/eval/') for r in read(p)]
            if all('instruction' in r for r in rows+eval_rows):
                eval_groups = {content_hash(r['instruction']) for r in eval_rows}
                item['rows_matching_eval_instruction_exact'] = sum(content_hash(r['instruction']) in eval_groups for r in rows)
                item['unique_instruction_groups'] = len({content_hash(r['instruction']) for r in rows})
            if all('score' in r for r in rows):
                item['label_counts'] = dict(Counter(str(r['score']) for r in rows))
            else:
                target = rows[0].get('reward_model', {}).get('ground_truth') if rows else None
                item['target_location'] = 'reward_model.ground_truth; native target parser pending'
                item['target_type'] = type(target).__name__
                if isinstance(target, dict):
                    item['target_keys'] = sorted(target)
        return repo+'/'+filename, item
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for key, item in pool.map(fetch, jobs):
            result['sources'][key] = item
            print(json.dumps({'audited': key, 'rows': item['rows']}), flush=True)
    out = Path('data/phase3-source-audit.json')
    out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'status': 'complete', 'output': str(out), 'training_started': False}), flush=True)


if __name__ == '__main__':
    main()
