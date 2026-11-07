"""Pinned public text datasets and explicit-policy synthetic cases; no customer data."""
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from datasets import load_dataset

SEED = 20261105
SOURCES = {
    'banking77': {'repo': 'PolyAI/banking77', 'revision': '1846984189f2bc0ac51c6fc3a600fa9e1714e0c6', 'license': 'CC-BY-4.0'},
    'lex_glue': {'repo': 'coastalcph/lex_glue', 'revision': 'c23fdff1a6bf74e0e1a71cb86f1e781d37da888c', 'license': 'CC-BY-4.0; underlying sources retain their terms'},
}


def record(rid, domain, state, instructions, criteria, target, kind='choice', family=None):
    return {'id': rid, 'family': family or rid, 'domain': domain, 'state': state,
            'question': {'type': kind, 'instructions': instructions, 'criteria': criteria}, 'target': str(target)}


def sampled_options(names, target, rng, n=8):
    labels = rng.sample([x for x in names if x != target], min(n - 1, len(names) - 1)) + [target]
    rng.shuffle(labels)
    return {v: v.replace('_', ' ') for v in labels}


def public_rows():
    rng = random.Random(SEED)
    result = {'train': [], 'calibration': [], 'test': []}
    for alias, subset, label_field, domain, limit in [
        ('banking77', None, 'label', 'banking_intent', 3500),
        ('lex_glue', 'ledgar', 'label', 'legal_contract', 2500),
        ('lex_glue', 'unfair_tos', 'labels', 'legal_terms', 1500),
    ]:
        source = SOURCES[alias]
        ds = load_dataset(source['repo'], subset, revision=source['revision'])
        features = ds['train'].features[label_field]
        names = features.feature.names if label_field == 'labels' else features.names
        train_indices = list(range(len(ds['train']))); rng.shuffle(train_indices)
        if 'validation' in ds:
            pools = [('train', 'train', train_indices[:limit]),
                     ('calibration', 'validation', list(range(min(160, len(ds['validation']))))),
                     ('test', 'test', list(range(min(240, len(ds['test'])))))]
        else:
            pools = [('train', 'train', train_indices[:limit]),
                     ('calibration', 'train', train_indices[limit:limit+160]),
                     ('test', 'test', list(range(240)))]
        for split, native, indices in pools:
            for i in indices:
                row = ds[native][i]
                rid = f'{alias}-{subset}-{native}-{i}'
                if label_field == 'labels':
                    positives = list(row['labels'])
                    # Half positive when available; otherwise choose a negative predicate.
                    k = rng.choice(positives) if positives and rng.random() < .5 else rng.randrange(len(names))
                    item = record(rid, domain, row['text'], f'Does this clause contain {names[k].lower()}?',
                                  None, int(k in positives), kind='noul')
                    item['target'] = 'true' if k in positives else 'false'
                else:
                    target = names[row['label']]
                    criteria = sampled_options(names, target, rng)
                    item = record(rid, domain, row['text'],
                                  'Choose the best matching banking intent.' if alias == 'banking77' else 'Choose the contract clause topic.',
                                  criteria, target)
                item['source'] = {**source, 'subset': subset, 'native_split': native, 'row': i}
                result[split].append(item)
    return result


def synthetic_rows():
    rng = random.Random(SEED + 1)
    result = {'train': [], 'calibration': [], 'test': []}
    domains = ['aml_controls', 'credit_operations', 'insurance', 'investment_operations',
               'payments', 'data_privacy', 'financial_reporting', 'legal_workflow']
    for domain in domains:
        for i in range(240):
            split = 'test' if i % 10 == 0 else 'calibration' if i % 10 == 1 else 'train'
            cap = rng.choice([3, 5, 7, 10, 15, 25, 40, 60, 90])
            # Counterfactuals, including unknown and equality, remain in the same split.
            family = f'{domain}-{i}'
            noun = {'aml_controls':'verified independent indicators', 'credit_operations':'verified exposure units',
                    'insurance':'days since receipt of claim', 'investment_operations':'portfolio concentration percent',
                    'payments':'days since transaction posting', 'data_privacy':'days since verified request',
                    'financial_reporting':'material variance percent', 'legal_workflow':'days since service of notice'}[domain]
            for j, value in enumerate([cap - 1, cap, None]):
                text = f'Case {i}: observed {noun}: {value if value is not None else "unknown"}. Evidence is current. No override applies.'
                if split == 'test':
                    text = f'Assessment record {i}. {noun.capitalize()} were measured at {value if value is not None else "not established"}; there is no policy exception.'
                criteria = {'standard': f'Use when the observed value is below {cap}.',
                            'enhanced': f'Use when the observed value is at least {cap}.',
                            'review': 'Use when the value is unknown or evidence conflicts.'}
                item = record(f'{family}-{j}', domain, text,
                              f'Apply this hypothetical internal policy: below {cap} uses standard processing; at least {cap} uses enhanced processing; unknown requires review. This is a supplied policy, not a statement of law.',
                              criteria, 'review' if value is None else 'enhanced' if value >= cap else 'standard', family=family)
                item['source'] = {'generator': 'explicit-policy-v1', 'human_reviewed': False}
                result[split].append(item)
    routes = {
        'lookup': ['Find the fee schedule for wire payments.', 'Retrieve our opening hours.', 'Locate the account terms.', 'Look up the fund factsheet.'],
        'reasoning': ['Reconcile conflicting ledger entries and explain the discrepancy.', 'Analyze counterparty exposure across subsidiaries.', 'Compare covenant exceptions in three loan agreements.', 'Assess a complex multi-jurisdiction transaction.'],
        'code': ['Write Python to reconcile the cash ledger.', 'Debug the SQL that calculates account balances.', 'Implement a bond duration calculator.', 'Create unit tests for the payment parser.'],
        'human': ['Authorize an exception to the sanctions control.', 'Approve this account freeze.', 'Release a blocked payment without the required evidence.', 'Make the final legal determination on this disputed contract.'],
    }
    # Template families are held out as groups. Routing labels express declared capabilities,
    # not a measured ranking of commercial models.
    for k, (target, templates) in enumerate(routes.items()):
        for t, template in enumerate(templates):
            split = 'test' if t == 3 else 'calibration' if t == 2 else 'train'
            for i in range(80):
                criteria = {'lookup': 'A retrieval service for simple factual document lookup.',
                            'reasoning': 'A reasoning model for multi-step analysis.',
                            'code': 'A coding model for implementation and debugging.',
                            'human': 'A human reviewer for approvals and irreversible regulated decisions.'}
                item = record(f'route-{k}-{t}-{i}', 'model_routing', template + f' Internal work item {i}.',
                              'Choose the least-capable adequate route from the supplied capability descriptions. Required approvals go to a human.',
                              criteria, target, family=f'route-template-{k}-{t}')
                item['source'] = {'generator': 'capability-routing-v1', 'human_reviewed': False}
                result[split].append(item)
    # Ordinal scores use explicit numerical rubrics, not implied creditworthiness.
    for i in range(320):
        split = 'test' if i % 10 == 0 else 'calibration' if i % 10 == 1 else 'train'
        for n in range(3):
            item = record(f'score-{i}-{n}', 'document_completeness',
                          f'File {i} has {n} verified required documents out of 2.',
                          'Score completeness: 0 means no verified documents; 1 means one; 2 means both.',
                          ['No required documents verified', 'One required document verified', 'Both required documents verified'],
                          n, kind='score', family=f'score-{i}')
            item['source'] = {'generator': 'completeness-v1', 'human_reviewed': False}
            result[split].append(item)
    return result


def main():
    out = Path('data/private'); out.mkdir(parents=True, exist_ok=True)
    public, synthetic = public_rows(), synthetic_rows()
    seen_text, seen_families, manifest = set(), set(), {'seed': SEED, 'sources': SOURCES, 'splits': {}}
    # Test/calibration take precedence when a public source contains repeated clauses.
    for split in ['test', 'calibration', 'train']:
        rows = public[split] + synthetic[split]
        random.Random(SEED).shuffle(rows)
        accepted, current_hashes = [], set()
        families = {r['family'] for r in rows}
        assert not families & seen_families
        seen_families |= families
        for r in rows:
            text_hash = hashlib.sha256(' '.join(r['state'].lower().split()).encode()).hexdigest()
            if text_hash in seen_text:
                continue
            current_hashes.add(text_hash)
            accepted.append(r)
        seen_text |= current_hashes
        path = out / f'{split}.jsonl'
        path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in accepted))
        manifest['splits'][split] = {'rows': len(accepted), 'domains': dict(Counter(r['domain'] for r in accepted)),
                                    'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    manifest['limitations'] = ['Banking/LEDGAR use 8 sampled candidates including the gold label, not full 77/100-way classification.',
                              'Synthetic policy cases are not human reviewed and do not assert current legal requirements.',
                              'Routing is capability-based; no empirical model-utility oracle is available.',
                              'No private production evaluation data supplied. Public pretraining contamination cannot be excluded.']
    Path('data/manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
