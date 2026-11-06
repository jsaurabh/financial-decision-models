"""Strict paired comparison: identical inputs and candidate orders, deduplicated inputs."""
import json
from pathlib import Path

import numpy as np

from .evaluate_jev import summarize


def read(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines()]


def pair_records(gemma, jev):
    jev_map = {r['id']:r for r in jev}
    left, right, seen = [], [], set()
    for record in gemma:
        other = jev_map.get(record['id'])
        if other is None:
            continue
        if record['input_sha256'] != other['input_sha256'] or record['labels'] != other['labels']:
            raise ValueError(f'Input or candidate-order mismatch for {record["id"]}')
        if record['target'] != other['target']:
            raise ValueError(f'Reference mismatch for {record["id"]}')
        # Synthetic probes may repeat a semantic input with different case IDs.
        if record['input_sha256'] in seen:
            continue
        seen.add(record['input_sha256']); left.append(record); right.append(other)
    return left, right


def comparison(gemma, jev, requested):
    left, right = pair_records(gemma, jev)
    result = {'paired_unique_inputs':len(left), 'gemma_records_before_dedup':len(gemma),
              'jev_records_before_dedup':len(jev), 'dataset_rows':requested,
              'input_hashes_and_candidate_orders_match':True, 'gemma':summarize(left), 'jev':summarize(right)}
    result['paired_outcomes'] = {
        'both_correct':sum(a['correct'] and b['correct'] for a,b in zip(left,right)),
        'gemma_only_correct':sum(a['correct'] and not b['correct'] for a,b in zip(left,right)),
        'jev_only_correct':sum(not a['correct'] and b['correct'] for a,b in zip(left,right)),
        'both_wrong':sum(not a['correct'] and not b['correct'] for a,b in zip(left,right))}
    result['domains'] = {domain:{'n':sum(r['domain']==domain for r in left),
        'gemma':summarize([r for r in left if r['domain']==domain]),
        'jev':summarize([r for r in right if r['domain']==domain])} for domain in sorted({r['domain'] for r in left})}
    # Keep the original case-weighted benchmark available next to deduplicated metrics.
    jev_map = {r['id']:r for r in jev}
    raw_left = [r for r in gemma if r['id'] in jev_map]
    raw_right = [jev_map[r['id']] for r in raw_left]
    result['case_weighted_original_protocol'] = {'n':len(raw_left),'gemma':summarize(raw_left),'jev':summarize(raw_right)}
    return result


def main():
    live_summary=json.loads(Path('results/jev-live/summary.json').read_text())
    result = {'jev_models':live_summary['resolved_models'],'matching':'input SHA-256, candidate order, target; repeated semantic inputs deduplicated',
              'limits':['BANKING77 and LEDGAR use eight sampled candidates, not full-label benchmarks.',
                        'The original test and frozen banking results were inspected during development; improvements are development results.',
                        'The synthetic probe was inspected after V3 and is a development probe for V4. The public-fresh holdout is final-only for V4.',
                        'Each suite has its own matched coverage. API probability rounding can affect NLL.'], 'versions':{}}
    for version in ['v1','v2','v3','v4']:
        path = Path('results') / f'gemma4-finance-{version}'
        if not path.exists():
            continue
        suites = {}
        for suite,filename in [('test','test.jsonl'),('legacy-banking-128','legacy-banking-128.jsonl'),('fresh-robustness','fresh-robustness.jsonl'),('public-fresh','public-fresh.jsonl')]:
            gemma_path = path / f'{suite}-predictions.jsonl'; jev_path = Path('results/jev-live') / f'{suite}-predictions.jsonl'
            if gemma_path.exists() and jev_path.exists():
                requested = len(read(Path('data/private') / filename))
                suites[suite] = comparison(read(gemma_path),read(jev_path),requested)
        result['versions'][version] = suites
    Path('results/gemma-vs-jev.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({v:{s:{'n':r['paired_unique_inputs'],'gemma':r['gemma']['accuracy'],'jev':r['jev']['accuracy']}
                       for s,r in suites.items()} for v,suites in result['versions'].items()},indent=2))


if __name__ == '__main__':
    main()
