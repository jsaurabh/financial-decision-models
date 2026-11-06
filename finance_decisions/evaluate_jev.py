"""Live matched-input Jev benchmark with hash-checked incremental caching.

API calls contain state and question only. Gold labels are joined afterwards.
"""
import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import requests
from sklearn.metrics import f1_score, precision_recall_fscore_support, average_precision_score

from .metrics import metrics
from .schema import options

URL = 'https://api.typesafe.ai/v1/systemone'


def digest(row):
    return hashlib.sha256(json.dumps({'state': row['state'], 'question': row['question']},
        sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()


def call(row, alias):
    headers = {'Authorization': 'Bearer ' + os.environ['TYPESAFE_API_KEY'], 'Content-Type': 'application/json'}
    payload = {'model': alias, 'state': row['state'], 'questions': {'decision': row['question']}}
    start = time.perf_counter()
    for attempt in range(5):
        try:
            response = requests.post(URL, headers=headers, json=payload, timeout=60)
            if response.status_code in (401,403):
                raise RuntimeError('Jev rejected the configured API credential')
            if response.status_code in (429,500,502,503,504):
                if attempt == 4:
                    raise RuntimeError(f'Jev HTTP {response.status_code} after retries')
                time.sleep(min(2**attempt, 8)); continue
            if response.status_code != 200:
                raise RuntimeError(f'Jev HTTP {response.status_code}')
            raw = response.json(); value = raw['answers']['decision']
            labels, _ = options(row['question'])
            if row['question']['type'] == 'noul':
                p = [1 - float(value['noul']), float(value['noul'])]
            else:
                p = [float(value['probabilities'][label]) for label in labels]
            if not all(math.isfinite(v) and 0 <= v <= 1 for v in p) or not .98 <= sum(p) <= 1.02:
                raise ValueError('Invalid probability distribution from Jev')
            p = [v / sum(p) for v in p]
            prediction = value.get('choice') if row['question']['type'] == 'choice' else None
            prediction = prediction if prediction in labels else labels[int(np.argmax(p))]
            return {'id': row['id'], 'input_sha256': digest(row), 'domain': row['domain'],
                    'labels': labels, 'probabilities': p, 'prediction': prediction,
                    'provider': raw['model'], 'provider_confidence': value.get('confidence'),
                    'usage': raw.get('usage'), 'latency_s_including_network': time.perf_counter() - start,
                    'attempts': attempt + 1, 'timestamp_utc': datetime.now(timezone.utc).isoformat()}
        except (requests.Timeout, requests.ConnectionError):
            if attempt == 4:
                raise RuntimeError('Jev transport failed after retries') from None
            time.sleep(min(2**attempt,8))


def summarize(records):
    logits = np.full((len(records), 8), -1e4)
    targets = []
    for i,r in enumerate(records):
        logits[i,:len(r['labels'])] = np.log(np.maximum(r['probabilities'], 1e-12))
        targets.append(r['labels'].index(r['target']))
    result = metrics(logits, targets)
    result['accuracy'] = float(np.mean([r['correct'] for r in records]))
    result['semantic_macro_f1'] = float(f1_score([r['target'] for r in records], [r['prediction'] for r in records], average='macro', zero_division=0))
    if records and all(r['labels']==['false','true'] for r in records):
        truth = [r['target']=='true' for r in records]; pred = [r['prediction']=='true' for r in records]
        precision,recall,f1,_ = precision_recall_fscore_support(truth,pred,average='binary',zero_division=0)
        result['positive_class'] = {'precision':float(precision),'recall':float(recall),'f1':float(f1),
            'average_precision':float(average_precision_score(truth,[r['probabilities'][1] for r in records]))}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='jev-latest')
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--out', default='results/jev-live')
    args = parser.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    suites = {name:[json.loads(x) for x in Path(path).read_text().splitlines()] for name,path in [
        ('test','data/private/test.jsonl'), ('legacy-banking-128','data/private/legacy-banking-128.jsonl'),
        ('fresh-robustness','data/private/fresh-robustness.jsonl')]}
    if Path('data/private/public-fresh.jsonl').exists():
        suites['public-fresh']=[json.loads(x) for x in Path('data/private/public-fresh.jsonl').read_text().splitlines()]
    all_rows = [r for rows in suites.values() for r in rows]
    cache_path = out / 'cache.jsonl'
    cached = {}
    if cache_path.exists():
        for line in cache_path.read_text().splitlines():
            item = json.loads(line); cached[(item['id'],item['input_sha256'])] = item
    pending = [r for r in all_rows if (r['id'],digest(r)) not in cached]
    errors = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(call, row, args.model): row for row in pending}
        for number, future in enumerate(concurrent.futures.as_completed(futures),1):
            row = futures[future]
            try:
                item = future.result()
                cached[(item['id'],item['input_sha256'])] = item
                with cache_path.open('a') as f:
                    f.write(json.dumps(item, allow_nan=False)+'\n')
            except Exception as exc:
                errors.append({'id':row['id'],'input_sha256':digest(row),'error_type':type(exc).__name__,'error':str(exc)})
            if number % 100 == 0 or number == len(pending):
                print(json.dumps({'completed':number,'requested':len(pending),'errors':len(errors)}),flush=True)
    (out / 'errors.json').write_text(json.dumps(errors,indent=2))
    summary = {'requested_alias':args.model,'resolved_models':dict(Counter(r['provider'] for r in cached.values())),
               'api_failures':len(errors),'suites':{},'note':'Jev uses native returned choice for top-1; NLL uses API-returned probabilities clipped at 1e-12. No latency claim against local GPU/CPU inference.'}
    for name,rows in suites.items():
        records = []
        for row in rows:
            item = cached.get((row['id'],digest(row)))
            if item is not None:
                records.append({**item,'target':row['target'],'correct':item['prediction']==row['target']})
        if not records:
            continue
        score = summarize(records)
        score.update({'requested':len(rows),'valid':len(records),'domains':{domain:summarize([r for r in records if r['domain']==domain]) for domain in sorted({r['domain'] for r in records})}})
        summary['suites'][name] = score
        (out / f'{name}-predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    summary['usage'] = dict(sum((Counter(r.get('usage') or {}) for r in cached.values()), Counter()))
    (out / 'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2),flush=True)
    if errors:
        raise SystemExit(f'{len(errors)} API requests failed; rerun to retry uncached cases')


if __name__ == '__main__':
    main()
