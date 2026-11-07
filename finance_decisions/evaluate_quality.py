"""Auditable predictions and semantic-label metrics from the saved release."""
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.special import softmax
from sklearn.metrics import f1_score, average_precision_score, precision_recall_fscore_support

from .model import DecisionModel, targets
from .schema import options
from .train import read, filter_rows, infer, report, shuffled_options, upload
import random


def evaluate(model, rows, out, filename, max_length=None):
    if max_length:
        model.max_length = max_length
    accepted, dropped = filter_rows(model, rows)
    logits = infer(model, accepted, batch_size=8)
    result = report(model, accepted, logits)
    result.update({'temperature': model.temperature, 'requested_rows': len(rows),
                   'dropped_overlength': dropped, 'max_length': model.max_length})
    records = []
    for row, logit in zip(accepted, logits):
        labels, _ = options(row['question']); p = softmax(np.array(logit[:len(labels)]) / model.temperature)
        predicted = labels[int(p.argmax())]
        records.append({'id': row['id'], 'domain': row['domain'], 'target': row['target'],
                        'split': 'calibration' if filename == 'calibration' else 'evaluation',
                        'task_key': row['domain'] + ':' + hashlib.sha256(json.dumps(row['question'],sort_keys=True,ensure_ascii=False).encode()).hexdigest(),
                        'prediction': predicted, 'correct': predicted == row['target'], 'labels': labels,
                        'probabilities': p.tolist(), 'max_probability': float(p.max()),
                        'input_sha256': hashlib.sha256(json.dumps({'state': row['state'], 'question': row['question']},
                             sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()})
        if 'weight' in row:
            records[-1]['weight'] = row['weight']
    for domain, score in result['domains'].items():
        domain_records = [r for r in records if r['domain'] == domain]
        score['semantic_macro_f1'] = float(f1_score([r['target'] for r in domain_records],
                                                    [r['prediction'] for r in domain_records], average='macro', zero_division=0))
        if domain_records and all(r['labels'] == ['false', 'true'] for r in domain_records):
            truth = [r['target'] == 'true' for r in domain_records]
            prediction = [r['prediction'] == 'true' for r in domain_records]
            precision, recall, f1, _ = precision_recall_fscore_support(truth, prediction, average='binary', zero_division=0)
            score['positive_class'] = {'precision': float(precision), 'recall': float(recall), 'f1': float(f1),
                'prevalence': float(np.mean(truth)),
                'average_precision': float(average_precision_score(truth, [r['probabilities'][1] for r in domain_records]))}
    reordered = shuffled_options(accepted, random.Random(42))
    order_logits = infer(model, reordered, batch_size=8)
    order_labels = [options(r['question'])[0][int(np.argmax(l))] for r,l in zip(reordered, order_logits)]
    result['order_control'] = {'accuracy': float(np.mean([label == row['target'] for label,row in zip(order_labels, reordered)])),
                               'semantic_prediction_agreement': float(np.mean([label == r['prediction'] for label,r in zip(order_labels, records)]))}
    choice_indices = [i for i,r in enumerate(accepted) if r['question']['type'] == 'choice']
    if choice_indices:
        result['order_control']['choice_only_accuracy'] = float(np.mean([order_labels[i] == accepted[i]['target'] for i in choice_indices]))
        result['order_control']['choice_only_prediction_agreement'] = float(np.mean([order_labels[i] == records[i]['prediction'] for i in choice_indices]))
    (out / f'{filename}.json').write_text(json.dumps(result, indent=2))
    (out / f'{filename}-predictions.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in records))
    errors = [r for r in records if not r['correct']]
    (out / f'{filename}-errors.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in errors))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--artifact', default='runs/gemma4-finance-v1/release')
    parser.add_argument('--out', default='results/gemma4-finance-v1')
    parser.add_argument('--hf-repo', default='jsaurabh/gemma4-e2b-finance-decisions-v1')
    parser.add_argument('--fresh-probe', action='store_true')
    parser.add_argument('--public-holdout', action='store_true')
    parser.add_argument('--export-calibration', action='store_true', help='Save separate calibration predictions for per-task threshold selection')
    args = parser.parse_args()
    model = DecisionModel.load(args.artifact, device='cuda')
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    if args.export_calibration:
        evaluate(model, read('data/private/calibration.jsonl'), out, 'calibration')
    print(json.dumps({'test': evaluate(model, read('data/private/test.jsonl'), out, 'test')}, indent=2), flush=True)
    if args.fresh_probe:
        print(json.dumps({'fresh_robustness': evaluate(model, read('data/private/fresh-robustness.jsonl'), out, 'fresh-robustness')}, indent=2), flush=True)
    if args.public_holdout:
        print(json.dumps({'public_fresh': evaluate(model, read('data/private/public-fresh.jsonl'), out, 'public-fresh')},indent=2),flush=True)
    # The historical evaluation is longer than the training cap. We preserve it in full
    # and state this explicitly instead of silently truncating or dropping hard cases.
    legacy = evaluate(model, read('data/private/legacy-banking-128.jsonl'), out, 'legacy-banking-128', max_length=1024)
    legacy['historical_comparison'] = json.loads(Path('data/legacy-banking-manifest.json').read_text())['historical_results']
    (out / 'legacy-banking-128.json').write_text(json.dumps(legacy, indent=2))
    upload(out, args.hf_repo, 'quality')
    print(json.dumps({'legacy': legacy, 'status': 'quality_evaluation_complete'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
