"""Post-evaluation diagnostics; never a fresh-test or threshold-selection report."""
import json
from collections import Counter, defaultdict
from pathlib import Path

from sklearn.metrics import average_precision_score, roc_auc_score

from .phase3_data import read, sha
from .phase4_eval import validate
from .phase4_metrics import labels


def financial_pairs(rows, predictions):
    pairs = defaultdict(list)
    for row in rows:
        identity = row['metadata'].get('pair', row['source'].get('id'))
        if identity is None:
            raise ValueError('Missing financial pair identity')
        pairs[(row['task'], identity)].append(row)
    counts = Counter()
    for items in pairs.values():
        if len(items) != 2 or {r['target'] for r in items} != {'false', 'true'}:
            raise ValueError('Financial diagnostic requires complete opposite-label pairs')
        counts['requested_pairs'] += 1
        if any(predictions.get(r['id'], {}).get('status') != 'ok' for r in items):
            counts['incomplete_prediction_pairs'] += 1
            continue
        counts['valid_pairs'] += 1
        guesses = [predictions[r['id']]['prediction'] for r in items]
        hits = sum(p == r['target'] for r, p in zip(items, guesses))
        counts['same_prediction_pairs'] += guesses[0] == guesses[1]
        counts['both_correct_pairs'] += hits == 2
        counts['both_wrong_pairs'] += hits == 0
    valid = [r for r in rows if predictions.get(r['id'], {}).get('status') == 'ok']
    truth = [r['target'] == 'true' for r in valid]
    probability = [predictions[r['id']]['probabilities'][labels(r).index('true')] for r in valid]
    return {
        **dict(counts),
        'predicted_labels': dict(Counter(predictions[r['id']]['prediction'] for r in valid)),
        'positive_fraction_valid': sum(truth) / len(truth) if truth else None,
        'roc_auc_valid': float(roc_auc_score(truth, probability)) if len(set(truth)) == 2 else None,
        'average_precision_valid': float(average_precision_score(truth, probability)) if any(truth) else None,
        'thresholds_selected_on_these_cases': False,
    }


def trace_diagnostics(rows, predictions):
    valid = [r for r in rows if predictions.get(r['id'], {}).get('status') == 'ok']
    supported_risk = [r for r in valid if r['target'] != 'false']
    lengths = {}
    for lower, upper in [(0, 2048), (2049, 4096), (4097, 8192), (8193, None)]:
        part = [r for r in rows if r['token_lengths']['gemma'] is not None
                and r['token_lengths']['gemma'] >= lower
                and (upper is None or r['token_lengths']['gemma'] <= upper)]
        if part:
            lengths[f'{lower}-{upper or "above"}'] = {
                'requested': len(part),
                'valid': sum(predictions.get(r['id'], {}).get('status') == 'ok' for r in part),
                'correct': sum(predictions.get(r['id'], {}).get('status') == 'ok'
                               and predictions[r['id']]['prediction'] == r['target'] for r in part),
            }
    return {
        'requested': len(rows), 'valid': len(valid),
        'predicted_labels': dict(Counter(predictions[r['id']]['prediction'] for r in valid)),
        'valid_risky_cases': len(supported_risk),
        'risky_cases_detected': sum(predictions[r['id']]['prediction'] == 'true' for r in supported_risk),
        'gemma_token_length_bins': lengths,
    }


def main():
    manifest = Path('data/phase4-manifest.json')
    comparison = Path('results/phase4/comparison.json')
    aggregate = json.loads(comparison.read_text())
    rows = []
    for name, entry in [('core_test', 'splits'), ('transfer_test', 'transfer_test')]:
        path = Path(f'data/private/phase4/{name}.jsonl')
        config = json.loads(manifest.read_text())
        expected = config[entry][name]['sha256'] if entry == 'splits' else config[entry]['sha256']
        assert sha(path.read_bytes()) == expected
        rows.extend(read(path))
    byid = {r['id']: r for r in rows}
    systems = {}
    for role in ['gemma', 'jev']:
        receipts = read(f'data/private/phase4-report/{role}/predictions.jsonl')
        predictions = {p['id']: p for p in receipts}
        assert len(predictions) == len(receipts) == len(rows)
        for identity, receipt in predictions.items():
            validate(byid[identity], receipt)
        systems[role] = predictions
    financial = {}
    for task in ['financial_verification', 'transfer/finqa_verification']:
        part = [r for r in rows if r['task'] == task]
        financial[task] = {role: financial_pairs(part, ps) for role, ps in systems.items()}
    trace = [r for r in rows if r['task'].startswith('transfer/tracesafe/binary/')]
    routing = {}
    for task in ['routing_quality', 'routing_costaware']:
        routing[task] = {
            role: aggregate['cohorts']['core'][role]['tasks'][task]['routing_valid_only']
            for role in systems}
        routing[task]['fixed_model_chosen_from_training'] = aggregate['cohorts']['core']['gemma']['tasks'][task]['training_profile_router_valid_only']
        routing[task]['prediction_counts'] = {
            role: dict(Counter(ps[r['id']]['prediction'] for r in rows if r['task'] == task))
            for role, ps in systems.items()}
    train = read('data/private/phase4/train.jsonl')
    result = {
        'status': 'posthoc_failure_diagnostics',
        'manifest_sha256': sha(manifest.read_bytes()),
        'comparison_sha256': sha(comparison.read_bytes()),
        'artifact': aggregate['artifacts']['gemma'],
        'exposure_policy': 'These completed benchmark results now inform development. Preserve original receipts; subsequent scores on these cases are regression evidence, not a new untouched confirmation.',
        'financial': financial,
        'financial_training_labels': dict(Counter(r['target'] for r in train if r['task'] == 'financial_verification')),
        'tracesafe_binary': {role: trace_diagnostics(trace, ps) for role, ps in systems.items()},
        'routing': routing,
        'safety_tradeoffs': {
            role: aggregate['cohorts']['core'][role]['tasks']['tool_action_safety']['strict_safety_valid_only']
            for role in systems},
        'causal_conclusion': 'Observed prediction collapse and capability gaps. This audit alone does not distinguish backbone limits, answer-code binding, training objectives, multi-task interference, or dataset adaptation errors.',
    }
    out = Path('results/phase4/failure-audit.json')
    out.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
