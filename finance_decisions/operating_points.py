"""Binary precision/recall diagnostics and calibration-only threshold fitting.

Weights must come from a declared sampling design. They are never inferred
from this public benchmark's class prevalence.
"""
import hashlib
import json

import numpy as np
from sklearn.metrics import average_precision_score


def _fingerprint(records):
    lines = sorted(json.dumps(r, sort_keys=True, allow_nan=False) for r in records)
    return hashlib.sha256('\n'.join(lines).encode()).hexdigest()


def _arrays(records):
    if not records:
        raise ValueError('At least one binary record is required')
    weighted = ['weight' in row for row in records]
    if any(weighted) and not all(weighted):
        raise ValueError('Supply a weight for every record, or none')
    truth, score, weights = [], [], []
    for row in records:
        if row.get('labels') != ['false', 'true'] or row.get('target') not in ('false', 'true'):
            raise ValueError('Expected false/true binary records')
        p = np.asarray(row['probabilities'], dtype=float)
        if p.shape != (2,) or not np.isfinite(p).all() or (p < 0).any() or (p > 1).any() or not np.isclose(p.sum(), 1, atol=1e-6):
            raise ValueError('Expected two finite probabilities summing to one')
        truth.append(row['target'] == 'true'); score.append(p[1])
        weights.append(float(row.get('weight', 1)))
    weights = np.asarray(weights)
    if not np.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
        raise ValueError('Weights must be finite, nonnegative, and have positive total')
    return np.asarray(truth), np.asarray(score), weights


def _point(tp, fp, positives, negatives, predicted_count, threshold):
    precision = float(tp / (tp + fp)) if tp + fp > 0 else None
    recall = float(tp / positives) if positives > 0 else None
    fn, tn = positives - tp, negatives - fp
    return {'threshold': float(threshold), 'precision': precision, 'recall': recall,
            'f1': float(2 * tp / (2 * tp + fp + fn)) if 2 * tp + fp + fn > 0 else 0.,
            'tp': float(tp), 'fp': float(fp), 'tn': float(tn), 'fn': float(fn),
            'predicted_positive_count': int(predicted_count)}


def threshold_metrics(records, threshold, *, inclusive=True):
    truth, score, weights = _arrays(records)
    if not np.isfinite(threshold):
        raise ValueError('Threshold must be finite')
    predicted = score >= threshold if inclusive else score > threshold
    return _point(weights[truth & predicted].sum(), weights[~truth & predicted].sum(),
                  weights[truth].sum(), weights[~truth].sum(), predicted.sum(), threshold)


def binary_curve(records):
    truth, score, weights = _arrays(records)
    order = np.argsort(-score, kind='stable')
    y, p, w = truth[order], score[order], weights[order]
    # Ties move together: a threshold cannot split equal returned scores.
    ends = np.r_[np.flatnonzero(np.diff(p)), len(p) - 1]
    tp, fp = np.cumsum(w * y), np.cumsum(w * ~y)
    positives, negatives = weights[truth].sum(), weights[~truth].sum()
    points = [_point(tp[i], fp[i], positives, negatives, i + 1, p[i]) for i in ends]
    return {'n': len(records), 'positives': int(truth.sum()), 'negatives': int((~truth).sum()),
            'weighted': 'weight' in records[0], 'weighted_positive_mass': float(positives),
            'weighted_negative_mass': float(negatives),
            'average_precision': float(average_precision_score(truth, score, sample_weight=weights))
                                 if positives > 0 and negatives > 0 else None,
            'comparison_rule': 'p_true >= threshold; equal scores move together', 'points': points}


def fit_threshold(calibration_records, *, min_precision=None, min_predicted_positives=1):
    """Fit maximum F1, or maximum recall subject to a precision constraint.

    Each row must explicitly identify itself as calibration data. A selected
    threshold is an empirical operating point, not a precision guarantee.
    """
    if not calibration_records or any(r.get('split') != 'calibration' for r in calibration_records):
        raise ValueError('Threshold selection requires explicitly tagged calibration records')
    if any(not r.get('id') or not r.get('input_sha256') or not r.get('task_key') for r in calibration_records):
        raise ValueError('Calibration requires IDs, input hashes, and a task_key')
    if len({r['id'] for r in calibration_records}) != len(calibration_records):
        raise ValueError('Calibration IDs must be unique')
    if len({r['task_key'] for r in calibration_records}) != 1:
        raise ValueError('Fit a separate threshold for each task_key')
    if min_precision is not None and not 0 < min_precision <= 1:
        raise ValueError('min_precision must be in (0, 1]')
    if min_predicted_positives < 1:
        raise ValueError('Require at least one predicted positive')
    curve = binary_curve(calibration_records)
    if curve['weighted_positive_mass'] == 0 or curve['weighted_negative_mass'] == 0:
        raise ValueError('Calibration requires positive and negative examples')
    feasible = [p for p in curve['points'] if p['predicted_positive_count'] >= min_predicted_positives
                and p['precision'] is not None and (min_precision is None or p['precision'] >= min_precision)]
    objective = 'maximum_f1' if min_precision is None else 'maximum_recall_at_minimum_precision'
    result = {'status': 'unattainable', 'threshold': None, 'objective': objective,
              'selection_split': 'calibration', 'n': curve['n'], 'positives': curve['positives'],
              'min_precision': min_precision, 'min_predicted_positives': min_predicted_positives,
              'task_key': calibration_records[0]['task_key'], 'calibration_sha256': _fingerprint(calibration_records),
              'weighted': curve['weighted'], 'note': 'Empirical selection; evaluate the frozen threshold on independent data.'}
    if feasible:
        key = (lambda p: (p['f1'], p['precision'], p['threshold'])) if min_precision is None else (
            lambda p: (p['recall'], p['precision'], p['threshold']))
        selected = max(feasible, key=key)
        result.update(status='selected', threshold=selected['threshold'], calibration_metrics=selected)
    return result


def evaluate_frozen_threshold(policy, evaluation_records, calibration_records):
    """Apply an already selected policy and reject overlapping calibration inputs."""
    if policy.get('status') != 'selected' or policy.get('selection_split') != 'calibration':
        raise ValueError('A selected calibration policy is required')
    if _fingerprint(calibration_records) != policy.get('calibration_sha256'):
        raise ValueError('Calibration records do not match the frozen policy')
    if any(not r.get('id') or not r.get('input_sha256') for r in evaluation_records):
        raise ValueError('Evaluation requires IDs and input hashes')
    if any(r.get('task_key') != policy.get('task_key') for r in evaluation_records):
        raise ValueError('Policy and evaluation task_key must match')
    for key in ['id', 'input_sha256']:
        left = {r[key] for r in calibration_records if r.get(key)}
        right = {r[key] for r in evaluation_records if r.get(key)}
        if left & right:
            raise ValueError(f'Calibration/evaluation {key} overlap')
    if any(r.get('split') == 'calibration' for r in evaluation_records):
        raise ValueError('Evaluation records must be independent of calibration')
    return {**threshold_metrics(evaluation_records, policy['threshold']),
            'n': len(evaluation_records), 'weighted': any('weight' in r for r in evaluation_records),
            'task_key': policy['task_key'], 'threshold_frozen': True}
