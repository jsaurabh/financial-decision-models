import copy

import pytest

from finance_decisions.operating_points import binary_curve, evaluate_frozen_threshold, fit_threshold, threshold_metrics


def rows(scores, labels, split='calibration'):
    return [{'id': f'{split}-{i}', 'input_sha256': f'{split}-input-{i}', 'task_key': 'workflow',
             'labels': ['false', 'true'], 'probabilities': [1-p, p],
             'target': 'true' if y else 'false', 'split': split}
            for i, (p, y) in enumerate(zip(scores, labels))]


def test_curve_keeps_ties_together_and_reports_no_feasible_precision():
    data = rows([.7, .7], [True, False])
    curve = binary_curve(data)
    assert len(curve['points']) == 1
    assert curve['points'][0]['precision'] == .5
    assert fit_threshold(data, min_precision=.9)['status'] == 'unattainable'
    assert fit_threshold(data, min_precision=.9)['threshold'] is None


def test_selection_uses_calibration_and_frozen_threshold_has_independent_errors():
    calibration = rows([.9, .8, .7, .6], [True, False, True, False])
    policy = fit_threshold(calibration, min_precision=.9)
    assert policy['threshold'] == .9
    assert policy['calibration_metrics']['recall'] == .5
    evaluation = rows([.95, .85, .1], [False, True, False], split='evaluation')
    measured = evaluate_frozen_threshold(policy, evaluation, calibration)
    assert measured['fp'] == 1 and measured['fn'] == 1
    assert measured['f1'] == 0
    with pytest.raises(ValueError, match='calibration records'):
        fit_threshold(evaluation)


def test_frozen_evaluation_rejects_reused_inputs_or_changed_calibration():
    calibration = rows([.9, .1], [True, False])
    policy = fit_threshold(calibration)
    evaluation = rows([.8, .2], [True, False], split='evaluation')
    evaluation[0]['input_sha256'] = calibration[0]['input_sha256']
    with pytest.raises(ValueError, match='overlap'):
        evaluate_frozen_threshold(policy, evaluation, calibration)
    with pytest.raises(ValueError, match='do not match'):
        evaluate_frozen_threshold(policy, [], [])


def test_weights_change_precision_and_partial_or_invalid_weights_are_rejected():
    data = rows([.9, .8, .1], [True, False, True])
    assert threshold_metrics(data, .8)['precision'] == .5
    weighted = copy.deepcopy(data)
    for r, w in zip(weighted, [9., 1., 1.]): r['weight'] = w
    assert threshold_metrics(weighted, .8)['precision'] == .9
    with pytest.raises(ValueError, match='every record'):
        binary_curve([weighted[0], data[1]])
    weighted[0]['weight'] = -1
    with pytest.raises(ValueError, match='nonnegative'):
        binary_curve(weighted)


def test_threshold_policies_are_task_specific_and_tie_convention_is_explicit():
    data = rows([.5, .2], [True, False])
    assert threshold_metrics(data, .5)['tp'] == 1
    assert threshold_metrics(data, .5, inclusive=False)['tp'] == 0
    data[1]['task_key'] = 'another-workflow'
    with pytest.raises(ValueError, match='separate threshold'):
        fit_threshold(data)
