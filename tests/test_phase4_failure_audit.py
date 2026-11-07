import pytest

from finance_decisions.phase4_failure_audit import financial_pairs


def test_financial_pairs_detect_constant_predictions_despite_balanced_accuracy():
    rows = [dict(id=target, task='financial_verification', target=target,
                 source={}, metadata={'pair': 'same-question'}, question={'type': 'noul'})
            for target in ['false', 'true']]
    predictions = {r['id']: {'status': 'ok', 'prediction': 'false', 'probabilities': [.9, .1]} for r in rows}
    result = financial_pairs(rows, predictions)
    assert result['same_prediction_pairs'] == 1
    assert result['both_correct_pairs'] == result['both_wrong_pairs'] == 0
    assert result['roc_auc_valid'] == .5
    with pytest.raises(ValueError, match='complete opposite-label pairs'):
        financial_pairs(rows[:1], predictions)
