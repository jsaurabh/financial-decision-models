import pytest

from finance_decisions.evaluate_jev import digest
from finance_decisions.phase3_jev_compare import summarize, validate


def row(identity,target):
    return {'id':identity,'family':identity,'domain':'risk','state':'Evidence '+identity,'target':target,
            'question':{'type':'noul','instructions':'Does this contain a risk?'}}


def prediction(source,value):
    return {'id':source['id'],'input_sha256':digest(source),'labels':['false','true'],
            'prediction':value,'status':'ok'}


def test_matched_calibration_rejects_input_or_candidate_order_changes():
    source=row('one','true');p=prediction(source,'true')
    validate(source,p)
    with pytest.raises(ValueError,match='candidate order'):
        validate(source,{**p,'labels':['true','false']})
    with pytest.raises(ValueError,match='canonical input'):
        validate(source,{**p,'input_sha256':'different'})


def test_incomplete_jev_coverage_cannot_produce_a_model_advantage():
    rows=[row('one','true'),row('two','false')]
    gemma={'risk':{'n':2,'accuracy':.5,'macro_f1':.5,'positive_recall':1.,'false_positive_rate':1.}}
    result=summarize(rows,{'one':prediction(rows[0],'true')},gemma)['risk']
    assert result['jev_coverage']==.5
    assert result['jev_accuracy_missing_as_wrong']==.5
    assert result['gemma_minus_jev_accuracy'] is None
    complete=summarize(rows,{r['id']:prediction(r,r['target']) for r in rows},gemma)['risk']
    assert complete['jev_positive_recall']==1.
    assert complete['jev_false_positive_rate']==0.
    assert complete['gemma_minus_jev_accuracy']==-.5
