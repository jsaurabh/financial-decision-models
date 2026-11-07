import pytest
from finance_decisions.final_metrics import paired_cluster, holm, system_metrics, compare

def row(i,target='true',family=None,suite='test',domain='oos'):
    return {'id':str(i),'target':target,'family':family or str(i),'suite':suite,'domain':domain,'cohort':'fresh','metadata':{}}

def prediction(value):return {'status':'ok','prediction':value,'labels':['false','true'],'probabilities':[.1,.9] if value=='true' else [.9,.1]}

def test_identical_systems_have_zero_paired_effect():
    rows=[row(i) for i in range(20)]
    result=paired_cluster(rows,[0]*20,reps=100)
    assert result['difference']==0 and result['ci95']==[0,0] and result['p_two_sided']==1

def test_family_dependence_cannot_invent_independent_support():
    rows=[row(i,family='one') for i in range(100)]
    result=paired_cluster(rows,[1]*100,reps=100)
    assert result['difference']==1 and result['families']==1
    assert result['ci95'] is None and result['p_two_sided']==1

def test_missing_predictions_count_wrong_not_removed_from_accuracy():
    rows=[row(0),row(1)]
    metrics=system_metrics(rows,{'0':prediction('true')})
    assert metrics['coverage']==.5 and metrics['accuracy_missing_as_wrong']==.5
    assert metrics['accuracy_valid_only']==1

def test_holm_correction_and_mixture_task_weights():
    assert holm({'a':.01,'b':.04,'c':.03,'d':.8})==pytest.approx({'a':.04,'b':.09,'c':.09,'d':.8})
    rows=[row(0,suite='small')]+[row(i+1,suite='big') for i in range(99)]
    result=paired_cluster(rows,[1]+[-1]*99,reps=100)
    assert result['difference']==0

def test_missing_domain_blocks_broad_claim():
    rows=[row(i) for i in range(12)]
    g={str(i):prediction('true') for i in range(12)}
    j={str(i):prediction('false') for i in range(12)}
    report=compare(rows,g,j,reps=100)
    assert not report['broad_classification_superiority_supported']
    assert report['confirmatory_domains']['legal']['status']=='no_eligible_cases'
