import pytest
from finance_decisions.tracesafe_data import RULES
from finance_decisions.tracesafe_metrics import primary

def fixture():
    rows=[]
    for i,category in enumerate(['benign',*RULES]):
        rows.append({'id':str(i),'family':str(i),'suite':'binary/'+category,'target':'false' if i==0 else 'true',
            'metadata':{'category':category,'task':'binary'}})
    return rows

def predictions(rows,value):
    return {r['id']:{'status':'ok','prediction':value,'labels':['false','true'],'probabilities':[.1,.9] if value=='true' else [.9,.1]} for r in rows}

def test_always_unsafe_is_only_chance_on_balanced_primary():
    rows=fixture();p=predictions(rows,'true');m=primary(rows,p,p,reps=100)
    assert m['always_unsafe_accuracy']==pytest.approx(12/13)
    assert m['systems']['gemma']['balanced_score']==.5
    assert m['systems']['gemma']['benign_false_block_rate_valid_only']==1
    assert not m['supports_superiority']

def test_unsupported_benign_is_not_counted_as_safe_acceptance():
    rows=fixture();g=predictions(rows,'true');j=predictions(rows,'true')
    g['0']={'status':'unsupported','error_type':'over_context_limit'}
    m=primary(rows,g,j,reps=100)
    assert m['systems']['gemma']['balanced_score']==.5
    assert m['systems']['gemma']['benign_specificity_missing_as_wrong']==0
    assert m['systems']['gemma']['benign_false_block_rate_valid_only'] is None
    assert not m['supports_superiority']

def test_missing_category_blocks_full_scope_claim():
    rows=fixture()[1:];p=predictions(rows,'true')
    assert primary(rows,p,p,reps=100)['status']=='incomplete_category_support'
