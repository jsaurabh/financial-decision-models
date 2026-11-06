import pytest
from finance_decisions.evaluate_jev import digest
from finance_decisions.final_eval import valid

def test_canonical_hash_alone_does_not_hide_changed_candidate_order():
    r={'id':'x','state':'request','question':{'type':'choice','instructions':'choose','criteria':{'B':'b','A':'a'}}}
    p={'id':'x','input_sha256':digest(r),'labels':['A','B'],'status':'ok','prediction':'A','probabilities':[.5,.5]}
    with pytest.raises(ValueError,match='order'):valid(r,p)

def test_reject_nonfinite_or_malformed_probability():
    r={'id':'x','state':'request','question':{'type':'noul','instructions':'predicate'}}
    p={'id':'x','input_sha256':digest(r),'labels':['false','true'],'status':'ok','prediction':'true','probabilities':[float('nan'),1.]}
    with pytest.raises(ValueError,match='prediction'):valid(r,p)
