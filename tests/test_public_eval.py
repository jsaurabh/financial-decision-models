import copy,json
import pytest,torch
from finance_decisions.schema import options,render
from finance_decisions.public_metrics import correct,score
from test_phase2 import tiny


def test_extended_candidates_are_explicit_and_do_not_relax_api_defaults():
    row={'id':'x','state':'Evidence','question':{'type':'choice','instructions':'Choose.','criteria':{str(i):f'candidate {i}' for i in range(11)}}}
    with pytest.raises(ValueError):options(row['question'])
    assert len(options(row['question'],max_choices=11)[0])==11
    model=tiny('independent-causal');model.max_choices=11
    output=model(model.prepare([row]),[11]);assert output.shape==(1,11)
    changed=copy.deepcopy(row);changed['question']['criteria']=dict(reversed(list(row['question']['criteria'].items())))
    torch.testing.assert_close(output,model(model.prepare([changed]),[11]).flip(-1),atol=1e-5,rtol=1e-5)
    long={'type':'noul','instructions':'Predicate?'}
    with pytest.raises(ValueError):render('a'*13000,long)
    assert render('a'*13000,long,max_chars=20000)


def test_public_metrics_keep_missing_cases_and_router_ties():
    row={'id':'one','suite':'routerbench_0shot_quality','family':'f','target':'a',
         'metadata':{'accepted':['a','b'],'cost_penalty':0,'outcomes':{'a':{'quality':1.,'cost_usd':.1},'b':{'quality':1.,'cost_usd':.01}}}}
    missing=dict(row,id='two')
    rec={'id':'one','status':'ok','labels':['a','b'],'probabilities':[.2,.8],'prediction':'b'}
    out=score([row,missing],[rec])[row['suite']]
    assert correct(row,'b') and out['coverage']==.5 and out['accuracy_missing_as_wrong']==.5
    assert out['routing_valid_only']['oracle_regret']==0
    assert out['set_aware_nll_valid_only']==pytest.approx(0)


def test_ts_controversial_label_has_explicit_strict_and_loose_meaning():
    rows=[{'id':str(i),'suite':'tsbench_test','family':str(i),'target':target,'metadata':{}} for i,target in enumerate(['safe','controversial','unsafe'])]
    records=[{'id':r['id'],'status':'ok','prediction':'safe','labels':['safe','controversial','unsafe'],'probabilities':[.8,.1,.1]} for r in rows]
    m=score(rows,records)['tsbench_test']
    assert m['strict_binary_valid_only']['unsafe_miss_rate']==1.
    assert m['loose_binary_valid_only']['unsafe_miss_rate']==1.
    assert m['strict_binary_valid_only']['safe_false_block_rate']==0.


def test_jsonl_retains_unicode_line_separators_inside_state(tmp_path):
    from finance_decisions.public_eval import read
    path=tmp_path/'rows.jsonl';row={'state':'First\u2028second\u2029third','question':{}}
    path.write_text(json.dumps(row,ensure_ascii=False)+'\n'+json.dumps(row,ensure_ascii=False)+'\n')
    assert read(path)==[row,row]


def test_public_pairing_rejects_candidate_order_changes():
    from finance_decisions.public_report import paired
    from finance_decisions.evaluate_jev import digest
    row={'id':'one','suite':'s','family':'f','state':'state','question':{'type':'choice','instructions':'Choose','criteria':{'a':'first','b':'second'}},'target':'a','metadata':{}}
    rec={'id':'one','status':'ok','prediction':'a','input_sha256':digest(row),'labels':['a','b']}
    changed=dict(rec,labels=['b','a'])
    with pytest.raises(ValueError,match='mismatch'):paired([row],{'one':rec},{'one':changed})
    assert paired([row],{'one':rec},{'one':rec})['joint_minus_causal_accuracy']==0
