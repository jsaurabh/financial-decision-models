from finance_decisions.final_data import audit, build

def test_overlap_removes_whole_native_document_and_keeps_unseen_family():
    known=[{'state':'Prior context with a sufficiently long original clause about auditing bank books and financial records included in this contract.', 'question':{'instructions':'x'}}]
    rows=[{'id':'a','suite':'t','family':'doc1','_core':'a sufficiently long original clause about auditing bank books and financial records'},
          {'id':'b','suite':'t','family':'doc1','_core':'another completely new clause that belongs to the same original contract document'},
          {'id':'c','suite':'t','family':'doc2','_core':'a separate document discussing transfer of securities during a merger process'}]
    kept,info=audit(rows,known)
    assert [r['id'] for r in kept]==['c']
    assert info['overlap_exclusions_by_suite']=={'t':2}

def test_actual_adapters_keep_annotations_out_of_model_state():
    rows,counts,excluded=build()
    assert counts['router_common_requests']==809
    assert counts['deepset_test']==116
    for r in rows:
        if r['suite'].startswith('routerarena'):
            assert set(r['state'])=={'context','question','options'}
            assert set(r['metadata']['outcomes'])==set(r['question']['criteria'])
        if r['suite']=='finqa_verification':
            assert set(r['state'])=={'page_before_table','table','page_after_table','question','proposed_answer'}
        if r['suite']=='injecagent_recall':
            assert set(r['state'])=={'User Instruction','User Tool','Tool Parameters','Tool Response'}
