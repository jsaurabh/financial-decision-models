from copy import deepcopy

import pytest

from finance_decisions.recipe_audit_data import CAPS, choose, controls
from finance_decisions.recipe_audit_train import memorized
from finance_decisions.phase4_metrics import accepted, labels


def rows_for_controls():
    result=[]
    for task in CAPS['train']:
        for index in range(16):
            criteria={'a':'supported conclusion','b':'unsupported conclusion'}
            metadata={}
            target='a' if index%2==0 else 'b'
            state={}
            if task=='financial_verification':
                criteria={'false':'Numerically incorrect','true':'Numerically correct'}
                target='true' if index%2==0 else 'false'
                metadata={'pair':str(index//2),'accepted':[target]}
            if task.startswith('routing_'):
                state={'training_model_profiles':{k:{'mean_quality':float(k==target),
                        'mean_cost_usd':0.} for k in criteria}}
                metadata={'outcomes':{k:{'quality':float(k==target),'cost_usd':0.} for k in criteria},
                          'accepted':[target],'cost_penalty':0.}
            result.append({'id':f'{task}/{index}','task':task,'family':str(index),
                'domain':'model_routing' if task.startswith('routing_') else 'financial',
                'target':target,'state':state,'source':{},'metadata':metadata,
                'question':{'type':'choice','instructions':'Choose the supported result',
                            'criteria':criteria}})
    return result


def test_controls_preserve_references_ties_and_routing_bindings():
    rows=rows_for_controls();before=deepcopy(rows);byid={r['id']:r for r in rows}
    generated=controls(rows)
    assert rows==before and len(generated)==304
    for row in generated:
        control=row['metadata']['control'];parent=byid[control['parent_id']]
        assert row['target'] in labels(row)
        assert control['to_parent_label'][row['target']]==parent['target']
        assert {control['to_parent_label'][key] for key in accepted(row)}==set(accepted(parent))
        if row['task'].startswith('routing_'):
            for key,value in row['metadata']['outcomes'].items():
                assert value==parent['metadata']['outcomes'][control['to_parent_label'][key]]
            assert set(row['state']['training_model_profiles'])==set(labels(row))


def test_financial_sampling_keeps_complete_pairs_and_memorization_requires_real_fit():
    rows=[r for r in rows_for_controls() if r['task']=='financial_verification']
    sample=choose(rows,{'financial_verification':8})
    pairs={r['metadata']['pair'] for r in sample}
    assert len(sample)==8 and len(pairs)==4
    for pair in pairs:
        assert {r['target'] for r in sample if r['metadata']['pair']==pair}=={'false','true'}
    score={'accuracy':1.,'financial_pairs':{'both_correct_fraction':1.}}
    assert memorized(score)
    assert not memorized(score|{'financial_pairs':{'both_correct_fraction':.5}})
    assert not memorized(score|{'native_answer_format':{'unconstrained_top1_allowed_rate':.5}})
    with pytest.raises(ValueError):choose(rows[:1],{'financial_verification':4})


def test_partial_report_keeps_failed_capabilities_visible_without_declaring_winner(tmp_path, monkeypatch):
    import hashlib
    import json
    from finance_decisions.recipe_audit_report import render
    monkeypatch.chdir(tmp_path)
    (tmp_path/'data').mkdir()
    content=b'{"status":"sealed_before_fitting"}\n'
    (tmp_path/'data/recipe-audit-manifest.json').write_bytes(content)
    summary={'manifest_sha256':hashlib.sha256(content).hexdigest(),'selected_step':128,
        'metrics':{'selection_score':.6,
            'financial_pairs':{'roc_auc_valid':.5,'both_correct_fraction':0.},
            'tasks':{'routing_costaware':{'routing_valid_only':{'utility':.4},
                'training_profile_router_valid_only':{'utility':.7}},
                'tool_action_safety':{'strict_safety_valid_only':{
                    'unsafe_or_review_recall':.95,'safe_false_block_rate':.2}}}},
        'acceptance_diagnostics':{'financial_auc_at_least_0_7':False},
        'controls':{kind:{'semantic_prediction_consistency':.5} for kind in
                    ['permutation','aliases','predicate_reversal']}}
    report,text=render({'pilot_code_plain_lr2e-5':summary},{'expected_arms':9,'status':'incomplete'},
                       {'stop_returncode':0},{})
    assert report['status']=='partial' and not report['deployment_winner_selected']
    assert not report['summaries']['pilot_code_plain_lr2e-5']['acceptance_diagnostics']['financial_auc_at_least_0_7']
    assert '0.400 | 0.700' in text and '20.0%' in text
    assert '| pilot_native_sft_lr1e-4 |' not in text
