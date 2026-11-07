import json
from pathlib import Path
from types import SimpleNamespace

from finance_decisions import phase4_report as report
from finance_decisions.phase4_eval import meta


def test_gemma_only_report_ignores_cancelled_qwen_receipts_and_summary(tmp_path, monkeypatch):
    rows=[]
    domains={'financial_verification':'financial','banking_intent':'financial',
             'legal_maud':'legal','legal_topic':'legal','tool_action_safety':'tool_safety',
             'prompt_injection':'tool_safety','malicious_request':'tool_safety',
             'routing_quality':'model_routing','routing_costaware':'model_routing'}
    for cohort in ['core','transfer']:
        for task,domain in domains.items():
            for index in range(2):
                criteria={'safe':'Safe action','unsafe':'Unsafe action'} if task=='tool_action_safety' else {'false':'False','true':'True'}
                keys=list(criteria);target=keys[index]
                metadata={'outcomes':{key:{'quality':float(key==target),'cost_usd':.001} for key in keys},'cost_penalty':0} if domain=='model_routing' else {}
                rows.append({'id':f'{cohort}/{task}/{index}','family':f'{cohort}/{index}',
                    'suite':task,'task':task,'domain':domain,'cohort':cohort,'target':target,
                    'state':{},'question':{'type':'choice','instructions':'Choose','criteria':criteria},
                    'metadata':metadata,'source':{}})
    cache=tmp_path/'cache';monkeypatch.setattr(report,'CACHE',cache);monkeypatch.setattr(report,'OUT',tmp_path/'output')
    for role in ['gemma','jev']:
        folder=cache/role;folder.mkdir(parents=True)
        receipts=[meta(row)|{'status':'ok','prediction':row['target'],
                    'probabilities':[float(key==row['target']) for key in row['question']['criteria']],
                    'provider':'jev-1.14.0'} for row in rows]
        (folder/'predictions.jsonl').write_text(''.join(json.dumps(p)+'\n' for p in receipts))
    # Stale cancelled-arm files must never be read or considered for selection.
    (cache/'qwen').mkdir();(cache/'qwen/predictions.jsonl').write_text('INVALID CANCELLED ARM')
    summary=tmp_path/'summary.json';summary.write_text(json.dumps({'validation':{'selection_score':.5}}))
    artifact=tmp_path/'artifact.json';artifact.write_text(json.dumps({'repo':'gemma-only','revision':'pinned'}))
    def download(repo,name,**kwargs):
        assert 'qwen' not in repo and 'qwen' not in name
        return str(artifact if name.endswith('artifact.json') else summary)
    monkeypatch.setattr(report,'hf_hub_download',download)
    monkeypatch.setattr(report,'HfApi',lambda:SimpleNamespace(upload_folder=lambda **kwargs:None))
    manifest={'historical_exposure':'disclosed','transfer_test':{'historical_exposure':'disclosed'},'limitations':[]}
    result=report.publish(rows,manifest,{'gemma':{'stop_returncode':0}})
    assert result['status']=='complete'
    assert result['active_candidates']==['gemma']
    assert result['primary_candidate_selected_by_full_validation']=='gemma'
    assert set(result['receipts'])=={'gemma','jev'}
    assert set(result['core_paired_comparisons'])=={'gemma'}
    text=(report.OUT/'comparison.md').read_text()
    assert '| Gemma accuracy | Jev accuracy |' in text
    assert 'Qwen accuracy' not in text and 'cancelled' in text
