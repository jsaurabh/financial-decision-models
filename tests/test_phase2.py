import copy
import json
from types import SimpleNamespace

import pytest
import torch
from transformers import Gemma4TextConfig, Gemma4TextModel
from finance_decisions.candidate_model import CandidateDecisionModel
from finance_decisions.phase2_data import synthetic


def tiny(mode):
    config=Gemma4TextConfig(vocab_size=128,vocab_size_per_layer_input=128,hidden_size=32,
        intermediate_size=64,num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=1,
        head_dim=8,global_head_dim=8,hidden_size_per_layer_input=8,num_kv_shared_layers=0,
        layer_types=['sliding_attention','full_attention'],sliding_window=16,max_position_embeddings=512,
        use_double_wide_mlp=False,rope_parameters={'sliding_attention':{'rope_type':'default','rope_theta':10000.},
        'full_attention':{'rope_type':'proportional','rope_theta':10000.,'partial_rotary_factor':.5}})
    class Tokenizer:
        padding_side='right';pad_token_id=0
        def __call__(self,text,add_special_tokens=True):
            return {'input_ids':([2] if add_special_tokens else [])+[4+sum(w.encode())%120 for w in text.split()]}
    return CandidateDecisionModel(Gemma4TextModel(config),Tokenizer(),512,attention_mode=mode).eval()


def test_only_global_edges_change_and_padding_is_excluded():
    a=tiny('independent-causal');c=tiny('joint-bidirectional');c.load_state_dict(a.state_dict())
    row={'state':'Verified evidence.','question':{'type':'choice','instructions':'Choose.','criteria':{'a':'first conclusion','b':'second conclusion'}}}
    longer=copy.deepcopy(row);longer['state']+=' Added record.'*5
    ba=a.prepare([row,longer]);bc=c.prepare([row,longer])
    for k in ['input_ids','position_ids','candidate_positions']:assert torch.equal(ba[k],bc[k])
    assert torch.equal(ba['attention_mask']['sliding_attention'],bc['attention_mask']['sliding_attention'])
    n=a.token_length(row);prefix,blocks=a.encode(row);start=len(prefix)
    assert not ba['attention_mask']['full_attention'][0,0,0,start]
    assert bc['attention_mask']['full_attention'][0,0,0,start]
    assert not ba['attention_mask']['full_attention'][0,0,start,start+len(blocks[0])]
    assert bc['attention_mask']['full_attention'][0,0,:n,:n].all()
    assert not bc['attention_mask']['full_attention'][0,0,:,n:].any()
    assert not bc['attention_mask']['full_attention'][0,0,n:,:].any()
    for model in [a,c]:
        baseline=model(model.prepare([row]),[2])[0,:2]
        padded=model(model.prepare([row,longer]),[2,2])[0,:2]
        torch.testing.assert_close(baseline,padded,atol=1e-5,rtol=1e-5)
        shuffled=copy.deepcopy(row);shuffled['question']['criteria']=dict(reversed(list(row['question']['criteria'].items())))
        torch.testing.assert_close(baseline,model(model.prepare([shuffled]),[2])[0,:2].flip(0),atol=1e-5,rtol=1e-5)
        torch.nn.functional.cross_entropy(model(model.prepare([row]),[2]),torch.tensor([1])).backward()
        assert model.readout.weight.grad.abs().sum()>0
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_synthetic_oracles_and_group_separation():
    splits=synthetic();families=set();inputs=set()
    for split,rows in splits.items():
        current={r['family'] for r in rows};assert not families&current;families|=current
        current_inputs={json.dumps([r['state'],r['question']],sort_keys=True) for r in rows}
        assert not inputs&current_inputs;inputs|=current_inputs
        for r in rows:
            o=r['source']['oracle'];domain=r['domain']
            if domain=='net_exposure_arithmetic':expected='review' if o['collateral'] is None else 'enhanced' if o['a']+o['b']-o['collateral']>=o['threshold'] else 'standard'
            elif domain=='deadline_arithmetic':
                assert o['working_days']==sum((o['start_weekday']+i)%7<5 for i in range(1,o['elapsed_days']+1))
                expected='review' if o['disputed'] else 'enhanced' if o['working_days']>=o['limit'] else 'standard'
            elif domain=='concentration_arithmetic':expected='review' if o['conflict'] else 'enhanced' if 100*o['issuer']>=o['percent_limit']*o['total'] else 'standard'
            elif domain=='contract_exception':expected='review' if o['unknown'] else 'enhanced' if o['default_requires_consent']!=o['exception_applies'] else 'standard'
            elif domain=='evidence_precedence':expected='review' if o['verified'] is None else 'enhanced' if o['verified']>=o['limit'] else 'standard'
            else:expected={'retrieval':'lookup','analysis':'analysis','source code':'code','binding approval':'human'}[o['requested_deliverable']]
            assert r['target']==expected
