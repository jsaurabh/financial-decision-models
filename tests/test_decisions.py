import copy
import random

import pytest
import torch
from peft import LoraConfig, get_peft_model
from transformers import Gemma4TextConfig, Gemma4TextModel

from finance_decisions.model import DecisionModel, targets
from finance_decisions.schema import answer, options, render
from finance_decisions.train import shuffled_options
from finance_decisions.robustness import augment


def test_choice_shuffle_preserves_semantic_target():
    row = {'state': 'evidence', 'question': {'type': 'choice', 'instructions': 'Choose.',
           'criteria': {'yes': 'Approve', 'no': 'Reject', 'review': 'Review'}}, 'target': 'no'}
    changed = shuffled_options([row], random.Random(3))[0]
    assert options(changed['question'])[0][targets([changed])[0]] == 'no'
    assert row['question']['criteria'] == {'yes': 'Approve', 'no': 'Reject', 'review': 'Review'}


def test_typed_answers_and_bounds():
    question = {'type': 'score', 'criteria': ['none', 'some', 'complete']}
    response = answer(question, [.1, .2, .7])
    assert response['value'] == pytest.approx(1.6)
    assert response['needs_review']
    assert answer({'type': 'noul'}, [.1, .9])['value'] is True
    assert options({'type':'noul','criteria':{'false':'not verified','true':'verified'}})[1] == ['not verified','verified']
    with pytest.raises(ValueError):
        options({'type': 'choice', 'criteria': {str(i): 'label' for i in range(9)}})
    with pytest.raises(ValueError):
        render('a' * 13000, {'type': 'noul', 'instructions': 'Predicate'})
    with pytest.raises(ValueError):
        render('short', {'type':'choice','instructions':'Choose','criteria':{'a'*13000:'one','b':'two'}})


def test_native_gemma_padding_mask_and_lora_gradients():
    torch.manual_seed(2)
    config = Gemma4TextConfig(vocab_size=128, vocab_size_per_layer_input=128,
        hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=1, head_dim=8, global_head_dim=8,
        hidden_size_per_layer_input=8, num_kv_shared_layers=0, layer_types=['sliding_attention', 'full_attention'],
        sliding_window=16, max_position_embeddings=128, use_double_wide_mlp=False,
        rope_parameters={'sliding_attention': {'rope_type': 'default', 'rope_theta': 10000.},
                         'full_attention': {'rope_type': 'proportional', 'rope_theta': 10000., 'partial_rotary_factor': .5}})
    class Tokenizer:
        padding_side = 'right'; pad_token_id = 0
    backbone = Gemma4TextModel(config)
    backbone.requires_grad_(False)
    backbone = get_peft_model(backbone, LoraConfig(r=2, lora_alpha=4, target_modules=['q_proj', 'v_proj']))
    model = DecisionModel(backbone, Tokenizer())
    model.eval()
    single = {'input_ids': torch.tensor([[2, 10, 11]]), 'attention_mask': torch.tensor([[1,1,1]])}
    padded = {'input_ids': torch.tensor([[2,10,11,0,0]]), 'attention_mask': torch.tensor([[1,1,1,0,0]])}
    torch.testing.assert_close(model(single, [3]), model(padded, [3]), atol=1e-5, rtol=1e-5)
    logits = model(single, [3]); assert (logits[:, 3:] < -9000).all()
    torch.nn.functional.cross_entropy(logits, torch.tensor([1])).backward()
    assert model.readout.weight.grad.abs().sum() > 0
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for n,p in model.named_parameters() if 'lora_B' in n)


def test_missing_evidence_augmentation_keeps_reference_and_policy():
    row = {'domain': 'payments', 'state': 'Case 3: observed days since posting: unknown. Evidence is current. No override applies.',
           'question': {'type': 'choice', 'instructions': 'Apply policy.', 'criteria': {'review':'Unknown evidence','normal':'Known evidence'}},
           'target': 'review'}
    for seed in range(12):
        changed = augment(row, random.Random(seed))
        assert changed['target'] == 'review'
        assert changed['question'] == row['question']
        assert changed['state'] != row['state']


def test_live_jev_payload_excludes_reference_and_preserves_native_tie_break(monkeypatch):
    from finance_decisions.evaluate_jev import call
    monkeypatch.setenv('TYPESAFE_API_KEY', 'unit-test-placeholder')
    row = {'id':'case','domain':'banking','state':'The reference is kept outside this input.',
           'question':{'type':'choice','instructions':'Choose','criteria':{'a':'first','b':'second'}}, 'target':'a'}
    class Response:
        status_code = 200
        def json(self):
            return {'model':'jev-unit-test','answers':{'decision':{'choice':'b','probabilities':{'a':.5,'b':.5}}},'usage':{'input_tokens':10}}
    def post(url, headers, json, timeout):
        assert set(json) == {'model','state','questions'}
        assert 'target' not in json and 'reference' not in json
        assert json['questions']['decision'] == row['question']
        return Response()
    monkeypatch.setattr('requests.post', post)
    result = call(row, 'jev-latest')
    assert result['prediction'] == 'b'
    assert 'target' not in result


def test_paired_comparison_rejects_different_candidate_order_and_deduplicates():
    from finance_decisions.compare_quality import pair_records
    first = {'id':'one','input_sha256':'same','labels':['a','b'],'target':'a'}
    repeated = {**first,'id':'two'}
    assert len(pair_records([first,repeated],[first,repeated])[0]) == 1
    with pytest.raises(ValueError, match='candidate-order mismatch'):
        pair_records([first],[{**first,'labels':['b','a']}])


def test_packed_candidates_are_order_invariant_and_receive_gradients():
    from finance_decisions.candidate_model import CandidateDecisionModel
    import hashlib
    config = Gemma4TextConfig(vocab_size=128,vocab_size_per_layer_input=128,hidden_size=32,
        intermediate_size=64,num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=1,
        head_dim=8,global_head_dim=8,hidden_size_per_layer_input=8,num_kv_shared_layers=0,
        layer_types=['sliding_attention','full_attention'],sliding_window=16,max_position_embeddings=512,
        use_double_wide_mlp=False,rope_parameters={'sliding_attention':{'rope_type':'default','rope_theta':10000.},
        'full_attention':{'rope_type':'proportional','rope_theta':10000.,'partial_rotary_factor':.5}})
    class Tokenizer:
        padding_side='right';pad_token_id=0
        def __call__(self,text,add_special_tokens=True):
            ids=[4+int(hashlib.sha256(word.encode()).hexdigest()[:4],16)%120 for word in text.split()]
            return {'input_ids':([2] if add_special_tokens else [])+ids}
    model=CandidateDecisionModel(Gemma4TextModel(config),Tokenizer(),256).eval()
    row={'state':'There is complete evidence.','question':{'type':'choice','instructions':'Choose.',
         'criteria':{'yes':'Complete evidence','no':'Missing evidence','later':'Wait for missing evidence'}}}
    other=copy.deepcopy(row);other['question']['criteria']=dict(reversed(list(row['question']['criteria'].items())))
    a=model(model.prepare([row]),[3])[0,:3]
    b=model(model.prepare([other]),[3])[0,:3].flip(0)
    torch.testing.assert_close(a,b,atol=1e-5,rtol=1e-5)
    longer=copy.deepcopy(row);longer['state']+=' Another recorded fact.'*4
    padded=model(model.prepare([row,longer]),[3,3])[0,:3]
    torch.testing.assert_close(a,padded,atol=1e-5,rtol=1e-5)
    torch.nn.functional.cross_entropy(model(model.prepare([row]),[3]),torch.tensor([0])).backward()
    assert model.readout.weight.grad.abs().sum()>0
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
