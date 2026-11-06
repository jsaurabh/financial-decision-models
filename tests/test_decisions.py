import copy
import random

import pytest
import torch
from peft import LoraConfig, get_peft_model
from transformers import Gemma4TextConfig, Gemma4TextModel

from finance_decisions.model import DecisionModel, targets
from finance_decisions.schema import answer, options, render
from finance_decisions.train import shuffled_options


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
    with pytest.raises(ValueError):
        options({'type': 'choice', 'criteria': {str(i): 'label' for i in range(9)}})
    with pytest.raises(ValueError):
        render('a' * 13000, {'type': 'noul', 'instructions': 'Predicate'})


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
