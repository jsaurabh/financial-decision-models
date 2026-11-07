"""Exercise the native conditional wrapper, PEFT updates and saved head together."""
import json

import torch
from torch import nn
from peft import LoraConfig, get_peft_model
from transformers import Gemma4Config, Gemma4TextConfig, Gemma4ForConditionalGeneration

from finance_decisions.code_model import CodeDecisionModel, SPECS


class StubTokenizer:
    pad_token_id = 0
    padding_side = 'left'

    def save_pretrained(self, path):
        path.mkdir(parents=True, exist_ok=True)
        (path / 'tokenizer_config.json').write_text(json.dumps({'stub': True}))


def tiny_conditional(lora=False):
    text = Gemma4TextConfig(vocab_size=128, vocab_size_per_layer_input=128,
        hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=1, head_dim=8, global_head_dim=8,
        hidden_size_per_layer_input=8, num_kv_shared_layers=0,
        layer_types=['sliding_attention', 'full_attention'], sliding_window=16,
        max_position_embeddings=512, use_double_wide_mlp=False,
        final_logit_softcapping=5., rope_parameters={
            'sliding_attention': {'rope_type': 'default', 'rope_theta': 10000.},
            'full_attention': {'rope_type': 'proportional', 'rope_theta': 10000.,
                              'partial_rotary_factor': .5}})
    with torch.random.fork_rng():
        torch.manual_seed(182)
        full = Gemma4ForConditionalGeneration(Gemma4Config(text_config=text))
    readout = nn.Linear(32, 2, bias=False)
    with torch.no_grad():
        readout.weight.copy_(full.lm_head.weight[1:3])
    full.model.requires_grad_(False)
    if lora:
        full.model.language_model = get_peft_model(full.model.language_model,
            LoraConfig(r=4, lora_alpha=8, lora_dropout=0., bias='none',
                       target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj',
                                       'gate_proj', 'up_proj', 'down_proj']))
    model = CodeDecisionModel(full.model, StubTokenizer(), readout, SPECS['gemma'],
                              'gemma', 5., codes=['B', 'C'])
    return full, model


def test_native_conditional_wrapper_matches_lm_rows_with_left_padding():
    native, model = tiny_conditional()
    native.eval(); model.eval()
    inputs = {'input_ids': torch.tensor([[3, 7, 8], [0, 4, 8]]),
              'attention_mask': torch.tensor([[1, 1, 1], [0, 1, 1]])}
    with torch.inference_mode():
        expected = native(**inputs, use_cache=False).logits[:, -1, 1:3]
        actual = model(inputs | {'_counts': torch.tensor([2, 2])})
    assert torch.allclose(actual, expected, atol=1e-6)


def test_peft_training_can_learn_state_and_survive_save_load(tmp_path, monkeypatch):
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        _, model = tiny_conditional(lora=True)
        frozen = {n: p.detach().clone() for n, p in model.backbone.named_parameters()
                  if not p.requires_grad}
        model.backbone.language_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={'use_reentrant': False})
        # The last token is identical. The decision must use the earlier state.
        inputs = {'input_ids': torch.tensor([[3, 7, 8], [4, 7, 8]]),
                  'attention_mask': torch.ones(2, 3, dtype=torch.long),
                  '_counts': torch.tensor([2, 2])}
        target = torch.tensor([0, 1])
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                      lr=.02, weight_decay=0.)
        model.train()
        initial = float(torch.nn.functional.cross_entropy(model(inputs), target).detach())
        for _ in range(100):
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(model(inputs), target)
            loss.backward()
            assert all(torch.isfinite(p.grad).all() for p in model.parameters()
                       if p.grad is not None)
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            expected = model(inputs)
            assert expected.argmax(-1).tolist() == target.tolist()
            assert float(torch.nn.functional.cross_entropy(expected, target)) < initial * .1
        assert any(p.detach().count_nonzero() > 0 for n, p in model.backbone.named_parameters()
                   if 'lora_B' in n)
        for name, parameter in model.backbone.named_parameters():
            if name in frozen:
                assert torch.equal(parameter, frozen[name])
        model.save(tmp_path / 'release')
        monkeypatch.setattr(CodeDecisionModel, 'pretrained', classmethod(
            lambda cls, family, **kwargs: tiny_conditional(lora=False)[1]))
        monkeypatch.setattr('finance_decisions.code_model.AutoTokenizer.from_pretrained',
                            lambda *args, **kwargs: StubTokenizer())
        restored = CodeDecisionModel.load(tmp_path / 'release', device='cpu', dtype=torch.float32)
        with torch.inference_mode():
            assert torch.allclose(restored(inputs), expected, atol=1e-6)
        # Same explicit re-enable used by phase4 when resuming an inference adapter.
        for name, parameter in restored.backbone.named_parameters():
            parameter.requires_grad_('lora_' in name)
        restored.train(); restored(inputs).sum().backward()
        assert any(p.grad is not None for n, p in restored.backbone.named_parameters()
                   if 'lora_B' in n)
    finally:
        torch.set_num_threads(previous_threads)
