import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from torch import nn
from transformers import AutoModel, AutoTokenizer, Gemma4ForConditionalGeneration, Gemma4TextModel

from .schema import MAX_CHOICES, render, options

BASE = 'google/gemma-4-E2B-it'
REVISION = '4adf15748de9e1a0097b6274cee9c15836c41c37'


class DecisionModel(nn.Module):
    def __init__(self, backbone, tokenizer, max_length=384, encoder=False):
        super().__init__()
        self.backbone, self.tokenizer = backbone, tokenizer
        self.max_length, self.encoder = max_length, encoder
        self.readout = nn.Linear(backbone.config.hidden_size, MAX_CHOICES, bias=False).float()
        nn.init.normal_(self.readout.weight, std=0.01)
        self.temperature = 1.0
        self.tokenizer.padding_side = 'right'
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    @classmethod
    def pretrained_gemma(cls, dtype=torch.bfloat16):
        # Load the official architecture so checkpoint prefixes are handled exactly.
        full = Gemma4ForConditionalGeneration.from_pretrained(
            BASE, revision=REVISION, dtype=dtype, attn_implementation='sdpa')
        model = cls(full.model.language_model, AutoTokenizer.from_pretrained(BASE, revision=REVISION))
        del full
        return model

    def prepare(self, rows, device=None):
        texts = [render(r['state'], r['question'])[0] for r in rows]
        batch = self.tokenizer(texts, padding=True, truncation=False, return_tensors='pt', return_token_type_ids=False)
        if batch['input_ids'].shape[1] > self.max_length:
            raise ValueError(f'Input exceeds {self.max_length} tokens; no silent truncation is permitted')
        return {k: v.to(device or next(self.parameters()).device) for k, v in batch.items()}

    def token_length(self, row):
        return len(self.tokenizer(render(row['state'], row['question'])[0])['input_ids'])

    def forward(self, batch, counts):
        kwargs = {} if self.encoder else {'use_cache': False}
        hidden = self.backbone(**batch, **kwargs).last_hidden_state
        if self.encoder:
            pooled = hidden[:, 0]
        else:
            last = batch['attention_mask'].sum(-1) - 1
            pooled = hidden[torch.arange(hidden.shape[0], device=hidden.device), last]
        logits = self.readout(pooled.float())
        count = torch.as_tensor(counts, device=logits.device)
        return logits.masked_fill(torch.arange(MAX_CHOICES, device=logits.device)[None, :] >= count[:, None], -1e4)

    def save(self, path, merge=False):
        path = Path(path); path.mkdir(parents=True, exist_ok=True)
        backbone = self.backbone.merge_and_unload() if merge else self.backbone
        backbone.save_pretrained(path / 'backbone', safe_serialization=True)
        self.tokenizer.save_pretrained(path / 'tokenizer')
        save_file({'weight': self.readout.weight.detach().cpu().contiguous()}, path / 'readout.safetensors')
        config = {'architecture': 'single-pass-last-token-readout', 'base_model': BASE, 'base_revision': REVISION,
                  'max_length': self.max_length, 'max_choices': MAX_CHOICES, 'encoder': self.encoder,
                  'temperature': self.temperature, 'merged': merge,
                  'confidence_note': 'temperature-scaled conditional probabilities; review threshold requires in-domain validation'}
        (path / 'decision_config.json').write_text(json.dumps(config, indent=2))

    @classmethod
    def load(cls, path, device='cpu', dtype=torch.bfloat16):
        path = Path(path)
        cfg = json.loads((path / 'decision_config.json').read_text())
        if cfg['encoder']:
            backbone = AutoModel.from_pretrained(path / 'backbone', dtype=dtype)
        elif cfg['merged']:
            backbone = Gemma4TextModel.from_pretrained(path / 'backbone', dtype=dtype, attn_implementation='sdpa')
        else:
            from peft import PeftModel
            backbone = PeftModel.from_pretrained(cls.pretrained_gemma(dtype=dtype).backbone, path / 'backbone')
        constructor = cls
        if cfg['architecture'] == 'packed-independent-candidate-readout':
            from .candidate_model import CandidateDecisionModel
            constructor = CandidateDecisionModel
        model = constructor(backbone, AutoTokenizer.from_pretrained(path / 'tokenizer'), cfg['max_length'], cfg['encoder'])
        if hasattr(model, 'attention_mode'):
            model.attention_mode = cfg.get('attention_mode', 'independent-causal')
            if model.attention_mode not in ('independent-causal', 'joint-bidirectional'):
                raise ValueError('Unsupported saved attention mode')
            model.max_choices = cfg.get('max_choices', MAX_CHOICES)
            model.max_chars = cfg.get('max_chars', 12000)
            if type(model.max_choices) is not int or not 2 <= model.max_choices <= 255 or type(model.max_chars) is not int or model.max_chars <= 0:
                raise ValueError('Invalid saved candidate/context envelope')
        model.readout.load_state_dict(load_file(str(path / 'readout.safetensors')))
        model.temperature = cfg['temperature']
        return model.to(device).eval()


def counts(rows):
    return [len(options(r['question'])[0]) for r in rows]


def targets(rows):
    return [options(r['question'])[0].index(r['target']) for r in rows]
