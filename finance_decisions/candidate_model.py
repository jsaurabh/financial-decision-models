"""Pack independent candidate evaluations into one Gemma forward pass.

Candidate blocks share the state/question prefix, use the same scalar readout,
cannot attend to one another, and reset RoPE positions to the prefix boundary.
Reordering candidates therefore preserves each candidate's computation.
"""
import json
from pathlib import Path

import torch
from torch import nn

from .model import DecisionModel
from .schema import MAX_CHOICES, options, render


class CandidateDecisionModel(DecisionModel):
    def __init__(self, backbone, tokenizer, max_length=1536, encoder=False, attention_mode='independent-causal'):
        super().__init__(backbone, tokenizer, max_length, False)
        if attention_mode not in ('independent-causal', 'joint-bidirectional'):
            raise ValueError(f'Unknown candidate attention mode: {attention_mode}')
        self.attention_mode = attention_mode
        self.readout = nn.Linear(backbone.config.hidden_size, 1, bias=False).float()
        nn.init.normal_(self.readout.weight, std=.01)

    def encode(self, row):
        # Validate the same public schema before creating independent blocks.
        render(row['state'], row['question'])
        labels, descriptions = options(row['question'])
        state = row['state'] if isinstance(row['state'],str) else json.dumps(row['state'],sort_keys=True,ensure_ascii=False)
        prefix = self.tokenizer('State:\n' + state + '\nQuestion: ' + row['question']['instructions'] + '\n', add_special_tokens=True)['input_ids']
        blocks = [self.tokenizer(f'Candidate conclusion: {label}: {description}\nCompatibility:', add_special_tokens=False)['input_ids']
                  for label,description in zip(labels,descriptions)]
        return prefix, blocks

    def token_length(self, row):
        prefix, blocks = self.encode(row)
        return len(prefix) + sum(map(len,blocks))

    def prepare(self, rows, device=None):
        encoded = [self.encode(r) for r in rows]
        length = max(len(p)+sum(map(len,b)) for p,b in encoded)
        if length > self.max_length:
            raise ValueError(f'Packed input exceeds {self.max_length} tokens; shorten state or criteria')
        target_device = device or next(self.parameters()).device
        device = torch.device('cpu')
        ids = torch.full((len(rows),length),self.tokenizer.pad_token_id,dtype=torch.long,device=device)
        positions = torch.zeros_like(ids)
        outputs = torch.zeros((len(rows),MAX_CHOICES),dtype=torch.long,device=device)
        full = torch.zeros((len(rows),1,length,length),dtype=torch.bool,device=device)
        local = torch.zeros_like(full)
        window = self.backbone.config.sliding_window
        for i,(prefix,blocks) in enumerate(encoded):
            n = len(prefix); offset = n
            ids[i,:n] = torch.tensor(prefix,device=device); positions[i,:n] = torch.arange(n,device=device)
            delta = torch.arange(n)[:,None] - torch.arange(n)[None,:]
            full[i,0,:n,:n] = delta >= 0
            local[i,0,:n,:n] = (delta >= 0) & (delta < window)
            for j,block in enumerate(blocks):
                m = len(block)
                ids[i,offset:offset+m] = torch.tensor(block,device=device)
                positions[i,offset:offset+m] = torch.arange(n,n+m,device=device)
                full[i,0,offset:offset+m,:n] = True
                delta = torch.arange(m)[:,None] - torch.arange(m)[None,:]
                full[i,0,offset:offset+m,offset:offset+m] = delta >= 0
                prefix_delta = n + torch.arange(m)[:,None] - torch.arange(n)[None,:]
                local[i,0,offset:offset+m,:n] = prefix_delta < window
                local[i,0,offset:offset+m,offset:offset+m] = (delta >= 0) & (delta < window)
                outputs[i,j] = offset+m-1; offset += m
            if self.attention_mode == 'joint-bidirectional':
                # Qwen-style ablation: open global layers only. Native sliding
                # layers, tokens, reset positions and scalar head stay identical.
                full[i,0,:offset,:offset] = True
        return {'input_ids':ids.to(target_device),'position_ids':positions.to(target_device),
                'attention_mask':{'full_attention':full.to(target_device),'sliding_attention':local.to(target_device)},
                'candidate_positions':outputs.to(target_device)}

    def forward(self, batch, counts):
        inputs = {k:v for k,v in batch.items() if k != 'candidate_positions'}
        hidden = self.backbone(**inputs,use_cache=False).last_hidden_state
        pooled = hidden[torch.arange(hidden.shape[0],device=hidden.device)[:,None],batch['candidate_positions']]
        logits = self.readout(pooled.float()).squeeze(-1)
        return logits.masked_fill(torch.arange(MAX_CHOICES,device=logits.device)[None,:] >= torch.as_tensor(counts,device=logits.device)[:,None],-1e4)

    def save(self, path, merge=False):
        super().save(path,merge)
        config = Path(path)/'decision_config.json'; data=json.loads(config.read_text())
        data.update(architecture='packed-independent-candidate-readout',candidate_attention='prefix + own causal block',
                    candidate_position_reset=True,readout='shared scalar',attention_mode=self.attention_mode)
        if self.attention_mode == 'joint-bidirectional':
            data['candidate_attention'] = 'joint bidirectional global layers; independent causal sliding layers'
        config.write_text(json.dumps(data,indent=2))
