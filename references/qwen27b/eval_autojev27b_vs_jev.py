"""Minimal pinned Qwen reference loader; no training or API calls.

Adapted from jsaurabh/qwen-rlcd-jev-repro, daa6e77, MIT license.
"""
import json
from pathlib import Path
from peft import PeftModel
from safetensors.torch import load_file
from autojev.model import DecisionModel


def load_model(artifact):
    artifact=Path(artifact);cfg=json.loads((artifact/'decision_config.json').read_text())
    model=DecisionModel(train=False,base_model=cfg['base_model'],revision=cfg['revision'],
                        attention_mode=cfg.get('attention_mode','causal'),pooling=cfg.get('pooling','last'))
    model.backbone=PeftModel.from_pretrained(model.backbone,artifact/'adapter').eval()
    model.readout.load_state_dict(load_file(str(artifact/'readout.safetensors')))
    model.processor=model.processor.from_pretrained(str(artifact/'processor'))
    model.processor.tokenizer.padding_side='left';model.temperature=float(cfg['temperature']);model.eval()
    return model,cfg
