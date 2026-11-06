"""Publish private model cards and an immutable artifact index after evaluation."""
import hashlib
import json
from pathlib import Path
from huggingface_hub import HfApi,hf_hub_download


def main():
    root=Path('results/phase2');comparison=json.loads((root/'comparison.json').read_text());api=HfApi();index={}
    for arm in ['causal','joint']:
        repo=f'jsaurabh/gemma4-finance-phase2-{arm}'
        assert api.model_info(repo).private
        config=json.loads(Path(hf_hub_download(repo,'decision_config.json')).read_text())
        expected='independent-causal' if arm=='causal' else 'joint-bidirectional'
        assert config['attention_mode']==expected
        final_summary=json.loads((root/arm/'summary.json').read_text())
        # Refit on calibration in the final inference runtime, never test labels.
        config['temperature']=final_summary['temperature']
        api.upload_file(repo_id=repo,path_or_fileobj=(json.dumps(config,indent=2)+'\n').encode(),
            path_in_repo='decision_config.json',commit_message='Preserve final calibration-only temperature')
        s=json.loads((root/arm/'selection.json').read_text());m=comparison[arm]
        card=f'''---
license: apache-2.0
base_model: google/gemma-4-E2B-it
pipeline_tag: text-classification
library_name: transformers
tags:
- financial-services
- decision-model
- lora
---
# Gemma 4 financial decisions: phase 2 {arm}

Private research decision classifier, with a shared scalar candidate scorer and `{expected}` attention. Supports text-only choice/noul/score schemas with 2–8 options, up to 2,048 packed tokens. Model-size and CPU constraints were outside this quality experiment. The evaluation is choice/noul only.

The matched arms start at v4 revision `1307ef28b41b4b295c204f5f60f57f92caa09bb1`, `run/selected`. Training: 1,000 updates, same seed, examples and token budget; selected step {s['selected_step']} by new development NLL. Temperature {config['temperature']:.6f} was fitted only on the separate calibration split. The joint arm opens global attention across all candidate branches while retaining independent causal sliding layers. Serialization, reset positions, LoRA and scalar head match the causal arm.

Test accuracy on 1,011 newly frozen inputs: {m['accuracy']:.2%}; equal-domain accuracy: {m['domain_macro_accuracy']:.2%}. These numbers combine public sampled-candidate banking/contract tasks and unreviewed generated hypothetical-policy cases. They are not production validation. Routing is capability based, public pretraining exposure is unknown, and the two arms share a causal-trained initialization. A single paired seed cannot establish a general architecture advantage.

[Code and results](https://github.com/jsaurabh/financial-decision-models/tree/gemma4-finance/results/phase2), [frozen protocol](https://github.com/jsaurabh/financial-decision-models/blob/gemma4-finance/PHASE2_PROTOCOL.md), [private corpus](https://huggingface.co/datasets/jsaurabh/financial-decision-phase2).

## Load

Install the repository's pinned Gemma requirements. Root artifacts are a LoRA adapter, scalar readout, tokenizer and decision config over the pinned official base. This is a custom classifier, not a text generation endpoint.

```python
from huggingface_hub import snapshot_download
from finance_decisions.model import DecisionModel
path = snapshot_download(
    "{repo}",
    allow_patterns=["decision_config.json", "readout.safetensors", "backbone/**", "tokenizer/**"],
)
model = DecisionModel.load(path, device="cuda")
```

The config restores attention and temperature. See the repository for the typed API. No CPU latency claim is made for this phase.
'''
        api.upload_file(repo_id=repo,path_or_fileobj=card.encode(),path_in_repo='README.md',commit_message='Document frozen phase-two quality results and loader')
        api.upload_folder(repo_id=repo,folder_path=str(root/arm),path_in_repo=f'phase2-quality/{arm}',commit_message='Preserve final quality reports')
        info=api.model_info(repo)
        index[arm]={'repo':repo,'revision':info.sha,'private':info.private,'decision_config':config,
            'test_predictions_sha256':hashlib.sha256((root/arm/'test-predictions.jsonl').read_bytes()).hexdigest()}
    data=api.dataset_info('jsaurabh/financial-decision-phase2')
    index['corpus']={'repo':data.id,'revision':data.sha,'private':data.private,'manifest_sha256':hashlib.sha256(Path('data/phase2-manifest.json').read_bytes()).hexdigest()}
    index['qwen']={'repo':'jsaurabh/qwen-decision-27b-noncausal-lora','revision':'b2eefb929a883136b3bd56ce70999fdac656ee8a','role':'frozen larger reference; different earlier training'}
    index['jev']=json.loads((root/'jev/summary.json').read_text())['providers']
    (root/'artifacts.json').write_text(json.dumps(index,indent=2)+'\n');print('Private artifacts and model cards verified')

if __name__=='__main__':main()
