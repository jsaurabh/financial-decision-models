"""Score the frozen phase-2 suite after development selection and calibration."""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.special import softmax

from .evaluate_jev import digest, summarize
from .metrics import fit_temperature
from .model import DecisionModel, targets
from .schema import options
from .train import read, infer, shuffled_options, upload
from .phase2_train import verify_data, INIT_REPO, INIT_REVISION
from huggingface_hub import snapshot_download


def records(rows,logits,temperature):
    result=[]
    for row,logit in zip(rows,logits):
        labels=options(row['question'])[0];p=softmax(np.asarray(logit[:len(labels)])/temperature)
        pred=labels[int(p.argmax())]
        result.append({'id':row['id'],'family':row['family'],'domain':row['domain'],'split':row['split'],
            'input_sha256':digest(row),'labels':labels,'target':row['target'],'prediction':pred,
            'correct':pred==row['target'],'probabilities':p.tolist()})
    return result


def aggregate(items):
    return summarize(items)|{'domains':{d:summarize([r for r in items if r['domain']==d]) for d in sorted({r['domain'] for r in items})}}


def main():
    p=argparse.ArgumentParser();p.add_argument('--artifact');p.add_argument('--out',required=True)
    p.add_argument('--hf-repo');p.add_argument('--v4',action='store_true');p.add_argument('--qwen-source')
    args=p.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(8);splits=verify_data()
    if args.qwen_source:
        sys.path.insert(0,args.qwen_source)
        from eval_autojev27b_vs_jev import load_model
        artifact=snapshot_download('jsaurabh/qwen-decision-27b-noncausal-lora',revision='b2eefb929a883136b3bd56ce70999fdac656ee8a')
        model,cfg=load_model(Path(artifact))
        @torch.inference_mode()
        def predict(rows):
            output=[]
            for start in range(0,len(rows),2):
                output.extend(model(model.prepare(rows[start:start+2],max_length=8192)).float().cpu().tolist())
            return output
        provenance={'model':'Qwen 27B noncausal decision reference','artifact_revision':'b2eefb929a883136b3bd56ce70999fdac656ee8a',
                    'base_model':cfg['base_model'],'base_revision':cfg['revision'],'training_data':'different from Gemma; not an attention ablation','native_temperature':model.temperature}
    else:
        path=args.artifact
        if args.v4:path=str(Path(snapshot_download(INIT_REPO,revision=INIT_REVISION,allow_patterns=['run/selected/**']))/'run/selected')
        model=DecisionModel.load(path,device='cuda');model.max_length=2048
        def predict(rows):return infer(model,rows,batch_size=4)
        provenance={'artifact':path,'attention_mode':model.attention_mode,'native_temperature':model.temperature}
    dev_logits=predict(splits['development']);cal_logits=predict(splits['calibration'])
    temp=fit_temperature(cal_logits,targets(splits['calibration']))
    # Evaluate frozen test only after all model/temperature choices are complete.
    summary={'provenance':provenance,'temperature_fit_split':'calibration','temperature':temp}
    for split,logits in [('development',dev_logits),('calibration',cal_logits),('test',predict(splits['test']))]:
        items=records(splits[split],logits,temp)
        (out/f'{split}-predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in items))
        summary[split]=aggregate(items)
        if split=='development':summary['uncalibrated_development']=aggregate(records(splits[split],logits,1.))
    changed=shuffled_options(splits['test'],random.Random(42));other=records(changed,predict(changed),temp)
    original=read(out/'test-predictions.jsonl')
    summary['test_order_control']={'accuracy':sum(r['correct'] for r in other)/len(other),
        'semantic_prediction_agreement':sum(a['prediction']==b['prediction'] for a,b in zip(original,other))/len(other)}
    (out/'order-predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in other))
    summary['environment']={'torch':torch.__version__,'gpu':torch.cuda.get_device_name()}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    if args.hf_repo:upload(out,args.hf_repo,'phase2-quality/'+out.name)
    print(json.dumps({'status':'evaluation_complete','out':str(out),'test_accuracy':summary['test']['accuracy'],'dev_nll':summary['uncalibrated_development']['nll']}),flush=True)

if __name__=='__main__':main()
