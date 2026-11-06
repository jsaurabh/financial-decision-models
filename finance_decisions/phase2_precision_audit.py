"""Post-test diagnostic: precision/backend stability, never checkpoint selection."""
import gc,json,random
from pathlib import Path
import numpy as np
import torch
from scipy.special import softmax
from huggingface_hub import snapshot_download
from .model import DecisionModel
from .schema import options
from .train import read,infer,shuffled_options,upload


def main():
    torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False
    native_cudnn=torch.backends.cuda.cudnn_sdp_enabled()
    dev=read('data/private/phase2/development.jsonl');rng=random.Random(812)
    # Fixed development sample plus already inspected failures, explicitly post hoc.
    selected=[]
    for d in sorted({r['domain'] for r in dev}):selected+=rng.sample([r for r in dev if r['domain']==d],3)
    flagged={'phase2-test-task5-17-1','phase2-test-task2-22-0','phase2-lex_glue-ledgar-test-241'}
    selected += [r for r in read('data/private/phase2/test.jsonl') if r['id'] in flagged]
    changed=shuffled_options(selected,random.Random(42));output={}
    for arm in ['causal','joint']:
        repo=f'jsaurabh/gemma4-finance-phase2-{arm}'
        artifact=snapshot_download(repo,allow_patterns=['decision_config.json','backbone/**','readout.safetensors','tokenizer/**'])
        model=DecisionModel.load(artifact,device='cuda');output[arm]={}
        for precision in ['bf16','fp32']:
            if precision=='fp32':model.float()
            for backend in ['native','math']:
                torch.backends.cuda.enable_flash_sdp(backend=='native')
                torch.backends.cuda.enable_mem_efficient_sdp(backend=='native')
                torch.backends.cuda.enable_cudnn_sdp(native_cudnn if backend=='native' else False)
                torch.backends.cuda.enable_math_sdp(True)
                a=infer(model,selected,batch_size=1);b=infer(model,changed,batch_size=1)
                diffs=[];flips=[];correct=[]
                for row,other,x,y in zip(selected,changed,a,b):
                    labels=options(row['question'])[0];other_labels=options(other['question'])[0]
                    p=softmax(np.asarray(x[:len(labels)])/model.temperature);q=softmax(np.asarray(y[:len(labels)])/model.temperature)
                    mapped=dict(zip(other_labels,q));delta=max(abs(v-mapped[k]) for k,v in zip(labels,p));diffs.append(delta)
                    pred=labels[int(p.argmax())];other_pred=other_labels[int(q.argmax())]
                    if pred!=other_pred:flips.append({'id':row['id'],'prediction':pred,'reordered_prediction':other_pred,'max_probability_change':float(delta)})
                    correct.append(pred==row['target'])
                result={'n':len(selected),'flips':flips,'prediction_agreement':1-len(flips)/len(selected),
                    'max_probability_change':float(max(diffs)),'sample_accuracy_diagnostic_only':float(np.mean(correct))}
                output[arm][f'{precision}_{backend}']=result
                print(json.dumps({'arm':arm,'precision':precision,'backend':backend,'flips':len(flips),'max_probability_change':max(diffs)}),flush=True)
        del model;gc.collect();torch.cuda.empty_cache()
    out=Path('results/phase2/precision-audit');out.mkdir(parents=True,exist_ok=True)
    result={'scope':'Post-test diagnostic, not an independent quality estimate or a release selection.',
        'inputs':[{'id':r['id'],'split':r['split']} for r in selected],
        'torch':torch.__version__,'tf32':False,'results':output,
        'note':'BF16 base weights are upcast for FP32 computation; adapter/readout FP32 values are retained. No retraining or release modification.'}
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    upload(out,'jsaurabh/gemma4-finance-phase2-causal','phase2-precision-audit')

if __name__=='__main__':main()
