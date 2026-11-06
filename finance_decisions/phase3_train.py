"""Bounded source-supervised Gemma pilot; development/calibration only."""
import argparse
import hashlib
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import HfApi, snapshot_download
from peft import LoraConfig, get_peft_model
from scipy.special import softmax
from sklearn.metrics import f1_score

from .candidate_model import CandidateDecisionModel
from .metrics import fit_temperature
from .model import BASE, REVISION, DecisionModel
from .schema import options
from .train import shuffled_options

SEED = 20261112


def read(path):
    with Path(path).open() as f:
        return [json.loads(line) for line in f if line.strip()]


def keys(row):
    return options(row['question'], max_choices=11)[0]


def targets(rows):
    return [keys(r).index(r['target']) for r in rows]


def mark(status, **kw):
    obj = {'status':status,'time':time.time(),**kw}
    print(json.dumps(obj),flush=True)
    path = Path('/content/phase3-status.json')
    if path.parent.exists():
        temp = path.with_suffix('.tmp'); temp.write_text(json.dumps(obj)); temp.replace(path)


def upload(out, repo):
    api = HfApi(); assert api.model_info(repo).private
    for attempt in range(4):
        try:
            api.upload_folder(repo_id=repo, folder_path=str(out), path_in_repo='run', commit_message='Phase-3 training checkpoint')
            return
        except Exception:
            if attempt==3:
                raise
            mark('checkpoint_retry',attempt=attempt+1); time.sleep(15*(attempt+1))


@torch.inference_mode()
def infer(model, rows):
    model.eval(); logits = []
    for row in rows:
        logits.append(model(model.prepare([row]),[len(keys(row))])[0].float().cpu().tolist())
    return np.asarray(logits)


def summarize(rows, logits, temperature=1.):
    probabilities = softmax(logits/temperature,axis=-1)
    domains = defaultdict(list)
    for i,row in enumerate(rows):
        domains[row['domain']].append(i)
    result = {}
    for domain, indices in domains.items():
        y = [rows[i]['target'] for i in indices]
        pred = [keys(rows[i])[int(probabilities[i].argmax())] for i in indices]
        nll = [-math.log(max(probabilities[i,keys(rows[i]).index(rows[i]['target'])],1e-12)) for i in indices]
        normalized_nll = [v/math.log(len(keys(rows[i]))) for i,v in zip(indices,nll,strict=True)]
        entry = {'n':len(indices),'accuracy':sum(a==b for a,b in zip(y,pred))/len(y),
                 'macro_f1':float(f1_score(y,pred,average='macro',zero_division=0)),
                 'nll':float(np.mean(nll)),'normalized_nll':float(np.mean(normalized_nll))}
        if all(rows[i]['question']['type']=='noul' for i in indices):
            entry.update(positive_recall=sum(a=='true' and b=='true' for a,b in zip(y,pred))/max(1,y.count('true')),
                         false_positive_rate=sum(a=='false' and b=='true' for a,b in zip(y,pred))/max(1,y.count('false')))
        result[domain] = entry
    return {'domains':result,'selection_score':float(np.mean([m['normalized_nll'] for m in result.values()])),
            'selection':'Equal-domain mean NLL/log(candidate count); development only, research mixture not deployment weights.'}


def shuffled_row(row, position):
    return shuffled_options([row],random.Random(SEED+position))[0]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',default='runs/phase3-pilot')
    parser.add_argument('--hf-repo',default='jsaurabh/gemma4-finance-phase3-pilot')
    parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    torch.set_num_threads(8);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED);random.seed(SEED)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False);torch.backends.cuda.enable_math_sdp(True)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    manifest=json.loads(Path('data/phase3-manifest.json').read_text());assert manifest['status']=='sealed_before_training'
    splits={}
    for split, entry in manifest['splits'].items():
        path=Path('data/private/phase3')/(split+'.jsonl')
        assert hashlib.sha256(path.read_bytes()).hexdigest()==entry['sha256']
        splits[split]=read(path)
    assert set(splits)=={'train','development','calibration'}  # No final test is loaded.
    train,dev,cal=(splits[s] for s in ['train','development','calibration'])
    family_sets=[{r['family'] for r in splits[s]} for s in splits]
    assert all(a.isdisjoint(b) for i,a in enumerate(family_sets) for b in family_sets[i+1:])
    api=HfApi();api.create_repo(args.hf_repo,private=True,exist_ok=True);assert api.model_info(args.hf_repo).private
    if args.resume and not (out/'checkpoint/training_state.pt').exists():
        files=api.list_repo_files(args.hf_repo)
        if 'run/checkpoint/training_state.pt' in files:
            import shutil
            cached=Path(snapshot_download(args.hf_repo,allow_patterns=['run/**']))/'run'
            shutil.copytree(cached,out,dirs_exist_ok=True)
    state_path=out/'checkpoint/training_state.pt'
    resuming=state_path.exists()
    mark('loading_model',resume=resuming)
    if resuming:
        model=DecisionModel.load(out/'checkpoint',device='cuda')
    else:
        original=DecisionModel.pretrained_gemma()
        original.backbone.requires_grad_(False)
        backbone=get_peft_model(original.backbone,LoraConfig(r=16,lora_alpha=32,lora_dropout=.05,
            target_modules=['q_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],bias='none'))
        model=CandidateDecisionModel(backbone,original.tokenizer,max_length=8192,attention_mode='independent-causal')
        del original
        model=model.to('cuda')
    for name,param in model.backbone.named_parameters():
        param.requires_grad_('lora_' in name)
    model.max_choices=11;model.max_chars=1000000;model.max_length=8192;model.temperature=1.
    model.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    order=list(range(len(train)));random.Random(SEED).shuffle(order)
    stream=hashlib.sha256()
    for position,index in enumerate(order):
        row=shuffled_row(train[index],position)
        stream.update(json.dumps({'state':row['state'],'question':row['question'],'target':row['target']},sort_keys=True,ensure_ascii=False).encode())
    config={'seed':SEED,'initialization':'fresh official instruction base + fresh scalar head, rank16 LoRA',
        'base_model':BASE,'base_revision':REVISION,'attention':'independent-causal, strict math SDPA, TF32 disabled',
        'max_length':8192,'max_choices':11,'micro_batch':1,'effective_batch':8,'epochs':1,
        'training_stream_sha256':stream.hexdigest(),'manifest_sha256':hashlib.sha256(Path('data/phase3-manifest.json').read_bytes()).hexdigest(),
        'torch':torch.__version__,'gpu':torch.cuda.get_device_name(),'training_code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'steps':math.ceil(len(train)/8),'final_benchmark_evaluation':False}
    config_path=out/'run_config.json'
    if config_path.exists():
        assert json.loads(config_path.read_text())==config,'Resume configuration differs'
    config_path.write_text(json.dumps(config,indent=2)+'\n')
    # Longest-example backward preflight, without changing parameters or RNG.
    rng_cpu=torch.get_rng_state();rng_cuda=torch.cuda.get_rng_state_all()
    longest=max(train,key=lambda r:r['packed_tokens']);model.train()
    assert model.token_length(longest)==longest['packed_tokens']
    logits=model(model.prepare([longest]),[len(keys(longest))])
    loss=F.cross_entropy(logits,torch.tensor(targets([longest]),device='cuda'));loss.backward()
    assert model.readout.weight.grad is not None and torch.isfinite(model.readout.weight.grad).all()
    model.zero_grad(set_to_none=True);del logits,loss;torch.cuda.empty_cache()
    torch.set_rng_state(rng_cpu);torch.cuda.set_rng_state_all(rng_cuda)
    mark('preflight_passed',longest_packed_tokens=longest['packed_tokens'],peak_cuda_gib=torch.cuda.max_memory_allocated()/2**30)
    optimizer=torch.optim.AdamW([{'params':[p for p in model.backbone.parameters() if p.requires_grad],'lr':1e-4},
                               {'params':model.readout.parameters(),'lr':3e-4}],weight_decay=.01)
    start=0;tokens=0;best=math.inf;selected=0;logs=[]
    if resuming:
        saved=torch.load(state_path,map_location='cpu',weights_only=True)
        optimizer.load_state_dict(saved['optimizer']);start=saved['step'];tokens=saved['tokens'];best=saved['best'];selected=saved['selected']
        torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state_all(saved['cuda_rng']);random.setstate(saved['python_rng'])
        if (out/'training.jsonl').exists():logs=read(out/'training.jsonl')
    else:
        initial=summarize(dev,infer(model,dev));best=initial['selection_score']
        (out/'initial-development.json').write_text(json.dumps(initial,indent=2)+'\n');model.save(out/'selected')
    wall=time.monotonic()
    for step in range(start+1,config['steps']+1):
        model.train();optimizer.zero_grad(set_to_none=True)
        rate=min(1.,step/50)*(.1+.9*.5*(1+math.cos(math.pi*step/config['steps'])))
        optimizer.param_groups[0]['lr']=1e-4*rate;optimizer.param_groups[1]['lr']=3e-4*rate
        positions=list(range((step-1)*8,min(step*8,len(train))));losses=[]
        for position in positions:
            row=shuffled_row(train[order[position]],position)
            logits=model(model.prepare([row]),[len(keys(row))])
            loss=F.cross_entropy(logits,torch.tensor(targets([row]),device='cuda'))
            assert torch.isfinite(loss),'Nonfinite training loss'
            (loss/len(positions)).backward();losses.append(loss.item());tokens+=row['packed_tokens']
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
        entry={'step':step,'loss':sum(losses)/len(losses),'tokens':tokens,'session_wall_s':time.monotonic()-wall};logs.append(entry)
        if step==1 or step%10==0:mark('training',step=step,total_steps=config['steps'],loss=entry['loss'],tokens=tokens)
        if step%100==0 or step==config['steps']:
            score=summarize(dev,infer(model,dev))
            (out/f'development-step-{step}.json').write_text(json.dumps(score,indent=2)+'\n')
            if score['selection_score']<best:
                best=score['selection_score'];selected=step;model.save(out/'selected')
            model.save(out/'checkpoint')
            torch.save({'step':step,'tokens':tokens,'best':best,'selected':selected,'optimizer':optimizer.state_dict(),
                'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'python_rng':random.getstate()},state_path)
            (out/'training.jsonl').write_text(''.join(json.dumps(v)+'\n' for v in logs))
            mark('checkpointing',step=step,total_steps=config['steps'],development_score=score['selection_score'],selected_step=selected)
            upload(out,args.hf_repo)
    del model,optimizer;torch.cuda.empty_cache()
    model=DecisionModel.load(out/'selected',device='cuda');model.max_choices=11;model.max_chars=1000000
    cal_logits=infer(model,cal);model.temperature=fit_temperature(cal_logits,targets(cal))
    model.save(out/'release')
    summary={'status':'training_complete','selected_step':selected,'development_selection_score':best,
             'calibration_temperature':model.temperature,'calibration':summarize(cal,cal_logits,model.temperature),
             'tokens':tokens,'peak_cuda_gib':torch.cuda.max_memory_allocated()/2**30,'final_benchmark_evaluation':False,
             'scope':'Internal source-supervised research pilot; no final evaluation or production deployment.'}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (out/'release/README.md').write_text('# Phase-3 financial decisions pilot\n\nFresh official Gemma base, rank-16 LoRA and scalar candidate head. Inherits base and source terms. Internal research only; no production validation or final benchmark claim. Saved envelope: 11 candidates, 8192 packed tokens.\n')
    upload(out,args.hf_repo);mark('complete',repo=args.hf_repo,selected_step=selected)


if __name__=='__main__':
    main()
