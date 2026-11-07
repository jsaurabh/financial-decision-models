"""Matched continuation: one init/data stream, global mask is the sole ablation."""
import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download

from .metrics import metrics, fit_temperature
from .model import DecisionModel, counts, targets
from .train import read, infer, shuffled_options, upload

INIT_REPO = 'jsaurabh/gemma4-e2b-finance-decisions-v4'
INIT_REVISION = '1307ef28b41b4b295c204f5f60f57f92caa09bb1'
SEED = 20261108


def verify_data():
    manifest=json.loads(Path('data/phase2-manifest.json').read_text())
    splits={}
    for split,entry in manifest['splits'].items():
        path=Path(f'data/private/phase2/{split}.jsonl')
        assert hashlib.sha256(path.read_bytes()).hexdigest()==entry['sha256']
        splits[split]=read(path)
    return splits


def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['independent-causal','joint-bidirectional'],required=True)
    p.add_argument('--steps',type=int,default=1000);p.add_argument('--max-length',type=int,default=2048)
    p.add_argument('--hf-repo',required=True);p.add_argument('--out',required=True)
    args=p.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(8);torch.manual_seed(SEED);random.seed(SEED)
    torch.backends.cuda.matmul.allow_tf32=True
    init=Path(snapshot_download(INIT_REPO,revision=INIT_REVISION,allow_patterns=['run/selected/**']))/'run/selected'
    model=DecisionModel.load(init,device='cuda');model.attention_mode=args.mode;model.max_length=args.max_length
    for n,param in model.backbone.named_parameters():param.requires_grad_('lora_' in n)
    splits=verify_data()
    # Eligibility is frozen in the manifest before allocation, not selected by score.
    for split,rows in splits.items():
        assert all(model.token_length(r)<=args.max_length for r in rows), split
    dev=splits['development'];train=splits['train']
    rng=random.Random(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
    opt=torch.optim.AdamW([{'params':[p for p in model.backbone.parameters() if p.requires_grad],'lr':3e-5},
                          {'params':model.readout.parameters(),'lr':1e-4}],weight_decay=.01)
    cfg=vars(args)|{'seed':SEED,'initial_repo':INIT_REPO,'initial_revision':INIT_REVISION,'initial_path':'run/selected',
        'effective_batch':16,'micro_batch':4,'grad_accum':4,'selection':'lowest new-development NLL including step 0',
        'init_limitation':'Continuation from causal v4, not a from-scratch causal-mask study.',
        'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,'manifest':json.loads(Path('data/phase2-manifest.json').read_text())}
    (out/'run_config.json').write_text(json.dumps(cfg,indent=2));print(json.dumps(cfg),flush=True)
    initial=metrics(infer(model,dev,batch_size=4),targets(dev));best=initial['nll'];best_step=0
    model.save(out/'selected');logs=[];stream=hashlib.sha256();start=time.monotonic();tokens=0
    for step in range(1,args.steps+1):
        model.train();opt.zero_grad(set_to_none=True)
        schedule=min(1.,step/40)*(.1+.9*.5*(1+math.cos(math.pi*step/args.steps)))
        opt.param_groups[0]['lr']=3e-5*schedule;opt.param_groups[1]['lr']=1e-4*schedule
        # Half public examples, half new grounded hard cases. Same RNG stream per arm.
        public=[r for r in train if 'repo' in r['source']] if step==1 else public
        hard=[r for r in train if 'generator' in r['source']] if step==1 else hard
        rows=shuffled_options(rng.sample(public,8)+rng.sample(hard,8),rng)
        losses=[]
        stream.update(json.dumps(rows,sort_keys=True,ensure_ascii=False).encode())
        tokens+=sum(model.token_length(r) for r in rows)
        for offset in range(0,16,4):
            part=rows[offset:offset+4]
            loss=F.cross_entropy(model(model.prepare(part),counts(part)),torch.tensor(targets(part),device='cuda'))
            (loss/4).backward();losses.append(loss.item())
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
        entry={'step':step,'loss':sum(losses)/4,'wall_s':time.monotonic()-start,'tokens':tokens}
        logs.append(entry)
        if step%20==0:print(json.dumps(entry),flush=True)
        if step%200==0 or step==args.steps:
            score=metrics(infer(model,dev,batch_size=4),targets(dev))
            if score['nll']<best:best=score['nll'];best_step=step;model.save(out/'selected')
            model.save(out/'checkpoint')
            torch.save({'step':step,'optimizer':opt.state_dict(),'torch_rng':torch.get_rng_state(),
                'cuda_rng':torch.cuda.get_rng_state_all(),'python_rng':rng.getstate()},out/'checkpoint/training_state.pt')
            (out/'training.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in logs))
            print(json.dumps({'step':step,'dev':score,'selected_step':best_step}),flush=True)
            upload(out,args.hf_repo,'run')
    del model,opt;torch.cuda.empty_cache()
    model=DecisionModel.load(out/'selected',device='cuda')
    cal=splits['calibration'];cal_logits=infer(model,cal,batch_size=4)
    model.temperature=fit_temperature(cal_logits,targets(cal));model.save(out/'release')
    selection={'selected_step':best_step,'initial_dev':initial,'selected_dev_nll':best,'temperature':model.temperature,
        'training_input_stream_sha256':stream.hexdigest(),'training_tokens':tokens,'wall_s':time.monotonic()-start,
        'peak_cuda_gib':torch.cuda.max_memory_allocated()/2**30}
    (out/'selection.json').write_text(json.dumps(selection,indent=2))
    upload(out/'release',args.hf_repo);upload(out,args.hf_repo,'run')
    print(json.dumps({'status':'training_complete','selection':selection}),flush=True)

if __name__=='__main__':main()
