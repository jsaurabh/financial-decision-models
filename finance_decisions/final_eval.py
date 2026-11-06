"""Frozen, paired final evaluation. No optimizer, truncation or label feedback."""
import argparse, concurrent.futures, hashlib, json, time
from collections import Counter
from pathlib import Path

import numpy as np
from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from huggingface_hub.errors import EntryNotFoundError

from .evaluate_jev import call, digest
from .schema import options

REPO='jsaurabh/financial-decision-final-eval'
DATA=Path('data/private/final-eval/rows.jsonl')
PRIVATE=Path('data/private/final-eval-results')

def read(path):
    with Path(path).open() as f:return [json.loads(line) for line in f if line.strip()]

def labels(row):return options(row['question'],max_choices=11)[0]

def meta(row):
    return {k:row[k] for k in ['id','suite','domain','cohort','family','target']}|{'input_sha256':digest(row),'labels':labels(row)}

def valid(row,p):
    if p['id']!=row['id'] or p['input_sha256']!=digest(row) or p['labels']!=labels(row):raise ValueError('Prediction input/candidate order mismatch')
    if p.get('status')=='ok':
        prob=p['probabilities']
        if len(prob)!=len(labels(row)) or not all(np.isfinite(v) and 0<=v<=1 for v in prob) or abs(sum(prob)-1)>1e-5 or p['prediction'] not in labels(row):
            raise ValueError('Invalid frozen prediction')

def config(role,manifest):
    code=b''.join(Path('finance_decisions',name).read_bytes() for name in ['final_eval.py','model.py','candidate_model.py','schema.py','evaluate_jev.py'])
    return {'role':role,'rows_sha256':manifest['rows_sha256'],'protocol_sha256':manifest['protocol_sha256'],
        'system':manifest[role],'inference_code_sha256':hashlib.sha256(code).hexdigest(),
        'maximum_packed_tokens':8192,'maximum_choices':11,'truncation':False,'refitting':False,
        'native_decision_rule':True,'attention':'strict math SDPA, independent-causal candidate branches, TF32 off'}

def restore(out,cfg,rows):
    out.mkdir(parents=True,exist_ok=True);old=out/'config.json';cache=out/'predictions.jsonl'
    if not old.exists() and not cache.exists():
        try:
            prior=Path(hf_hub_download(REPO,f'runs/{out.name}/config.json',repo_type='dataset'))
            if json.loads(prior.read_text())!=cfg:raise ValueError('Remote resume configuration differs')
            remote=hf_hub_download(REPO,f'runs/{out.name}/predictions.jsonl',repo_type='dataset')
            cache.write_bytes(Path(remote).read_bytes());old.write_bytes(prior.read_bytes())
        except EntryNotFoundError:pass
    if old.exists() and json.loads(old.read_text())!=cfg:raise ValueError('Resume configuration differs')
    old.write_text(json.dumps(cfg,indent=2)+'\n');records={};byid={r['id']:r for r in rows}
    if cache.exists():
        # Only an incomplete final physical line is recoverable.
        with cache.open() as f:lines=list(f)
        for i,line in enumerate(lines):
            try:p=json.loads(line)
            except json.JSONDecodeError:
                if i!=len(lines)-1:raise
                cache.write_text(''.join(lines[:-1]));break
            valid(byid[p['id']],p)
            if p['id'] in records:raise ValueError('Duplicate prediction receipt')
            records[p['id']]=p
    return records

def append(out,records,p):
    if p['id'] in records:raise ValueError('Refusing to overwrite a frozen receipt')
    with (out/'predictions.jsonl').open('a') as f:f.write(json.dumps(p,ensure_ascii=False,allow_nan=False)+'\n')
    records[p['id']]=p

def progress(out,records,total,status='evaluating'):
    obj={'status':status,'role':out.name,'completed':len(records),'total':total,'time':time.time(),
         'cohort_completed':dict(Counter(p['cohort'] for p in records.values())),
         'statuses':dict(Counter(p['status'] for p in records.values()))}
    dest=out/'progress.json';tmp=dest.with_suffix('.tmp');tmp.write_text(json.dumps(obj));tmp.replace(dest)
    if out.name=='gemma' and Path('/content').exists():
        tmp=Path('/content/final-progress.tmp');tmp.write_text(json.dumps(obj));tmp.replace('/content/final-status.json')
    print(json.dumps(obj),flush=True)

def checkpoint(out,records,total,terminal=False):
    progress(out,records,total,'complete' if terminal else 'evaluating')
    for attempt in range(4):
        try:
            HfApi().upload_folder(repo_id=REPO,repo_type='dataset',folder_path=str(out),path_in_repo='runs/'+out.name,
                allow_patterns=['config.json','predictions.jsonl','progress.json','runtime.json'],commit_message='Frozen final-test receipts: '+out.name)
            return
        except Exception:
            if attempt==3:raise
            time.sleep(5*(attempt+1))

def historical_cache(rows):
    cache=Path('data/private/public-eval-report/jev/predictions.jsonl')
    if not cache.exists():cache=Path(hf_hub_download('jsaurabh/financial-decision-public-eval','runs/jev/predictions.jsonl',repo_type='dataset'))
    receipts=read(cache);receipts.sort(key=lambda p:p.get('timestamp_utc',''))
    result={}
    for p in receipts:
        if p.get('status')=='ok' and p.get('provider')=='jev-1.14.0':result.setdefault((p['input_sha256'],tuple(p['labels'])),p)
    return result

def jev(out,records,rows,manifest):
    cached=historical_cache(rows)
    for row in rows:
        if row['id'] in records or row['cohort']!='regression':continue
        receipt=cached.get((digest(row),tuple(labels(row))))
        if receipt:
            p=dict(receipt);p.update(meta(row));p['reused_from']='public-eval/jev; earliest matching provider receipt'
            valid(row,p);append(out,records,p)
    checkpoint(out,records,len(rows))
    pending=[r for r in rows if r['id'] not in records]
    def obtain(row):
        try:
            p=call(row,manifest['jev']['alias'],max_choices=11,timeout=180)|meta(row)|{'status':'ok'}
            valid(row,p)
            if p['provider']!=manifest['jev']['required_provider']:p.update(status='provider_mismatch',error_type='unexpected_provider_version')
        except Exception as exc:p=meta(row)|{'status':'error','error_type':type(exc).__name__}
        return p
    # Verify resolved provider before dispatching the bounded concurrent queue.
    if pending:
        p=obtain(pending.pop(0));append(out,records,p)
        if p['status']!='ok':
            checkpoint(out,records,len(rows));raise RuntimeError('Initial final-test API probe failed or provider changed; no bulk requests dispatched')
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        futures={pool.submit(obtain,row):row for row in pending}
        for i,future in enumerate(concurrent.futures.as_completed(futures),1):
            p=future.result();append(out,records,p)
            if i%50==0:progress(out,records,len(rows))
            if i%250==0:checkpoint(out,records,len(rows))
    checkpoint(out,records,len(rows),terminal=True)

def gemma(out,records,rows,manifest):
    import torch
    from scipy.special import softmax
    from .model import DecisionModel
    torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False);torch.backends.cuda.enable_math_sdp(True)
    item=manifest['gemma'];root=snapshot_download(item['repo'],revision=item['revision'],allow_patterns=['run/release/**'])
    model=DecisionModel.load(Path(root)/item['path'],device='cuda')
    assert model.max_length==8192 and model.max_choices==11 and model.attention_mode=='independent-causal'
    assert abs(model.temperature-item['temperature'])<1e-12
    model.requires_grad_(False);model.eval()
    (out/'runtime.json').write_text(json.dumps({'torch':torch.__version__,'gpu':torch.cuda.get_device_name(),'temperature':model.temperature}))
    measured=[]
    for index,row in enumerate(rows):
        if row['id'] in records:continue
        try:
            n=model.token_length(row)
            if n>8192:append(out,records,meta(row)|{'status':'unsupported','error_type':'over_context_limit','input_tokens':n})
            else:measured.append(({'fresh':0,'secondary':1,'regression':2}[row['cohort']],n,row))
        except Exception as exc:append(out,records,meta(row)|{'status':'error','error_type':'preparation_'+type(exc).__name__})
        if index%2000==0:print(json.dumps({'length_checked':index,'total':len(rows)}),flush=True)
    measured.sort(key=lambda item:(item[0],item[1],item[2]['id']))
    checkpoint(out,records,len(rows));index=0;since=0;reported=0
    def run(part):
        logits=model(model.prepare(part),[len(labels(r)) for r in part]).float().cpu().numpy()
        output=[]
        for row,logit in zip(part,logits,strict=True):
            keys=labels(row);p=softmax(np.asarray(logit[:len(keys)],dtype=float)/model.temperature)
            item=meta(row)|{'status':'ok','prediction':keys[int(p.argmax())],'probabilities':p.tolist()}
            valid(row,item);output.append(item)
        return output
    while index<len(measured):
        _,n,_=measured[index];batch=8 if n<=1024 else 2 if n<=4096 else 1
        part=[r for _,_,r in measured[index:index+batch]]
        with torch.inference_mode():
            try:output=run(part)
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache();output=[]
                for row in part:
                    try:output.extend(run([row]))
                    except torch.OutOfMemoryError:
                        torch.cuda.empty_cache();output.append(meta(row)|{'status':'error','error_type':'cuda_out_of_memory'})
            except Exception as exc:output=[meta(row)|{'status':'error','error_type':type(exc).__name__} for row in part]
        for p in output:append(out,records,p)
        index+=len(part);since+=len(part)
        if len(records)-reported>=100:progress(out,records,len(rows));reported=len(records)
        if since>=250:checkpoint(out,records,len(rows));since=0
    checkpoint(out,records,len(rows),terminal=True)

def main():
    import fcntl
    parser=argparse.ArgumentParser();parser.add_argument('--role',choices=['gemma','jev'],required=True);args=parser.parse_args()
    manifest=json.loads(Path('data/final-eval-manifest.json').read_text())
    assert hashlib.sha256(DATA.read_bytes()).hexdigest()==manifest['rows_sha256']
    assert hashlib.sha256(Path('FINAL_EVAL_PROTOCOL.md').read_bytes()).hexdigest()==manifest['protocol_sha256']
    rows=read(DATA);assert len(rows)==manifest['rows']==len({r['id'] for r in rows})
    assert HfApi().dataset_info(REPO).private
    out=PRIVATE/args.role;out.mkdir(parents=True,exist_ok=True)
    lock=(out/'worker.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    records=restore(out,config(args.role,manifest),rows)
    if args.role=='gemma':gemma(out,records,rows,manifest)
    else:jev(out,records,rows,manifest)

if __name__=='__main__':main()
