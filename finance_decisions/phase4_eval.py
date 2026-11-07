"""Matched core/transfer evaluation after checkpoint selection and calibration."""
import argparse,concurrent.futures,hashlib,json,time
from collections import Counter
from pathlib import Path
import numpy as np
from scipy.special import softmax
from huggingface_hub import HfApi,hf_hub_download
from huggingface_hub.errors import EntryNotFoundError
from .phase3_data import read,sha
from .evaluate_jev import call,digest
from .code_model import CodeDecisionModel
from .phase4_metrics import labels,metrics

REPO='jsaurabh/financial-decision-phase4'
DATA=Path('data/private/phase4')

def meta(row):return {k:row[k] for k in ['id','family','task','domain','cohort','target']}|{'input_sha256':digest(row),'labels':labels(row)}

def validate(row,p):
    if p['id']!=row['id'] or p['input_sha256']!=digest(row) or p['labels']!=labels(row):raise ValueError('Heldout input/candidate mismatch')
    if p.get('status')=='ok':
        v=p['probabilities']
        if len(v)!=len(labels(row)) or not all(np.isfinite(x) and 0<=x<=1 for x in v) or abs(sum(v)-1)>1e-5 or p['prediction'] not in labels(row):raise ValueError('Invalid heldout prediction')

def progress(out,records,total,status='evaluating'):
    result={'role':out.name,'status':status,'completed':len(records),'total':total,'cohorts':dict(Counter(p['cohort'] for p in records.values())),'statuses':dict(Counter(p['status'] for p in records.values())),'time':time.time()}
    (out/'progress.json').write_text(json.dumps(result));print(json.dumps(result),flush=True)
    if out.name!='jev' and Path('/content').exists():
        tmp=Path('/content/phase4-progress.tmp');tmp.write_text(json.dumps(result));tmp.replace('/content/phase4-status.json')

def checkpoint(out,records,total,done=False):
    progress(out,records,total,'complete' if done else 'evaluating')
    for attempt in range(4):
        try:HfApi().upload_folder(repo_id=REPO,repo_type='dataset',folder_path=str(out),path_in_repo='predictions/'+out.name,allow_patterns=['predictions.jsonl','config.json','progress.json'],commit_message='Frozen core and separate transfer receipts: '+out.name);return
        except Exception:
            if attempt==3:raise
            time.sleep(10*(attempt+1))

def append(out,records,p):
    if p['id'] in records:raise ValueError('Refusing to replace a heldout receipt')
    with (out/'predictions.jsonl').open('a') as f:f.write(json.dumps(p,ensure_ascii=False,allow_nan=False)+'\n')
    records[p['id']]=p

def main():
    import fcntl
    parser=argparse.ArgumentParser();parser.add_argument('--role',choices=['gemma','qwen','jev'],required=True);args=parser.parse_args();role=args.role
    manifest_path=Path('data/phase4-manifest.json');manifest=json.loads(manifest_path.read_text());assert manifest['status']=='sealed_before_training'
    rows=[]
    for split in ['core_test','transfer_test']:
        path=DATA/(split+'.jsonl');entry=manifest['transfer_test'] if split=='transfer_test' else manifest['splits'][split]
        assert sha(path.read_bytes())==entry['sha256'];rows+=read(path)
    assert len(rows)==len({r['id'] for r in rows});byid={r['id']:r for r in rows};assert HfApi().dataset_info(REPO).private
    out=Path('data/private/phase4-predictions')/role;out.mkdir(parents=True,exist_ok=True);lock=(out/'worker.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    cfg={'role':role,'manifest_sha256':sha(manifest_path.read_bytes()),'max_tokens':8192,'no_truncation':True,'core_and_transfer_separate':True,'inference_code_sha256':sha(Path(__file__).read_bytes()+Path('finance_decisions/code_model.py').read_bytes())}
    if role=='jev':cfg.update(alias='jev-latest',required_provider='jev-1.14.0',request_fields='model/state/questions only; reference and outcome metadata excluded')
    else:
        artifact=json.loads(Path('runs',f'phase4-{role}','artifact.json').read_text());cfg['artifact']=artifact
        assert artifact['manifest_sha256']==cfg['manifest_sha256']
    config_path=out/'config.json';cache=out/'predictions.jsonl'
    if not config_path.exists():
        try:
            rev=HfApi().dataset_info(REPO).sha;old=hf_hub_download(REPO,'predictions/'+role+'/config.json',repo_type='dataset',revision=rev)
            assert json.loads(Path(old).read_text())==cfg,'Remote inference configuration mismatch'
            p=hf_hub_download(REPO,'predictions/'+role+'/predictions.jsonl',repo_type='dataset',revision=rev);cache.write_bytes(Path(p).read_bytes())
        except EntryNotFoundError:pass
    if config_path.exists():assert json.loads(config_path.read_text())==cfg,'Local inference configuration mismatch'
    config_path.write_text(json.dumps(cfg,indent=2)+'\n');records={}
    if cache.exists():
        with cache.open() as f:lines=list(f)
        for i,line in enumerate(lines):
            try:p=json.loads(line)
            except json.JSONDecodeError:
                if i!=len(lines)-1:raise
                cache.write_text(''.join(lines[:-1]));break
            validate(byid[p['id']],p);assert p['id'] not in records;records[p['id']]=p
    pending=[r for r in rows if r['id'] not in records]
    if role=='jev':
        # Earliest matching1.14 receipts may be reused; never select by correctness.
        prior={}
        for folder in ['public-eval-report','final-eval-report','tracesafe-eval-report']:
            p=Path('data/private')/folder/'jev/predictions.jsonl'
            if p.exists():
                for receipt in sorted(read(p),key=lambda x:x.get('timestamp_utc','')):
                    if receipt.get('status')=='ok' and receipt.get('provider')=='jev-1.14.0':prior.setdefault((receipt['input_sha256'],tuple(receipt['labels'])),receipt)
        for r in pending:
            old=prior.get((digest(r),tuple(labels(r))))
            if old:
                p=dict(old)|meta(r)|{'reused_prior_receipt':True};validate(r,p);append(out,records,p)
        pending=[r for r in rows if r['id'] not in records];checkpoint(out,records,len(rows))
        def obtain(row):
            try:
                p=call(row,'jev-latest',max_choices=128,timeout=180)|meta(row)|{'status':'ok'};validate(row,p)
                if p['provider']!='jev-1.14.0':p.update(status='provider_mismatch',error_type='unexpected_provider_version')
            except Exception as exc:p=meta(row)|{'status':'error','error_type':type(exc).__name__,'error':str(exc)[:160]}
            return p
        if pending:
            p=obtain(pending.pop(0));append(out,records,p)
            if p['status']!='ok':checkpoint(out,records,len(rows));raise RuntimeError('Initial Jev request failed or provider changed; bulk queue not dispatched')
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
            futures={pool.submit(obtain,row):row for row in pending}
            for i,future in enumerate(concurrent.futures.as_completed(futures),1):
                append(out,records,future.result())
                if i%100==0:progress(out,records,len(rows))
                if i%250==0:checkpoint(out,records,len(rows))
    else:
        import torch
        torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cuda.enable_cudnn_sdp(False)
        model=CodeDecisionModel.load(Path('runs',f'phase4-{role}','release'),device='cuda');model.requires_grad_(False);model.eval()
        pending.sort(key=lambda r:(r['cohort']!='core',r['token_lengths'][role] or 10**9,r['id']))
        for i,row in enumerate(pending,1):
            n=row['token_lengths'][role]
            if n is None or n>8192:p=meta(row)|{'status':'unsupported','error_type':'over_context_limit','input_tokens':n}
            else:
                try:
                    with torch.inference_mode():values=model(model.prepare([row]))[0,:len(labels(row))].float().cpu().numpy()
                    probabilities=softmax(values/model.temperature);p=meta(row)|{'status':'ok','prediction':labels(row)[int(probabilities.argmax())],'probabilities':probabilities.tolist(),'input_tokens':n};validate(row,p)
                except torch.OutOfMemoryError:
                    torch.cuda.empty_cache();p=meta(row)|{'status':'error','error_type':'cuda_out_of_memory'}
            append(out,records,p)
            if i%100==0:progress(out,records,len(rows))
            if i%250==0:checkpoint(out,records,len(rows))
    checkpoint(out,records,len(rows),done=True)

if __name__=='__main__':main()
