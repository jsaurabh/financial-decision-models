"""Resumable frozen public evaluation: no optimizer, fitting, truncation or training."""
import argparse,gzip,hashlib,json,os,random,sys,time,gc
from pathlib import Path
import numpy as np
import torch
from scipy.special import softmax
from huggingface_hub import HfApi,snapshot_download,hf_hub_download
from .schema import options
from .evaluate_jev import call,digest
from .public_metrics import correct,score

REPO='jsaurabh/financial-decision-public-eval'
DATA=Path('data/private/public-eval/rows.jsonl')


def read(path):
    # Unicode U+2028/U+2029 inside JSON strings are not JSONL delimiters.
    with Path(path).open() as f:return [json.loads(l) for l in f if l.strip()]

def data():
    manifest=json.loads(Path('data/public-eval-manifest.json').read_text())
    if not DATA.exists():
        DATA.parent.mkdir(parents=True,exist_ok=True)
        with gzip.open('/content/public-rows.jsonl.gz','rb') as f:DATA.write_bytes(f.read())
    assert hashlib.sha256(DATA.read_bytes()).hexdigest()==manifest['rows_sha256']
    return read(DATA),manifest


def labels(row):return options(row['question'],max_choices=11)[0]

def checkpoint(out,records,rows,config):
    from collections import Counter
    (out/'summary.json').write_text(json.dumps({'config':config,'suites':score(rows,records),'completed_records':len(records),
        'resolved_providers':dict(Counter(r['provider'] for r in records if r.get('provider')))},indent=2)+'\n')
    api=HfApi();assert api.dataset_info(REPO).private
    api.upload_folder(repo_id=REPO,repo_type='dataset',folder_path=str(out),path_in_repo='runs/'+out.name,commit_message='Public evaluation progress: '+out.name)


def progress(role,model,n,total):
    obj={'status':'evaluating','role':role,'model':model,'completed':n,'total':total,'time':time.time()}
    print(json.dumps(obj),flush=True)
    if Path('/content').exists():
        p=Path('/content/public-status.json');temp=p.with_name('public-progress.tmp');temp.write_text(json.dumps(obj));temp.replace(p)


def restore(out,config):
    out.mkdir(parents=True,exist_ok=True)
    if not (out/'predictions.jsonl').exists():
        try:
            path=hf_hub_download(REPO,'runs/'+out.name+'/predictions.jsonl',repo_type='dataset')
            old=hf_hub_download(REPO,'runs/'+out.name+'/config.json',repo_type='dataset')
            if json.loads(Path(old).read_text())==config:(out/'predictions.jsonl').write_bytes(Path(path).read_bytes())
        except Exception:pass
    old=out/'config.json'
    if old.exists() and json.loads(old.read_text())!=config:raise ValueError('Resume config differs')
    old.write_text(json.dumps(config,indent=2)+'\n')
    if not (out/'predictions.jsonl').exists():return []
    # Ignore only a torn final line; never ignore a corrupt interior record.
    lines=(out/'predictions.jsonl').read_text().splitlines();result=[]
    for i,line in enumerate(lines):
        try:result.append(json.loads(line))
        except json.JSONDecodeError:
            if i!=len(lines)-1:raise
    if len(result)!=len(lines):(out/'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in result))
    return result


def append(out,records,new):
    with (out/'predictions.jsonl').open('a') as f:
        for r in new:f.write(json.dumps(r,allow_nan=False)+'\n')
    records.extend(new)


def metadata(row):
    return {'id':row['id'],'suite':row['suite'],'family':row['family'],'input_sha256':digest(row),'labels':labels(row),'target':row['target']}


def local(role,rows,manifest,smoke):
    torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False);torch.backends.cuda.enable_math_sdp(True)
    artifact_index=json.loads(Path('results/phase2/artifacts.json').read_text())
    names=['causal','joint'] if role=='gemma' else ['qwen']
    for name in names:
        item=artifact_index[name];out=Path('results/public-eval')/name
        config={'model':name,'repo':item['repo'],'revision':item['revision'],'manifest_sha256':manifest['rows_sha256'],
            'max_tokens':8192,'max_choices':11,'weights':'bfloat16 with published float32 adapters/readout','attention_backend':'strict math SDPA; native Qwen recurrence unchanged',
            'temperature':'published checkpoint; not refitted','context_note':'Inference-only extension beyond Gemma release 2048-token / 8-candidate validation envelope.'}
        config['inference_code_sha256']=hashlib.sha256(b''.join(Path('finance_decisions',f).read_bytes() for f in ['public_eval.py','model.py','candidate_model.py','schema.py'])).hexdigest()
        if role=='gemma':
            from .model import DecisionModel
            artifact=snapshot_download(item['repo'],revision=item['revision'],allow_patterns=['decision_config.json','readout.safetensors','backbone/**','tokenizer/**'])
            model=DecisionModel.load(artifact,device='cuda');model.max_length=8192;model.max_chars=1000000;model.max_choices=11
            def length(row):return model.token_length(row)
            def run(part):return model(model.prepare(part),[len(labels(r)) for r in part])
        else:
            sys.path.insert(0,str(Path('references/qwen27b').resolve()))
            from eval_autojev27b_vs_jev import load_model
            from autojev.model import decision_messages
            artifact=snapshot_download(item['repo'],revision=item['revision']);model,_=load_model(Path(artifact))
            def length(row):
                text=model.processor.apply_chat_template(decision_messages(row,model.codes),tokenize=False,add_generation_prompt=True,enable_thinking=False)
                return len(model.processor(text=[text],images=None,padding=False)['input_ids'][0])
            def run(part):return model(model.prepare(part,max_length=8192))
        model.requires_grad_(False);model.eval();config['torch']=torch.__version__;config['gpu']=torch.cuda.get_device_name();config['temperature_value']=model.temperature
        for suite in ['when2call','routerbench_0shot_quality','tsbench_asb-traj','contractnli']:
            example=next(r for r in rows if r['suite']==suite)
            n=length(example)
            if n<=8192:
                with torch.inference_mode():probe=run([example])
                assert probe.shape[0]==1 and probe.shape[1]>=len(labels(example)) and torch.isfinite(probe).all()
                print(json.dumps({'model':name,'schema_smoke':suite,'input_tokens':n,'choices':len(labels(example))}),flush=True)
                del probe
        records=restore(out,config);cached={(r['id'],r['input_sha256']) for r in records};pending=[r for r in rows if (r['id'],digest(r)) not in cached]
        measured=[];new=[]
        for row_number,row in enumerate(pending):
            try:
                n=length(row)
                if n>8192:new.append(metadata(row)|{'status':'unsupported','error_type':'over_context_limit','input_tokens':n})
                else:measured.append((n,row))
            except Exception as exc:new.append(metadata(row)|{'status':'error','error_type':'preparation_'+type(exc).__name__})
            if row_number%2000==0:print(json.dumps({'model':name,'length_checked':row_number,'total':len(pending)}),flush=True)
        append(out,records,new);measured.sort(key=lambda x:(x[0],x[1]['id']))
        progress(role,name,len(records),len(rows));checkpoint(out,records,rows,config)
        index=0;since=0;last_progress=0
        while index<len(measured):
            n=measured[index][0]
            batch=(4 if n<=512 else 2 if n<=2048 else 1) if role=='qwen' else (8 if n<=1024 else 2 if n<=4096 else 1)
            part=[r for _,r in measured[index:index+batch]]
            # Limit batches at the next length bucket; max packed length remains bounded.
            with torch.inference_mode():
                try:
                    logits=run(part).float().cpu().numpy();new=[]
                    for row,logit in zip(part,logits):
                        keys=labels(row);p=softmax(np.asarray(logit[:len(keys)],dtype=float)/model.temperature)
                        if not np.isfinite(p).all():raise FloatingPointError('Nonfinite probabilities')
                        pred=keys[int(p.argmax())]
                        new.append(metadata(row)|{'status':'ok','prediction':pred,'probabilities':p.tolist(),'correct':correct(row,pred)})
                except torch.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    new=[]
                    for row in part:
                        try:
                            l=run([row]).float().cpu().numpy()[0];keys=labels(row);p=softmax(np.asarray(l[:len(keys)],dtype=float)/model.temperature);pred=keys[int(p.argmax())]
                            new.append(metadata(row)|{'status':'ok','prediction':pred,'probabilities':p.tolist(),'correct':correct(row,pred)})
                        except torch.OutOfMemoryError:
                            torch.cuda.empty_cache();new.append(metadata(row)|{'status':'error','error_type':'cuda_out_of_memory'})
                except Exception as exc:
                    new=[metadata(row)|{'status':'error','error_type':type(exc).__name__} for row in part]
            append(out,records,new);index+=len(part);since+=len(part)
            if len(records)-last_progress>=100:progress(role,name,len(records),len(rows));last_progress=len(records)
            if since>=500 or index==len(measured):checkpoint(out,records,rows,config);since=0
        checkpoint(out,records,rows,config)
        del model;gc.collect();torch.cuda.empty_cache()


def jev(rows,manifest):
    import concurrent.futures
    out=Path('results/public-eval/jev')
    config={'model':'jev-latest','manifest_sha256':manifest['rows_sha256'],'max_choices':11,'request':'state/question only; reference and outcome metadata never sent','temperature':'native provider output'}
    records=restore(out,config);cached={(r['id'],r['input_sha256']) for r in records};pending=[r for r in rows if (r['id'],digest(r)) not in cached]
    byid={r['id']:r for r in rows};new=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        futures={pool.submit(call,r,'jev-latest',max_choices=11,timeout=180):r for r in pending}
        for i,future in enumerate(concurrent.futures.as_completed(futures),1):
            row=futures[future]
            try:
                prediction=future.result();prediction.update(metadata(row));prediction.update(status='ok',correct=correct(row,prediction['prediction']))
            except Exception as exc:prediction=metadata(row)|{'status':'error','error_type':type(exc).__name__,'error':str(exc)}
            append(out,records,[prediction])
            if i%100==0:print(json.dumps({'jev_completed':len(records),'total':len(rows),'errors':sum(r['status']!='ok' for r in records)}),flush=True)
            if i%500==0:checkpoint(out,records,rows,config)
    checkpoint(out,records,rows,config)


def main():
    p=argparse.ArgumentParser();p.add_argument('--role',choices=['gemma','qwen','jev'],required=True);p.add_argument('--smoke',action='store_true');args=p.parse_args()
    rows,manifest=data()
    if args.smoke:
        suites=sorted({r['suite'] for r in rows});rows=[next(r for r in rows if r['suite']==s) for s in suites]
    if args.role=='jev':jev(rows,manifest)
    else:local(args.role,rows,manifest,args.smoke)

if __name__=='__main__':main()
