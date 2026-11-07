"""Train on four aligned domains; only validation selects, only calibration fits T."""
import argparse,hashlib,json,math,random,time
from collections import Counter
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import minimize_scalar
from scipy.special import softmax
from huggingface_hub import HfApi,snapshot_download,hf_hub_download
from huggingface_hub.errors import EntryNotFoundError
from .code_model import CodeDecisionModel,SPECS
from .phase3_data import read,sha
from .phase4_metrics import labels,accepted,metrics
from .evaluate_jev import digest

SEED=20261107
DATA=Path('data/private/phase4')
DATASET='jsaurabh/financial-decision-phase4'

def mark(family,status,**kw):
    obj={'role':family,'status':status,'time':time.time(),**kw};print(json.dumps(obj),flush=True)
    if Path('/content').exists():
        p=Path('/content/phase4-status.json');tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(obj));tmp.replace(p)

def meta(row):return {'id':row['id'],'family':row['family'],'task':row['task'],'domain':row['domain'],'labels':labels(row),'target':row['target'],'input_sha256':digest(row)}

@torch.inference_mode()
def infer(model,rows):
    model.eval();predictions={};logits={}
    for row in sorted(rows,key=lambda r:r['token_lengths'][model.family] or 10**9):
        n=row['token_lengths'][model.family]
        if n is None or n>model.max_length:predictions[row['id']]=meta(row)|{'status':'unsupported','error_type':'over_context_limit'};continue
        try:
            values=model(model.prepare([row]))[0,:len(labels(row))].float().cpu().numpy()
            p=softmax(values/model.temperature);assert np.isfinite(p).all()
            logits[row['id']]=values.tolist();predictions[row['id']]=meta(row)|{'status':'ok','prediction':labels(row)[int(p.argmax())],'probabilities':p.tolist()}
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache();predictions[row['id']]=meta(row)|{'status':'error','error_type':'cuda_out_of_memory'}
    return predictions,logits

def set_loss(logits,row):
    indices=[labels(row).index(key) for key in accepted(row)]
    return torch.logsumexp(logits,dim=-1)-torch.logsumexp(logits[:,indices],dim=-1)

def permuted(row,position):
    if row['question']['type']!='choice':return row
    row=dict(row);question=dict(row['question']);items=list(question['criteria'].items());random.Random(SEED+position).shuffle(items);question['criteria']=dict(items);row['question']=question;return row

def upload(out,repo):
    api=HfApi();assert api.model_info(repo).private
    for attempt in range(4):
        try:api.upload_folder(repo_id=repo,folder_path=str(out),path_in_repo='run',commit_message='Aligned four-domain training checkpoint');return
        except Exception:
            if attempt==3:raise
            time.sleep(10*(attempt+1))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--family',choices=list(SPECS),required=True);args=parser.parse_args();family=args.family
    torch.set_num_threads(8);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED);random.seed(SEED)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cuda.enable_flash_sdp(True);torch.backends.cuda.enable_mem_efficient_sdp(True);torch.backends.cuda.enable_math_sdp(True);torch.backends.cuda.enable_cudnn_sdp(False)
    manifest_path=Path('data/phase4-manifest.json');manifest=json.loads(manifest_path.read_text());assert manifest['status']=='sealed_before_training'
    assert sha(Path('PHASE4_TRAINING_PROTOCOL.md').read_bytes())==manifest['protocol_sha256']
    assert sha(Path('data/phase4-answer-codes.json').read_bytes())==manifest['answer_code_vocabulary_sha256']
    splits={}
    # Test files do not exist in the training archive and are never loaded here.
    for split in ['train','validation','calibration','selection_validation']:
        path=DATA/(split+'.jsonl');entry=manifest['selection_validation'] if split=='selection_validation' else manifest['splits'][split]
        assert sha(path.read_bytes())==entry['sha256'];splits[split]=read(path)
    train,validation,calibration,selection=(splits[s] for s in ['train','validation','calibration','selection_validation'])
    assert set(r['id'] for r in selection)<=set(r['id'] for r in validation)
    for a,b in [('train','validation'),('train','calibration'),('validation','calibration')]:assert {r['family'] for r in splits[a]}.isdisjoint({r['family'] for r in splits[b]})
    repo='jsaurabh/financial-decision-phase4-'+family;out=Path('runs')/('phase4-'+family);out.mkdir(parents=True,exist_ok=True)
    api=HfApi();assert api.model_info(repo).private
    state=out/'checkpoint/training_state.pt'
    if not state.exists():
        files=api.list_repo_files(repo)
        if 'run/checkpoint/training_state.pt' in files:
            import shutil
            p=Path(snapshot_download(repo,allow_patterns=['run/**']))/'run';shutil.copytree(p,out,dirs_exist_ok=True)
    if not (out/'artifact.json').exists() and (out/'release/decision_config.json').exists():
        try:
            p=hf_hub_download(DATASET,'runs/'+family+'/artifact.json',repo_type='dataset')
            artifact=json.loads(Path(p).read_text());assert artifact['manifest_sha256']==sha(manifest_path.read_bytes())
            (out/'artifact.json').write_bytes(Path(p).read_bytes())
        except EntryNotFoundError:pass
    if (out/'artifact.json').exists() and (out/'summary.json').exists() and (out/'release/decision_config.json').exists():
        assert json.loads((out/'run_config.json').read_text())['manifest_sha256']==sha(manifest_path.read_bytes())
        api.upload_file(repo_id=DATASET,repo_type='dataset',path_or_fileobj=str(out/'artifact.json'),path_in_repo='runs/'+family+'/artifact.json')
        mark(family,'training_complete',repo=repo,resumed_completed_release=True);return
    resuming=state.exists();mark(family,'loading_model',resume=resuming)
    model=CodeDecisionModel.load(out/'checkpoint',device='cuda') if resuming else CodeDecisionModel.pretrained(family)
    for name,p in model.backbone.named_parameters():p.requires_grad_('lora_' in name)
    model.readout.requires_grad_(True);model.backbone.language_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    order=list(range(len(train)));random.Random(SEED).shuffle(order);steps=math.ceil(len(order)/8)
    config={'family':family,'base':SPECS[family],'architecture':'native-chat-code-readout','initialization':'fresh instruction base; LM-initialized answer code rows; rank16 language-only LoRA',
        'manifest_sha256':sha(manifest_path.read_bytes()),'seed':SEED,'epochs':1,'micro_batch':1,'effective_batch':8,'steps':steps,'max_tokens':8192,'max_choices':128,
        'backbone_lr':2e-5,'readout_lr':5e-5,'selection':'validation-only equal four-domain macro-F1/normalized routing regret','calibration':'positive temperature on calibration only',
        'code_sha256':sha(Path(__file__).read_bytes()+Path('finance_decisions/code_model.py').read_bytes()+Path('finance_decisions/phase4_metrics.py').read_bytes()),
        'training_order_sha256':sha(json.dumps([train[i]['id'] for i in order]).encode()),'early_stopping':'two consecutive scheduled validation checks without improvement',
        'attention_backend':'native causal SDPA automatic Flash/memory-efficient/math; native Qwen recurrence; TF32 disabled','torch':torch.__version__}
    (out/'runtime.json').write_text(json.dumps({'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,'time':time.time()},indent=2)+'\n')
    config_path=out/'run_config.json'
    if config_path.exists():assert json.loads(config_path.read_text())==config,'Resume config mismatch'
    config_path.write_text(json.dumps(config,indent=2)+'\n')
    # Longest-example backward preflight checks actual gradient and memory paths.
    rng_cpu=torch.get_rng_state();rng_cuda=torch.cuda.get_rng_state_all();model.train()
    longest=max(train,key=lambda r:r['token_lengths'][family]);assert model.token_length(longest)==longest['token_lengths'][family]
    logits=model(model.prepare([longest]));loss=set_loss(logits,longest).mean();loss.backward()
    assert model.readout.weight.grad is not None and torch.isfinite(model.readout.weight.grad).all()
    grads=[p.grad for n,p in model.backbone.named_parameters() if 'lora_' in n and p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    model.zero_grad(set_to_none=True);torch.set_rng_state(rng_cpu);torch.cuda.set_rng_state_all(rng_cuda);torch.cuda.empty_cache()
    mark(family,'gradient_preflight_passed',tokens=longest['token_lengths'][family],peak_cuda_gib=torch.cuda.max_memory_allocated()/2**30)
    optimizer=torch.optim.AdamW([{'params':[p for p in model.backbone.parameters() if p.requires_grad],'lr':2e-5},{'params':model.readout.parameters(),'lr':5e-5}],weight_decay=.01)
    start=0;best=-math.inf;selected=0;stale=0;tokens=0
    if resuming:
        saved=torch.load(state,map_location='cpu',weights_only=True);optimizer.load_state_dict(saved['optimizer']);start=saved['step'];best=saved['best'];selected=saved['selected'];stale=saved['stale'];tokens=saved['tokens']
        torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state_all(saved['cuda_rng'])
    else:
        predictions,_=infer(model,selection);score=metrics(selection,predictions);best=score['selection_score']
        (out/'initial_validation.json').write_text(json.dumps(score,indent=2)+'\n');model.save(out/'selected');mark(family,'initial_validation_complete',selection_score=best)
    task_counts=Counter(r['task'] for r in train);domain_tasks={d:{r['task'] for r in train if r['domain']==d} for d in {r['domain'] for r in train}}
    wall=time.monotonic()
    step=start
    for step in (range(start+1,steps+1) if stale<2 else []):
        model.train();optimizer.zero_grad(set_to_none=True);rate=min(1.,step/100)*(.1+.9*.5*(1+math.cos(math.pi*step/steps)))
        optimizer.param_groups[0]['lr']=2e-5*rate;optimizer.param_groups[1]['lr']=5e-5*rate;losses=[]
        positions=list(range((step-1)*8,min(step*8,len(order))))
        for position in positions:
            row=permuted(train[order[position]],position);logits=model(model.prepare([row]));loss=set_loss(logits,row).mean()/math.log(len(labels(row)))
            weight=len(train)/(4*len(domain_tasks[row['domain']])*task_counts[row['task']]);loss=loss*weight;assert torch.isfinite(loss)
            (loss/len(positions)).backward();losses.append(float(loss.detach()));tokens+=row['token_lengths'][family]
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
        if step==1 or step%20==0:mark(family,'training',step=step,total_steps=steps,loss=sum(losses)/len(losses),tokens=tokens,wall_s=time.monotonic()-wall)
        if step%500==0 or step==steps:
            predictions,_=infer(model,selection);score=metrics(selection,predictions);(out/f'validation-step-{step}.json').write_text(json.dumps(score,indent=2)+'\n')
            if score['selection_score']>best+1e-6:best=score['selection_score'];selected=step;stale=0;model.save(out/'selected')
            else:stale+=1
            model.save(out/'checkpoint');torch.save({'step':step,'best':best,'selected':selected,'stale':stale,'tokens':tokens,'optimizer':optimizer.state_dict(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()},state)
            upload(out,repo);mark(family,'checkpointed',step=step,total_steps=steps,selected_step=selected,validation_selection_score=score['selection_score'])
            if stale>=2:break
    del model,optimizer;torch.cuda.empty_cache();model=CodeDecisionModel.load(out/'selected',device='cuda')
    validation_predictions,val_logits=infer(model,validation);cal_predictions,cal_logits=infer(model,calibration)
    def objective(log_t):
        losses=[]
        for r in calibration:
            if r['id'] not in cal_logits:continue
            p=softmax(np.array(cal_logits[r['id']],dtype=float)/math.exp(log_t));indices=[labels(r).index(k) for k in accepted(r)]
            losses.append(-math.log(max(float(p[indices].sum()),1e-12)))
        return float(np.mean(losses))
    fitted=minimize_scalar(objective,bounds=(-3,3),method='bounded');model.temperature=math.exp(float(fitted.x));model.save(out/'release')
    for name,predictions,logits in [('validation',validation_predictions,val_logits),('calibration',cal_predictions,cal_logits)]:
        for identity,values in logits.items():
            p=softmax(np.array(values,dtype=float)/model.temperature);predictions[identity].update(probabilities=p.tolist(),temperature=model.temperature)
        (out/(name+'_predictions.jsonl')).write_text(''.join(json.dumps(p,ensure_ascii=False,allow_nan=False)+'\n' for p in predictions.values()))
    summary={'status':'training_complete','selected_step':selected,'last_training_step':step,'total_planned_steps':steps,'early_stopped':stale>=2,'tokens':tokens,'temperature':model.temperature,
        'validation':metrics(validation,validation_predictions),'selection_score':best,'test_accessed_during_training':False,'peak_cuda_gib':torch.cuda.max_memory_allocated()/2**30}
    seen=[train[i] for i in order[:min(selected*8,len(train))]]
    summary.update(selected_training_examples=len(seen),selected_training_tasks=dict(Counter(r['task'] for r in seen)),selected_training_domains=dict(Counter(r['domain'] for r in seen)))
    (out/'release/README.md').write_text('# Aligned financial decisions phase4\n\nFresh pinned '+SPECS[family]['repo']+' base, native causal chat answer-code head and rank16 language LoRA. Core tasks are represented in training/validation/calibration/test; source/task transfer is separate. Source/base terms apply. Research model; no production validation or superiority claim from training completion.\n')
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');upload(out,repo)
    artifact={'repo':repo,'revision':api.model_info(repo).sha,'path':'run/release','manifest_sha256':sha(manifest_path.read_bytes()),'training_summary_sha256':sha((out/'summary.json').read_bytes())}
    (out/'artifact.json').write_text(json.dumps(artifact,indent=2)+'\n')
    api.upload_file(repo_id=DATASET,repo_type='dataset',path_or_fileobj=str(out/'artifact.json'),path_in_repo='runs/'+family+'/artifact.json')
    mark(family,'training_complete',selected_step=selected,repo=repo)

if __name__=='__main__':main()
