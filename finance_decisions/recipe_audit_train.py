"""Bounded Gemma-only memorization and matched recipe controls, development only."""
import hashlib
import json
import math
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import HfApi, snapshot_download
from peft import LoraConfig, PeftModel, get_peft_model
from scipy.special import softmax
from transformers import AutoTokenizer, Gemma4ForConditionalGeneration

from .code_model import CODES, SPECS, CodeDecisionModel, messages
from .evaluate_jev import digest
from .phase3_data import read, sha
from .phase4_failure_audit import financial_pairs
from .phase4_metrics import accepted, labels, metrics
from .phase4_train import permuted

SEED = 20261107
DATA = Path('data/private/recipe-audit')
OUT = Path('runs/gemma-recipe-audit')
MODEL_REPO = 'jsaurabh/financial-decision-gemma-recipe-audit'
DATASET_REPO = MODEL_REPO+'-data'
TARGETS = ['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj',
           'in_proj_qkv','in_proj_z','in_proj_a','in_proj_b','out_proj']
MEMORY_ARMS = [('mem_code_lr2e-5','code_plain',2e-5),
               ('mem_code_lr1e-4','code_plain',1e-4),
               ('mem_sft_lr1e-4','native_sft',1e-4)]
PILOT_ARMS = [(f'pilot_{arch}_lr{rate}',arch,lr)
              for arch in ['code_logk','code_plain','native_sft']
              for rate,lr in [('2e-5',2e-5),('1e-4',1e-4)]]


def mark(status, **fields):
    result = {'status':status,'time':time.time(),**fields}
    print(json.dumps(result),flush=True)
    if Path('/content').exists():
        p = Path('/content/recipe-status.json');tmp=p.with_suffix('.tmp')
        tmp.write_text(json.dumps(result));tmp.replace(p)
    return result


def upload_model(folder, name):
    api = HfApi()
    for attempt in range(4):
        try:
            api.upload_folder(repo_id=MODEL_REPO,folder_path=str(folder),
                path_in_repo='runs/'+name,commit_message='Bounded Gemma recipe diagnostic: '+name)
            return api.model_info(MODEL_REPO).sha
        except Exception:
            if attempt==3:raise
            time.sleep(10*(attempt+1))


def upload_dataset(path, target):
    for attempt in range(4):
        try:
            HfApi().upload_file(repo_id=DATASET_REPO,repo_type='dataset',
                path_or_fileobj=str(path),path_in_repo=target,
                commit_message='Gemma development-only recipe diagnostic')
            return
        except Exception:
            if attempt==3:raise
            time.sleep(10*(attempt+1))


def accepted_loss(logits, indices):
    """Singleton case is ordinary next-token CE; multiple native ties are valid."""
    return (torch.logsumexp(logits.float(),dim=-1)
            -torch.logsumexp(logits[:,indices].float(),dim=-1)).mean()


class RecipeModel(torch.nn.Module):
    def __init__(self, architecture, checkpoint=None):
        super().__init__()
        self.architecture=architecture
        native=architecture=='native_sft'
        spec=SPECS['gemma']
        if checkpoint is not None:
            saved=json.loads((Path(checkpoint)/'recipe_model.json').read_text())
            assert saved['architecture']==architecture and saved['base']==spec
        if native:
            self.network=Gemma4ForConditionalGeneration.from_pretrained(
                spec['repo'],revision=spec['revision'],dtype=torch.bfloat16,
                attn_implementation='sdpa',device_map={'':'cuda'})
            self.network.requires_grad_(False)
            self.tokenizer=AutoTokenizer.from_pretrained(spec['repo'],revision=spec['revision'])
        else:
            self.network=CodeDecisionModel.pretrained('gemma',lora=False)
            self.tokenizer=self.network.tokenizer
        if checkpoint is None:
            # Model construction/readout allocation must not change adapter initialization.
            torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
            self.body.language_model=get_peft_model(self.body.language_model,
                LoraConfig(r=16,lora_alpha=32,lora_dropout=.05,bias='none',target_modules=TARGETS))
        else:
            self.body.language_model=PeftModel.from_pretrained(self.body.language_model,
                Path(checkpoint)/'adapter',is_trainable=True)
            if not native:
                from safetensors.torch import load_file
                self.network.readout.load_state_dict(load_file(str(Path(checkpoint)/'readout.safetensors')))
        for name,p in self.body.named_parameters():
            p.requires_grad_('lora_' in name)
        self.body.language_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={'use_reentrant':False})
        self.tokenizer.padding_side='left'
        if self.tokenizer.pad_token_id is None:self.tokenizer.pad_token=self.tokenizer.eos_token
        ids=[self.tokenizer.encode(code,add_special_tokens=False) for code in CODES]
        assert all(len(x)==1 for x in ids)
        self.answer_ids=[x[0] for x in ids]

    @property
    def body(self):
        return self.network.model if self.architecture=='native_sft' else self.network.backbone

    def adapter_fingerprint(self):
        value=hashlib.sha256()
        for name,p in self.body.language_model.named_parameters():
            if 'lora_' in name:
                value.update(name.encode());value.update(p.detach().float().cpu().numpy().tobytes())
        return value.hexdigest()

    def prepare(self,row):
        text=self.tokenizer.apply_chat_template(messages(row),tokenize=False,
                    add_generation_prompt=True,enable_thinking=False)
        batch=self.tokenizer(text,add_special_tokens=False,return_tensors='pt',
                            return_token_type_ids=False,truncation=False)
        if batch['input_ids'].shape[1]>8192:raise ValueError('No truncation; over8192 tokens')
        return {k:v.to('cuda') for k,v in batch.items()}

    def forward(self,row,need_loss=False):
        keys=labels(row);inputs=self.prepare(row);details={}
        accepted_positions=[keys.index(k) for k in accepted(row)]
        if self.architecture=='native_sft':
            # Last prompt position predicts the single answer token. This avoids
            # materializing prompt-length times full-vocabulary logits.
            full=self.network(**inputs,use_cache=False,logits_to_keep=1).logits[:,-1].float()
            code=full[:,self.answer_ids[:len(keys)]]
            if need_loss:
                loss=accepted_loss(full,[self.answer_ids[i] for i in accepted_positions])
            else:loss=None
            details={'allowed_code_probability_mass':float(torch.exp(
                torch.logsumexp(code.detach(),-1)-torch.logsumexp(full.detach(),-1)).item()),
                'unconstrained_top1_is_allowed_code':bool(full.detach().argmax(-1).item()
                                                        in self.answer_ids[:len(keys)])}
        else:
            inputs['_counts']=torch.tensor([len(keys)],device='cuda')
            code=self.network(inputs)[:,:len(keys)]
            loss=accepted_loss(code,accepted_positions) if need_loss else None
            if need_loss and self.architecture=='code_logk':loss=loss/math.log(len(keys))
        return code,loss,details

    def save(self,path):
        path=Path(path);path.mkdir(parents=True,exist_ok=True)
        self.body.language_model.save_pretrained(path/'adapter')
        if self.architecture!='native_sft':
            from safetensors.torch import save_file
            save_file({'weight':self.network.readout.weight.detach().cpu().contiguous()},
                      str(path/'readout.safetensors'))
        self.tokenizer.save_pretrained(path/'tokenizer')
        (path/'recipe_model.json').write_text(json.dumps({'architecture':self.architecture,
            'base':SPECS['gemma'],'codes':CODES,'temperature':1.,'max_tokens':8192},indent=2)+'\n')


@torch.inference_mode()
def infer(model,rows):
    model.eval();predictions={}
    for row in sorted(rows,key=lambda r:r['token_lengths']['gemma']):
        values,_,details=model(row)
        probability=softmax(values[0].float().cpu().numpy())
        assert np.isfinite(probability).all()
        keys=labels(row)
        predictions[row['id']]={'id':row['id'],'family':row['family'],'task':row['task'],
            'target':row['target'],'labels':keys,'input_sha256':digest(row),'status':'ok',
            'prediction':keys[int(probability.argmax())],'probabilities':probability.tolist(),**details}
    return predictions


def assess(rows,predictions):
    result=metrics(rows,predictions)
    financial=[r for r in rows if r['task']=='financial_verification']
    result['financial_pairs']=financial_pairs(financial,predictions)
    result['financial_pairs']['both_correct_fraction']=(
        result['financial_pairs'].get('both_correct_pairs',0)/result['financial_pairs']['requested_pairs'])
    result['accuracy']=sum(predictions[r['id']]['prediction'] in accepted(r) for r in rows)/len(rows)
    mass=[p['allowed_code_probability_mass'] for p in predictions.values() if 'allowed_code_probability_mass' in p]
    if mass:
        result['native_answer_format']={'mean_allowed_code_probability_mass':float(np.mean(mass)),
            'unconstrained_top1_allowed_rate':float(np.mean([
                p['unconstrained_top1_is_allowed_code'] for p in predictions.values()]))}
    return result


def control_report(rows,predictions,parents):
    groups=defaultdict(list)
    for row in rows:groups[row['metadata']['control']['kind']].append(row)
    result={}
    for kind,part in groups.items():
        hits=consistent=0
        for row in part:
            p=predictions[row['id']]['prediction'];control=row['metadata']['control']
            hits+=p in accepted(row)
            consistent+=control['to_parent_label'][p]==parents[control['parent_id']]['prediction']
        result[kind]={'requested':len(part),'accuracy':hits/len(part),
                      'semantic_prediction_consistency':consistent/len(part)}
    return result


def memorized(score):
    return (score['accuracy']>=.97 and score['financial_pairs']['both_correct_fraction']>=.9
            and score.get('native_answer_format',{}).get('unconstrained_top1_allowed_rate',1.)>=.95)


def run_arm(name,architecture,lr,rows,development,controls,phase,manifest_sha,protocol):
    folder=OUT/name;folder.mkdir(parents=True,exist_ok=True)
    if (folder/'summary.json').exists():
        if not (folder/'artifact.json').exists():
            revision=upload_model(folder,name)
            (folder/'artifact.json').write_text(json.dumps({'repo':MODEL_REPO,'revision':revision,
                'path':'runs/'+name+'/selected','manifest_sha256':manifest_sha},indent=2)+'\n')
        upload_dataset(folder/'summary.json','summaries/'+name+'.json')
        upload_dataset(folder/'artifact.json','artifacts/'+name+'.json')
        return json.loads((folder/'summary.json').read_text())
    maximum=protocol['memorization' if phase=='memorization' else 'pilot']['max_updates_per_arm']
    interval=protocol['memorization' if phase=='memorization' else 'pilot']['evaluate_every']
    config={'name':name,'architecture':architecture,'lr':lr,'phase':phase,
        'manifest_sha256':manifest_sha,'maximum_updates':maximum,'effective_batch':8,
        'protocol_sha256':sha(Path('data/recipe-audit-protocol.json').read_bytes()),
        'code_sha256':sha(Path(__file__).read_bytes()),'seed':SEED,'torch':torch.__version__}
    config_path=folder/'config.json'
    if config_path.exists():assert json.loads(config_path.read_text())==config
    config_path.write_text(json.dumps(config,indent=2)+'\n')
    state_path=folder/'checkpoint/state.pt';resume=state_path.exists()
    mark('loading_model',arm=name,phase=phase,resume=resume)
    model=RecipeModel(architecture,folder/'checkpoint' if resume else None)
    fingerprint=model.adapter_fingerprint()
    initialization=folder/'initialization.json'
    if resume:
        fingerprint=json.loads(initialization.read_text())['adapter_initialization_sha256']
    else:
        initialization.write_text(json.dumps({'adapter_initialization_sha256':fingerprint})+'\n')
    groups=[{'params':[p for p in model.body.parameters() if p.requires_grad],'lr':lr}]
    if architecture!='native_sft':groups.append({'params':model.network.readout.parameters(),'lr':2.5*lr})
    optimizer=torch.optim.AdamW(groups,weight_decay=.01)
    # Actual longest-example backward path, with no optimizer update or RNG drift.
    cpu_rng=torch.get_rng_state();cuda_rng=torch.cuda.get_rng_state_all()
    model.train();_,probe,_=model(max(rows,key=lambda r:r['token_lengths']['gemma']),need_loss=True)
    probe.backward();grads=[p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert any(p.grad is not None and p.grad.count_nonzero()>0 for n,p in model.body.named_parameters() if 'lora_B' in n)
    optimizer.zero_grad(set_to_none=True);torch.set_rng_state(cpu_rng);torch.cuda.set_rng_state_all(cuda_rng)
    mark('gradient_preflight_passed',arm=name,phase=phase,gradient_tensors=len(grads))
    selected=0;best=-math.inf;start=0;history=[]
    eval_rows=rows if phase=='memorization' else development
    if resume:
        state=torch.load(state_path,map_location='cpu',weights_only=True)
        optimizer.load_state_dict(state['optimizer']);start=state['step'];selected=state['selected'];best=state['best'];history=state['history']
        torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state_all(state['cuda_rng'])
    else:
        predictions=infer(model,eval_rows);score=assess(eval_rows,predictions)
        history.append({'step':0,'metrics':score});best=score['accuracy'] if phase=='memorization' else score['selection_score']
        model.save(folder/'selected')
        (folder/'initial-predictions.jsonl').write_text(''.join(json.dumps(p)+'\n' for p in predictions.values()))
        mark('initial_validation_complete',arm=name,phase=phase,selection_score=best)
    task_counts=Counter(r['task'] for r in rows)
    domain_tasks={d:{r['task'] for r in rows if r['domain']==d} for d in {r['domain'] for r in rows}}
    orders={};wall=time.monotonic();last=start;passed=False
    for step in range(start+1,maximum+1):
        model.train();optimizer.zero_grad(set_to_none=True);batch_losses=[]
        warmup=8 if phase=='memorization' else 32
        rate=min(1.,step/warmup)
        if phase!='memorization':rate*=.1+.9*.5*(1+math.cos(math.pi*step/maximum))
        for i,group in enumerate(optimizer.param_groups):group['lr']=lr*rate*(2.5 if i else 1.)
        for position in range((step-1)*8,step*8):
            epoch,index=divmod(position,len(rows))
            if epoch not in orders:
                order=list(range(len(rows)));random.Random(SEED+epoch).shuffle(order);orders[epoch]=order
            row=rows[orders[epoch][index]]
            if phase!='memorization':row=permuted(row,position)
            _,loss,_=model(row,need_loss=True)
            if phase!='memorization':loss=loss*len(rows)/(4*len(domain_tasks[row['domain']])*task_counts[row['task']])
            assert torch.isfinite(loss)
            (loss/8).backward();batch_losses.append(float(loss.detach()))
        gradnorm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        optimizer.step();last=step
        if step==1 or step%16==0:
            mark('training',arm=name,phase=phase,step=step,total_steps=maximum,
                loss=float(np.mean(batch_losses)),gradient_norm=float(gradnorm),wall_s=time.monotonic()-wall)
        if step%interval==0 or step==maximum:
            mark('validating',arm=name,phase=phase,step=step)
            predictions=infer(model,eval_rows);score=assess(eval_rows,predictions)
            value=score['accuracy'] if phase=='memorization' else score['selection_score']
            history.append({'step':step,'metrics':score})
            if value>best+1e-8:
                selected=step;best=value;model.save(folder/'selected')
            passed=phase=='memorization' and memorized(score)
            # A passing memorization fit must be the retained diagnostic checkpoint.
            if passed:
                selected=step;best=value;model.save(folder/'selected')
            model.save(folder/'checkpoint')
            torch.save({'step':step,'selected':selected,'best':best,'history':history,
                'optimizer':optimizer.state_dict(),'torch_rng':torch.get_rng_state(),
                'cuda_rng':torch.cuda.get_rng_state_all()},state_path)
            (folder/'history.json').write_text(json.dumps(history,indent=2,allow_nan=False)+'\n')
            upload_model(folder,name)
            mark('checkpointed',arm=name,phase=phase,step=step,selected_step=selected,
                selection_score=value,memorization_passed=passed)
            if passed:break
    del model,optimizer;torch.cuda.empty_cache()
    model=RecipeModel(architecture,folder/'selected')
    predictions=infer(model,eval_rows);score=assess(eval_rows,predictions)
    summary={'status':'complete','name':name,'architecture':architecture,'lr':lr,'phase':phase,
        'selected_step':selected,'last_step':last,'maximum_updates':maximum,'metrics':score,
        'manifest_sha256':manifest_sha,'adapter_initialization_sha256':fingerprint,
        'test_cases_accessed':False,'history':history}
    if phase=='memorization':summary['memorization_passed']=memorized(score)
    else:
        control_predictions=infer(model,controls)
        summary['controls']=control_report(controls,control_predictions,predictions)
        (folder/'control-predictions.jsonl').write_text(''.join(json.dumps(p)+'\n' for p in control_predictions.values()))
        m=score['tasks'];f=score['financial_pairs']
        summary['acceptance_diagnostics']={
            'financial_auc_at_least_0_7':f['roc_auc_valid']>=.7,
            'financial_pairs_both_correct_at_least_half':f['both_correct_fraction']>=.5,
            'costaware_utility_beats_fixed_training_router':m['routing_costaware']['routing_valid_only']['utility']>m['routing_costaware']['training_profile_router_valid_only']['utility'],
            'safety_tradeoffs_reported_separately':True}
    (folder/'selected-predictions.jsonl').write_text(''.join(json.dumps(p)+'\n' for p in predictions.values()))
    (folder/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    revision=upload_model(folder,name)
    upload_dataset(folder/'summary.json','summaries/'+name+'.json')
    artifact={'repo':MODEL_REPO,'revision':revision,'path':'runs/'+name+'/selected',
              'manifest_sha256':manifest_sha}
    (folder/'artifact.json').write_text(json.dumps(artifact,indent=2)+'\n')
    upload_dataset(folder/'artifact.json','artifacts/'+name+'.json')
    del model;torch.cuda.empty_cache()
    mark('arm_complete',arm=name,phase=phase,selected_step=selected,
         memorization_passed=summary.get('memorization_passed'))
    return summary


def main():
    import fcntl
    OUT.mkdir(parents=True,exist_ok=True)
    lock=(OUT/'job.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    torch.set_num_threads(8);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cuda.enable_flash_sdp(True);torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(True);torch.backends.cuda.enable_cudnn_sdp(False)
    manifest_path=Path('data/recipe-audit-manifest.json');manifest=json.loads(manifest_path.read_text())
    assert manifest['status']=='sealed_before_fitting'
    assert sha(Path('data/recipe-audit-protocol.json').read_bytes())==manifest['protocol_sha256']
    assert sha(Path('data/phase4-answer-codes.json').read_bytes())==manifest['answer_vocabulary_sha256']
    for name,value in manifest['code_sha256'].items():assert sha(Path(name).read_bytes())==value
    splits={}
    for split,entry in manifest['splits'].items():
        path=DATA/(split+'.jsonl');assert sha(path.read_bytes())==entry['sha256'];splits[split]=read(path)
    assert {r['family'] for r in splits['train']}.isdisjoint({r['family'] for r in splits['development']})
    assert {r['id'] for r in splits['memorization']}<={r['id'] for r in splits['train']}
    for forbidden in ['data/private/phase4/core_test.jsonl','data/private/phase4/transfer_test.jsonl']:
        if Path('/content').exists():assert not Path(forbidden).exists()
    api=HfApi();assert api.model_info(MODEL_REPO).private and api.dataset_info(DATASET_REPO).private
    if not any(OUT.glob('*/config.json')):
        files=api.list_repo_files(MODEL_REPO)
        if any(name.startswith('runs/') for name in files):
            import shutil
            source=Path(snapshot_download(MODEL_REPO,allow_patterns=['runs/**']))/'runs'
            shutil.copytree(source,OUT,dirs_exist_ok=True)
    protocol=json.loads(Path('data/recipe-audit-protocol.json').read_text())
    config_hash=sha(manifest_path.read_bytes());summaries={}
    for name,arch,lr in MEMORY_ARMS:
        summaries[name]=run_arm(name,arch,lr,splits['memorization'],[],[],
                                'memorization',config_hash,protocol)
    passed=any(summaries[name]['memorization_passed'] for name in ['mem_code_lr1e-4','mem_sft_lr1e-4'])
    if passed:
        for name,arch,lr in PILOT_ARMS:
            summaries[name]=run_arm(name,arch,lr,splits['train'],splits['development'],
                                   splits['controls'],'pilot',config_hash,protocol)
    result={'status':'complete' if passed else 'blocked_memorization','manifest_sha256':config_hash,
        'memorization_gate_passed':passed,'completed_arms':list(summaries),'expected_arms':9 if passed else 3,
        'gpu':torch.cuda.get_device_name(),'test_cases_accessed':False}
    (OUT/'job-summary.json').write_text(json.dumps(result,indent=2)+'\n')
    upload_dataset(OUT/'job-summary.json','job-summary.json')
    mark('pipeline_complete',**{k:v for k,v in result.items() if k!='status'},outcome=result['status'])


if __name__=='__main__':
    main()
