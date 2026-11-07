"""Publish matched core evidence and source/task transfer in separate tables."""
import argparse,json,subprocess,time
from collections import Counter
from pathlib import Path
import numpy as np
from huggingface_hub import HfApi,hf_hub_download
from huggingface_hub.errors import EntryNotFoundError
from .phase3_data import read,sha
from .phase4_eval import REPO,validate
from .phase4_metrics import metrics,correct
from .final_metrics import paired_cluster,holm,routing

CONTROL=Path('/workspace/phase4-control');CACHE=Path('data/private/phase4-report');OUT=Path('results/phase4')
SEED=20261107

def collect(role):
    folder=CACHE/role;folder.mkdir(parents=True,exist_ok=True)
    for attempt in range(4):
        try:
            revision=HfApi().dataset_info(REPO).sha
            for name in ['config.json','progress.json','predictions.jsonl']:
                try:
                    path=hf_hub_download(REPO,'predictions/'+role+'/'+name,repo_type='dataset',revision=revision,force_download=attempt>0)
                    tmp=(folder/name).with_suffix('.tmp');tmp.write_bytes(Path(path).read_bytes());tmp.replace(folder/name)
                except EntryNotFoundError:pass
            break
        except Exception:
            if attempt==3:return {'status':'collection_error'}
            time.sleep(5*(attempt+1))
    path=folder/'progress.json';return json.loads(path.read_text()) if path.exists() else {}

def classification_comparison(rows,a,b):
    part=[];weights={};tasks=sorted({r['task'] for r in rows})
    for task in tasks:
        task_rows=[r for r in rows if r['task']==task];units={r['source'].get('review_question','') for r in task_rows}
        for r in task_rows:
            q=dict(r);q['suite']=task+':'+r['source'].get('review_question','');part.append(q);weights[q['suite']]=1/(len(tasks)*len(units))
    delta=[int(correct(r,a.get(r['id'])))-int(correct(r,b.get(r['id']))) for r in part]
    return paired_cluster(part,delta,seed=SEED,task_weights=weights)

def publish(rows,manifest,controllers):
    systems={};byid={r['id']:r for r in rows};summaries={};artifacts={}
    for role in ['gemma','qwen','jev']:
        path=CACHE/role/'predictions.jsonl';pred={}
        if path.exists():
            for p in read(path):validate(byid[p['id']],p);assert p['id'] not in pred;pred[p['id']]=p
        systems[role]=pred
        cfg=CACHE/role/'config.json'
        if cfg.exists():assert json.loads(cfg.read_text())['manifest_sha256']==sha(Path('data/phase4-manifest.json').read_bytes())
        if role!='jev':
            try:
                p=hf_hub_download(REPO,'runs/'+role+'/artifact.json',repo_type='dataset');artifacts[role]=json.loads(Path(p).read_text())
                q=hf_hub_download(artifacts[role]['repo'],'run/summary.json',revision=artifacts[role]['revision']);summaries[role]=json.loads(Path(q).read_text())
            except EntryNotFoundError:pass
    winner=max(summaries,key=lambda role:(summaries[role]['validation']['selection_score'],role=='gemma')) if len(summaries)==2 else None
    cohorts={cohort:{role:metrics([r for r in rows if r['cohort']==cohort],pred) for role,pred in systems.items()} for cohort in ['core','transfer']}
    comparisons={}
    core=[r for r in rows if r['cohort']=='core']
    for role in ['gemma','qwen']:
        result={};pvalues={}
        for domain in ['financial','legal','tool_safety']:
            part=[r for r in core if r['domain']==domain];m=classification_comparison(part,systems[role],systems['jev']);result[domain]=m;pvalues[domain]=m.get('p_two_sided',1.)
        for task,field in [('routing_quality','quality'),('routing_costaware','utility')]:
            part=[r for r in core if r['task']==task]
            # The shared helper's two system labels name the candidate and Jev.
            m=routing(part,systems[role],systems['jev'],seed=SEED);result[task]=m
            pvalues[task]=m.get('paired_differences',{}).get(field,{}).get('p_two_sided',1.)
        adjusted=holm(pvalues);gates={}
        for name,m in result.items():
            field='quality' if name=='routing_quality' else 'utility'
            effect=m.get('paired_differences',{}).get(field,{}) if name.startswith('routing_') else m
            ci=effect.get('ci95');part=[r for r in core if r['task']==name] if name.startswith('routing_') else [r for r in core if r['domain']==name]
            coverage_a=sum(systems[role].get(r['id'],{}).get('status')=='ok' for r in part);coverage_b=sum(systems['jev'].get(r['id'],{}).get('status')=='ok' for r in part)
            gates[name]=bool(ci and ci[0]>0 and adjusted[name]<.05 and coverage_a>=coverage_b)
        f1_ok=all(cohorts['core'][role]['tasks'][t].get('semantic_macro_f1_coverage_penalized',0)>=cohorts['core']['jev']['tasks'][t].get('semantic_macro_f1_coverage_penalized',0) for t in cohorts['core'][role]['tasks'] if not t.startswith('routing_'))
        ga=cohorts['core'][role]['tasks']['tool_action_safety'].get('strict_safety_valid_only',{});jb=cohorts['core']['jev']['tasks']['tool_action_safety'].get('strict_safety_valid_only',{})
        safety_ok=bool(ga and jb and ga.get('safe_false_block_rate') is not None and jb.get('safe_false_block_rate') is not None and ga['safe_false_block_rate']<=jb['safe_false_block_rate'] and ga['unsafe_or_review_recall']>=jb['unsafe_or_review_recall'])
        for task in ['prompt_injection','malicious_request']:
            a=cohorts['core'][role]['tasks'][task].get('binary_valid_only',{});b=cohorts['core']['jev']['tasks'][task].get('binary_valid_only',{})
            safety_ok=safety_ok and bool(a and b and a.get('recall') is not None and b.get('recall') is not None and a.get('false_positive_rate') is not None and b.get('false_positive_rate') is not None and a['recall']>=b['recall'] and a['false_positive_rate']<=b['false_positive_rate'])
        stable_jev=all(systems['jev'].get(r['id'],{}).get('status')=='ok' and systems['jev'][r['id']].get('provider')=='jev-1.14.0' for r in core)
        result.update(holm_p=adjusted,positive_primary_gates=gates,all_classification_f1_no_worse=f1_ok,native_safety_tradeoff_no_worse=safety_ok,
            stable_complete_jev_core_reference=stable_jev,represented_core_superiority_supported=all(gates.values()) and f1_ok and safety_ok and stable_jev)
        comparisons[role]=result
    report={'status':'complete' if all(len(p)==len(rows) for p in systems.values()) else 'partial','manifest_sha256':sha(Path('data/phase4-manifest.json').read_bytes()),
        'primary_candidate_selected_by_full_validation':winner,'training_summaries':summaries,'artifacts':artifacts,'cohorts':cohorts,'core_paired_comparisons':comparisons,
        'receipts':{role:len(p) for role,p in systems.items()},'jev_resolved_providers':dict(Counter(p.get('provider','missing') for p in systems['jev'].values())),
        'controllers':controllers,'historical_exposure':manifest['historical_exposure'],'transfer_exposure':manifest['transfer_test']['historical_exposure'],'limitations':manifest['limitations'],
        'claim_scope':'Only represented core capabilities with recorded matched splits. Transfer results are separate. No production financial-services claim; public pretraining exposure unknown.'}
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'comparison.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    lines=['# Aligned four-domain phase4 comparison','',f"Status: **{report['status']}**. Primary candidate selected using full validation only: **{winner or 'unavailable'}**.",'',
        'All core tasks were represented in training, validation, calibration and core test. Source/task transfer is separate and never used for fitting or selection.','']
    for cohort in ['core','transfer']:
        lines += ['## '+cohort,'','| Task | Requested | Gemma accuracy | Qwen accuracy | Jev accuracy | Coverage G/Q/J |','|---|---:|---:|---:|---:|---|']
        tasks=cohorts[cohort]['jev']['tasks']
        for task in tasks:
            ms=[cohorts[cohort][role]['tasks'][task] for role in ['gemma','qwen','jev']]
            lines.append(f"| {task} | {ms[0]['requested']} | "+' | '.join(f"{m['accuracy_missing_as_wrong']:.1%}" for m in ms)+' | '+' / '.join(f"{m['coverage']:.1%}" for m in ms)+' |')
        lines+=['','Classification entries count missing decisions wrong. Routing accuracy is only tie-aware oracle-choice agreement; actual quality/cost/utility/regret and training-profile baselines are in the JSON and are the routing outcome metrics.']
    lines+=['','## Claim gates','']
    for role,result in comparisons.items():lines.append(f"- {role}: represented-core superiority gate **{'passed' if result['represented_core_superiority_supported'] else 'not passed'}**. Primary effects, Holm correction, F1 retention, safety tradeoffs and coverage are in the JSON.")
    lines+=['','## Provenance and limitations','',*['- '+x for x in manifest['limitations']],'- '+manifest['historical_exposure'],'- '+manifest['transfer_test']['historical_exposure'],
        '', 'See [full results](comparison.json), [training protocol](../../PHASE4_TRAINING_PROTOCOL.md) and [frozen manifest](../../data/phase4-manifest.json). Per-case outputs and model weights are private.']
    (OUT/'comparison.md').write_text('\n'.join(lines)+'\n');HfApi().upload_folder(repo_id=REPO,repo_type='dataset',folder_path=str(OUT),path_in_repo='aggregate',commit_message='Aligned core test and separate transfer results, including all losses')
    return report

def main():
    import fcntl
    parser=argparse.ArgumentParser();parser.add_argument('--watch',action='store_true');parser.add_argument('--push',action='store_true');args=parser.parse_args()
    CONTROL.mkdir(exist_ok=True);lock=(CONTROL/'report.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    manifest=json.loads(Path('data/phase4-manifest.json').read_text());rows=read('data/private/phase4/core_test.jsonl')+read('data/private/phase4/transfer_test.jsonl');started=time.time()
    while True:
        progress={role:collect(role) for role in ['gemma','qwen','jev']};controllers={}
        for role in ['gemma','qwen']:
            path=CONTROL/role/'status.json';controllers[role]=json.loads(path.read_text()) if path.exists() else {}
        stopped=all(x.get('stop_returncode')==0 for x in controllers.values());cpu=(CONTROL/'jev-terminal.json').exists();complete=all(x.get('completed',0)==len(rows) for x in progress.values())
        (CONTROL/'report-status.json').write_text(json.dumps({'status':'waiting','counts':{r:p.get('completed',0) for r,p in progress.items()},'gpu_workers_stopped':stopped,'time':time.time()},indent=2))
        if (stopped and cpu) or time.time()-started>9*3600 or not args.watch:break
        time.sleep(60)
    report=publish(rows,manifest,controllers)
    if args.push:
        paths=['results/phase4/comparison.json','results/phase4/comparison.md'];subprocess.run(['git','add','--',*paths],check=True)
        if subprocess.run(['git','diff','--cached','--quiet','--',*paths]).returncode:
            subprocess.run(['git','commit','--only','-m','Report aligned phase4 core and separate transfer evaluations','--',*paths],check=True)
            subprocess.run(['git','push','origin','HEAD'],check=True)
    (CONTROL/'report-status.json').write_text(json.dumps({'status':report['status'],'receipts':report['receipts'],'published':args.push,'time':time.time()},indent=2))

if __name__=='__main__':main()
