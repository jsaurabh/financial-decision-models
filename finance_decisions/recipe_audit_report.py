"""Collect development diagnostics after the owned GPU is released."""
import argparse
import json
import subprocess
import time
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

ROOT=Path('/workspace/recipe-audit-control')
CACHE=ROOT/'report-cache'
OUT=Path('results/recipe-audit')
REPO='jsaurabh/financial-decision-gemma-recipe-audit-data'
ARMS=['mem_code_lr2e-5','mem_code_lr1e-4','mem_sft_lr1e-4']+[
      f'pilot_{arch}_lr{rate}' for arch in ['code_logk','code_plain','native_sft']
      for rate in ['2e-5','1e-4']]


def collect():
    CACHE.mkdir(parents=True,exist_ok=True)
    for attempt in range(4):
        try:
            api=HfApi();revision=api.dataset_info(REPO).sha
            files=api.list_repo_files(REPO,repo_type='dataset',revision=revision)
            wanted=['job-summary.json']+['summaries/'+arm+'.json' for arm in ARMS]+[
                                        'artifacts/'+arm+'.json' for arm in ARMS]
            for name in wanted:
                if name not in files:continue
                path=hf_hub_download(REPO,name,repo_type='dataset',revision=revision)
                target=CACHE/name;target.parent.mkdir(parents=True,exist_ok=True)
                tmp=target.with_suffix('.tmp');tmp.write_bytes(Path(path).read_bytes());tmp.replace(target)
            return
        except Exception:
            if attempt==3:raise
            time.sleep(5*(attempt+1))


def render(summaries,job,controller,artifacts):
    manifest_path=Path('data/recipe-audit-manifest.json')
    from hashlib import sha256
    manifest_sha=sha256(manifest_path.read_bytes()).hexdigest()
    for item in summaries.values():assert item['manifest_sha256']==manifest_sha
    report={'status':'complete' if job and len(summaries)==job['expected_arms'] else 'partial',
        'job':job,'controller':controller,'manifest_sha256':manifest_sha,
        'summaries':summaries,'artifacts':artifacts,
        'claim_scope':'Training/development recipe diagnostics only. No fresh final test or Jev superiority claim.',
        'deployment_winner_selected':False}
    lines=['# Gemma E2B training-recipe diagnostic','',
        f"Status: **{report['status']}**. Outcome: **{job.get('status','incomplete')}**.",'',
        'All arms start from the same pinned Gemma4 E2B instruction base. Training/development examples, adapter initialization, effective batches and candidate permutations are controlled. Native SFT uses the frozen full-vocabulary LM head; code variants train the small FP32 answer-code head.','',
        'These are development comparisons. Test cases were not loaded. A higher aggregate alone does not qualify a model for deployment or establish superiority over Jev.','',
        '## Memorization checks','',
        '| Arm | Selected update | Training accuracy | Financial pairs both correct | Native top-one allowed | Passed |',
        '|---|---:|---:|---:|---:|---|']
    for arm in ARMS[:3]:
        if arm not in summaries:continue
        s=summaries[arm];m=s['metrics'];native=m.get('native_answer_format',{}).get('unconstrained_top1_allowed_rate')
        lines.append(f"| {arm} | {s['selected_step']} | {m['accuracy']:.1%} | {m['financial_pairs']['both_correct_fraction']:.1%} | "+(f'{native:.1%}' if native is not None else 'n/a')+f" | {s['memorization_passed']} |")
    lines+=['','The64 cases are training examples. This checks optimization and binding, not generalization.','',
        '## Matched development pilots','',
        '| Arm | Selected update | Aggregate | Financial AUC | Financial pairs both correct | Cost-aware utility | Fixed-router utility |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for arm in ARMS[3:]:
        if arm not in summaries:continue
        s=summaries[arm];m=s['metrics'];f=m['financial_pairs'];r=m['tasks']['routing_costaware']
        lines.append(f"| {arm} | {s['selected_step']} | {m['selection_score']:.3f} | {f['roc_auc_valid']:.3f} | {f['both_correct_fraction']:.1%} | {r['routing_valid_only']['utility']:.3f} | {r['training_profile_router_valid_only']['utility']:.3f} |")
    lines+=['','## Safety and decision binding','',
        '| Arm | Action unsafe/review recall | Safe false blocks | Permutation consistency | Alias consistency | Predicate-reversal consistency |',
        '|---|---:|---:|---:|---:|---:|']
    for arm in ARMS[3:]:
        if arm not in summaries:continue
        s=summaries[arm];v=s['metrics']['tasks']['tool_action_safety']['strict_safety_valid_only'];c=s['controls']
        recall=v.get('unsafe_or_review_recall');fpr=v.get('safe_false_block_rate')
        values=[f'{x:.1%}' if x is not None else 'n/a' for x in [recall,fpr]]
        values += [f"{c[k]['semantic_prediction_consistency']:.1%}" for k in ['permutation','aliases','predicate_reversal']]
        lines.append('| '+arm+' | '+' | '.join(values)+' |')
    lines+=['','Full histories, per-task metrics, acceptance diagnostics and immutable artifact revisions are in [comparison.json](comparison.json).',
        'The [frozen protocol](../../data/recipe-audit-protocol.json) and [manifest](../../data/recipe-audit-manifest.json) describe the controls.','',
        'A near-chance financial AUC, low pair discrimination or utility below the fixed router remains a failure even if the aggregate rises. Preserve old benchmark receipts; subsequent work requires a separately sealed final holdout.']
    return report,'\n'.join(lines)+'\n'


def main():
    import fcntl
    parser=argparse.ArgumentParser();parser.add_argument('--watch',action='store_true');parser.add_argument('--push',action='store_true');args=parser.parse_args()
    ROOT.mkdir(exist_ok=True);lock=(ROOT/'report.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    start=time.time();controller={}
    while True:
        try:collect()
        except Exception:
            (ROOT/'report-status.json').write_text(json.dumps({'status':'collection_retry','time':time.time()}))
            if time.time()-start>5*3600:raise
            time.sleep(60);continue
        path=ROOT/'status.json';controller=json.loads(path.read_text()) if path.exists() else {}
        completed=[a for a in ARMS if (CACHE/'summaries'/f'{a}.json').exists()]
        (ROOT/'report-status.json').write_text(json.dumps({'status':'waiting','completed_arms':completed,
            'gpu_stopped':controller.get('stop_returncode')==0,'time':time.time()},indent=2))
        if controller.get('stop_returncode')==0 or not args.watch or time.time()-start>5*3600:break
        time.sleep(60)
    summaries={a:json.loads((CACHE/'summaries'/f'{a}.json').read_text()) for a in ARMS if (CACHE/'summaries'/f'{a}.json').exists()}
    artifacts={a:json.loads((CACHE/'artifacts'/f'{a}.json').read_text()) for a in ARMS if (CACHE/'artifacts'/f'{a}.json').exists()}
    path=CACHE/'job-summary.json';job=json.loads(path.read_text()) if path.exists() else {}
    report,markdown=render(summaries,job,controller,artifacts)
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'comparison.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    (OUT/'comparison.md').write_text(markdown)
    HfApi().upload_folder(repo_id=REPO,repo_type='dataset',folder_path=str(OUT),
                         path_in_repo='aggregate',commit_message='Gemma-only training recipe development comparisons')
    if args.push:
        paths=['results/recipe-audit/comparison.json','results/recipe-audit/comparison.md']
        subprocess.run(['git','add','--',*paths],check=True)
        if subprocess.run(['git','diff','--cached','--quiet','--',*paths]).returncode:
            subprocess.run(['git','commit','--only','-m','Report controlled Gemma E2B training recipe diagnostics','--',*paths],check=True)
            subprocess.run(['git','push','origin','HEAD'],check=True)
    (ROOT/'report-status.json').write_text(json.dumps({'status':report['status'],
        'completed_arms':list(summaries),'published':args.push,'time':time.time()},indent=2))


if __name__=='__main__':main()
