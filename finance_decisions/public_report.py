"""Download private predictions, verify complete matching, and publish aggregates."""
import argparse,hashlib,json,os,subprocess,time
from collections import defaultdict
from pathlib import Path
import numpy as np
from huggingface_hub import HfApi,hf_hub_download
from .public_eval import REPO,data,digest,labels,read
from .public_metrics import correct,score

NAMES=['causal','joint','qwen','jev']


def paired(rows,a,b):
    groups=defaultdict(list)
    for row in rows:
        x,y=a.get(row['id']),b.get(row['id'])
        if not x or not y or x.get('status')!='ok' or y.get('status')!='ok':continue
        expected=digest(row)
        if x['input_sha256']!=expected or y['input_sha256']!=expected or x['labels']!=labels(row) or y['labels']!=labels(row):
            raise ValueError('Input/candidate mismatch in paired comparison')
        groups[row['family']].append(int(correct(row,y['prediction']))-int(correct(row,x['prediction'])))
    if not groups:return {'common_valid':0}
    sums=np.array([sum(v) for v in groups.values()]);counts=np.array([len(v) for v in groups.values()]);rng=np.random.default_rng(20261109)
    samples=[]
    for _ in range(2000):
        idx=rng.integers(0,len(groups),len(groups));samples.append(sums[idx].sum()/counts[idx].sum())
    return {'common_valid':int(counts.sum()),'groups':len(groups),'joint_minus_causal_accuracy':float(sums.sum()/counts.sum()),
        'family_bootstrap_95pct':np.quantile(samples,[.025,.975]).tolist(),'limits':'Exploratory paired source-family intervals; public data contamination and related templates limit inference.'}


def download(api,revision):
    root=Path('results/public-eval');root.mkdir(parents=True,exist_ok=True);allrecords={};summaries={}
    for name in NAMES:
        out=root/name;out.mkdir(exist_ok=True)
        for filename in ['summary.json','config.json','predictions.jsonl']:
            try:
                p=hf_hub_download(REPO,f'runs/{name}/{filename}',repo_type='dataset',revision=revision)
                (out/filename).write_bytes(Path(p).read_bytes())
            except Exception:
                if filename=='predictions.jsonl':allrecords[name]=[]
        if (out/'predictions.jsonl').exists():allrecords[name]=read(out/'predictions.jsonl')
        if (out/'summary.json').exists():summaries[name]=json.loads((out/'summary.json').read_text())
    return allrecords,summaries


def render(rows,records,summaries,revision):
    root=Path('results/public-eval');allmetrics={name:score(rows,records.get(name,[])) for name in NAMES}
    byname={n:{p['id']:p for p in records.get(n,[])} for n in NAMES}
    common={};pairs={}
    for suite in sorted({r['suite'] for r in rows}):
        subset=[r for r in rows if r['suite']==suite]
        valid=[r for r in subset if all(byname[n].get(r['id'],{}).get('status')=='ok' for n in NAMES)]
        for r in valid:
            for n in NAMES:
                p=byname[n][r['id']]
                if p['input_sha256']!=digest(r) or p['labels']!=labels(r) or p['target']!=r['target']:raise ValueError('Invalid matched prediction')
        common[suite]={'n':len(valid),'models':{n:float(np.mean([correct(r,byname[n][r['id']]['prediction']) for r in valid])) if valid else None for n in NAMES}}
        pairs[suite]=paired(subset,byname['causal'],byname['joint'])
    payload={'models':allmetrics,'common_valid':common,'paired_gemma':pairs,'artifact_revision':revision,
        'case_counts':{n:len(records.get(n,[])) for n in NAMES},'note':'Per-task static evaluation, not a whole Decision Index score. No training or temperature fitting.'}
    (root/'summary.json').write_text(json.dumps(payload,indent=2)+'\n')
    artifacts={'repo':REPO,'repo_type':'dataset','revision':revision,'private':True,'models':{}}
    for n in NAMES:
        artifacts['models'][n]={'config':summaries.get(n,{}).get('config'),
            'predictions_sha256':hashlib.sha256((root/n/'predictions.jsonl').read_bytes()).hexdigest() if (root/n/'predictions.jsonl').exists() else None}
    (root/'artifacts.json').write_text(json.dumps(artifacts,indent=2)+'\n')
    complete=all(len(records.get(n,[]))==len(rows) for n in NAMES)
    lines=['# Public decision evaluation','',f"Status: {'complete' if complete else 'partial; see coverage and worker status'}. Frozen corpus: {len(rows):,} decisions. No training, calibration fitting, input truncation or candidate reduction.",'',
        'Gemma and Qwen use strict math SDPA with frozen checkpoints and temperatures. The Gemma inference envelope is extended to eleven candidates and 8,192 tokens; this exceeds the original release validation envelope. Each model gets its original prompt and actual tokenizer. Unsupported or failed cases remain in the denominator below.','',
        '| Suite | Cases | Causal Gemma | Joint Gemma | Qwen 27B | Jev |','|---|---:|---:|---:|---:|---:|']
    suites=sorted(allmetrics['causal'])
    for suite in suites:
        values=[]
        for n in NAMES:
            m=allmetrics[n][suite]
            values.append(f"{m['accuracy_missing_as_wrong']:.2%} ({m['coverage']:.1%})" if m['valid'] else 'N/A (0% coverage)')
        lines.append(f"| {suite} | {allmetrics['causal'][suite]['requested']} | "+' | '.join(values)+' |')
    lines+=['','Entries show accuracy with missing predictions counted wrong, followed by coverage in parentheses. RouterBench entries are tie-aware oracle-optimal rates; realized quality, cost, utility and regret are the primary routing results in the detailed JSON. Different tasks should not be collapsed into one general-quality rank.','',
        '## Identical common-valid inputs','', '| Suite | Common cases | Causal | Joint | Qwen | Jev |','|---|---:|---:|---:|---:|---:|']
    for suite,m in common.items():
        vals=[f"{m['models'][n]:.2%}" if m['models'][n] is not None else 'N/A' for n in NAMES]
        lines.append(f"| {suite} | {m['n']} | "+' | '.join(vals)+' |')
    providers=summaries.get('jev',{}).get('resolved_providers',{})
    lines+=['',f"Resolved Jev providers: `{json.dumps(providers,sort_keys=True)}`.",'',
        'See [full metrics](summary.json) for valid-only classification F1, safety misses/false blocks under both severity conventions, binary positive-class scores, contract document-exact match, routing outcomes and exploratory paired Gemma intervals. API failures, over-context inputs and OOM failures are explicitly recorded.','',
        '## Reproduction and limits','',
        '- [Frozen protocol](../../PUBLIC_EVAL_PROTOCOL.md) and [data manifest](../../data/public-eval-manifest.json).','- [Pinned private output artifacts](artifacts.json).','- [Private predictions](https://huggingface.co/datasets/jsaurabh/financial-decision-public-eval).','',
        'RouterBench is a hash-selected initial subset, preserving both shot regimes/objectives and all eleven candidates. Other named source test records are retained. R-Judge contains 571 records at the pinned revision although its paper states 569. ContractNLI has seventeen decisions per contract. TS-Bench and R-Judge here evaluate static judgments, not end-to-end agent behavior.','',
        'The published public sources may have appeared in pretraining or earlier upstream fine-tuning; Qwen has previously been evaluated on Decision Index. This is not a new contamination-free or private production benchmark. Raw source contexts are not republished. Source-family overlap limits the apparent number of independent examples.']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')
    (root/'README.md').write_text('# Public decision evaluation\n\n'+('Completed' if complete else 'Partial')+' aggregate results are in [REPORT.md](REPORT.md). Per-case predictions remain in the [private output repository](https://huggingface.co/datasets/jsaurabh/financial-decision-public-eval). See the report for coverage, failures, inference extensions and limitations.\n')
    return complete


def main():
    p=argparse.ArgumentParser();p.add_argument('--watch',action='store_true');p.add_argument('--push',action='store_true');args=p.parse_args()
    rows,manifest=data();api=HfApi();start=time.time();poll_failures=0
    status_path=Path('/workspace/public-eval-control/report-status.json')
    status_path.parent.mkdir(parents=True,exist_ok=True)
    while True:
        try:
            info=api.dataset_info(REPO);assert info.private
            records,summaries=download(api,info.sha)
            control=Path('/workspace/public-eval-control/status.json')
            state=json.loads(control.read_text()) if control.exists() else {}
        except AssertionError:
            raise
        except Exception as exc:
            poll_failures+=1
            status={'status':'retrying','error_type':type(exc).__name__,'consecutive_poll_failures':poll_failures,'time':time.time()}
            status_path.write_text(json.dumps(status,indent=2));print(json.dumps(status),flush=True)
            if not args.watch or time.time()-start>9*3600:raise
            time.sleep(45);continue
        poll_failures=0
        stopped=all(state.get('workers',{}).get(r,{}).get('stop_returncode')==0 for r in ['gemma','qwen'])
        finished=all(len(records.get(n,[]))==len(rows) for n in NAMES)
        failed=stopped and any(state.get('workers',{}).get(r,{}).get('status') in ['failed','timeout'] for r in ['gemma','qwen'])
        progress={'status':'waiting','record_counts':{n:len(records.get(n,[])) for n in NAMES},'gpu_workers_stopped':stopped,'time':time.time()}
        status_path.write_text(json.dumps(progress,indent=2));print(json.dumps(progress),flush=True)
        if not args.watch or (finished and stopped) or failed or time.time()-start>9*3600:break
        time.sleep(90)
    complete=render(rows,records,summaries,info.sha)
    root=Path('results/public-eval')
    for f in [root/'REPORT.md',root/'summary.json',root/'artifacts.json']:
        api.upload_file(repo_id=REPO,repo_type='dataset',path_or_fileobj=str(f),path_in_repo='report/'+f.name,commit_message='Aggregate public evaluation report')
    if args.push:
        paths=[str(f) for f in [root/'README.md',root/'REPORT.md',root/'summary.json',root/'artifacts.json']]+[str(f) for n in NAMES for f in (root/n).glob('*.json')]
        allowed=set(paths);staged=set(subprocess.check_output(['git','diff','--cached','--name-only'],text=True).splitlines())
        branch=subprocess.check_output(['git','branch','--show-current'],text=True).strip()
        if staged<=allowed and branch=='gemma4-finance':
            secrets=[os.environ[k].encode() for k in ['HF_TOKEN','GH_TOKEN','TYPESAFE_API_KEY','COLAB_ADC_JSON'] if os.environ.get(k)]
            assert not any(secret in Path(f).read_bytes() for f in paths for secret in secrets)
            subprocess.run(['git','add',*paths],check=True)
            if subprocess.run(['git','diff','--cached','--quiet']).returncode:
                subprocess.run(['git','commit','-m','Publish '+('completed' if complete else 'partial')+' public benchmark evaluation aggregates'],check=True)
                subprocess.run(['git','push','origin','HEAD:gemma4-finance'],check=True)
        else:print('Report saved; automatic git push skipped because branch/staged changes differ',flush=True)
    status_path.write_text(json.dumps({'status':'complete' if complete else 'partial','time':time.time(),'report':str(root/'REPORT.md')},indent=2))

if __name__=='__main__':main()
