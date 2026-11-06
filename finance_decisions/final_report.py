"""Collect into a separate cache, publish final/partial aggregate evidence."""
import argparse, hashlib, json, subprocess, time
from pathlib import Path
from collections import Counter
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.errors import EntryNotFoundError

from .final_eval import REPO, DATA, read, valid
from .final_metrics import compare

CONTROL=Path('/workspace/final-eval-control')
CACHE=Path('data/private/final-eval-report')
OUT=Path('results/final-eval')

def collect(role):
    folder=CACHE/role;folder.mkdir(parents=True,exist_ok=True)
    result={}
    for name in ['config.json','progress.json','predictions.jsonl']:
        try:
            path=hf_hub_download(REPO,f'runs/{role}/{name}',repo_type='dataset')
            (folder/name).write_bytes(Path(path).read_bytes())
        except EntryNotFoundError:continue
    if (folder/'progress.json').exists():result=json.loads((folder/'progress.json').read_text())
    return result

def publish(manifest,rows,terminal):
    systems={}
    for role in ['gemma','jev']:
        file=CACHE/role/'predictions.jsonl';items=read(file) if file.exists() else []
        byid={r['id']:r for r in rows};pred={}
        cfg=CACHE/role/'config.json'
        if cfg.exists() and json.loads(cfg.read_text())['rows_sha256']!=manifest['rows_sha256']:raise ValueError('Collected run does not match sealed data')
        for p in items:
            valid(byid[p['id']],p)
            if p['id'] in pred:raise ValueError('Duplicate collected prediction')
            pred[p['id']]=p
        systems[role]=pred
    report=compare(rows,systems['gemma'],systems['jev'])
    report.update(status='complete' if all(len(systems[r])==len(rows) for r in systems) else 'partial',
        manifest_sha256=hashlib.sha256(Path('data/final-eval-manifest.json').read_bytes()).hexdigest(),
        rows_sha256=manifest['rows_sha256'],checkpoint=manifest['gemma'],terminal=terminal,
        receipts={r:len(p) for r,p in systems.items()},
        jev_resolved_providers=dict(Counter(p.get('provider','missing') for p in systems['jev'].values())),
        blocked=manifest['blocked'],limitations=manifest['limits'])
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'comparison.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    lines=['# Independent final-test comparison','',f"Run status: **{report['status']}**. Broader classification-superiority gate: **{'passed' if report['broad_classification_superiority_supported'] else 'not passed'}**.",'',
        'The checkpoint, temperature, prompts, eligibility rules and claim gates were frozen before final predictions. Public pretraining overlap remains unknown. All requested cases remain in the primary accuracy denominator; missing or unsupported predictions count wrong.','',
        '| Cohort | Task | Cases | Families | Gemma | Jev | Difference (95% paired family CI) | Coverage G/J |',
        '|---|---|---:|---:|---:|---:|---|---|']
    for name,m in report['suites'].items():
        a,b=m['gemma'],m['jev'];p=m['paired_accuracy'];ci=p['ci95']
        interval=f"[{ci[0]*100:+.1f}, {ci[1]*100:+.1f}]" if ci else 'unavailable'
        lines.append(f"| {m['cohort']} | {name} | {a['requested']} | {m['families']} | {a['accuracy_missing_as_wrong']:.1%} | {b['accuracy_missing_as_wrong']:.1%} | {p['difference']*100:+.1f} pp {interval} | {a['coverage']:.1%}/{b['coverage']:.1%} |")
    lines+=['','## Confirmatory domains','','Equal task weighting within each domain. Holm correction covers all four prespecified domains.','',
        '| Domain | Difference | 95% paired family CI | Holm p | Superiority gate |','|---|---:|---|---:|---|']
    for name,m in report['confirmatory_domains'].items():
        if 'difference' not in m:lines.append(f'| {name} | unavailable | no eligible cases | {m["holm_p"]:.4f} | not passed |');continue
        ci=m['ci95'];interval=f'[{ci[0]*100:+.1f}, {ci[1]*100:+.1f}] pp' if ci else 'unavailable'
        lines.append(f"| {name} | {m['difference']*100:+.1f} pp | {interval} | {m['holm_p']:.4f} | {'passed' if m['supports_superiority'] else 'not passed'} |")
    for name,m in report['suites'].items():
        if 'routing' not in m:continue
        r=m['routing'];lines+=['',f'## {name}', '',f"Common-valid cached routing requests: {r['common_valid']}/{r['requested']}."]
        if 'common_valid_means' in r:
            lines+=['','| System | Quality | Cost USD/request | Utility | Oracle regret |','|---|---:|---:|---:|---:|']
            for system,v in r['common_valid_means'].items():lines.append(f"| {system} | {v['quality']:.4f} | {v['cost_usd']:.6f} | {v['utility']:.4f} | {v['oracle_regret']:.4f} |")
        lines+=['',r.get('note','')]
    lines+=['','## Limits','',*['- '+x for x in report['limitations']],
        '- TraceSafe is gated (403) and unevaluated. No claim of general tool-action safety follows from the attack-only InjecAgent recall track.',
        '- Earlier public regression cases informed development and are excluded from independent confirmation.',
        '- FinQA is numerical answer verification, not the official generation score. RouterArena is a historical three-model replay, not the complete official leaderboard.',
        '', 'Full coverage, confusion matrices, binary error rates, routing uncertainty and provenance are in [comparison.json](comparison.json).',
        'Protocol: [FINAL_EVAL_PROTOCOL.md](../../FINAL_EVAL_PROTOCOL.md). Frozen source/count/hash manifest: [final-eval-manifest.json](../../data/final-eval-manifest.json).']
    (OUT/'comparison.md').write_text('\n'.join(lines)+'\n')
    HfApi().upload_folder(repo_id=REPO,repo_type='dataset',folder_path=str(OUT),path_in_repo='aggregate',commit_message='Frozen independent comparison; all negative results retained')
    return report

def main():
    import fcntl
    parser=argparse.ArgumentParser();parser.add_argument('--watch',action='store_true');parser.add_argument('--push',action='store_true');args=parser.parse_args()
    CONTROL.mkdir(exist_ok=True);lock=(CONTROL/'report.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    manifest=json.loads(Path('data/final-eval-manifest.json').read_text());assert hashlib.sha256(DATA.read_bytes()).hexdigest()==manifest['rows_sha256']
    rows=read(DATA);started=time.time()
    while True:
        progress={r:collect(r) for r in ['gemma','jev']}
        terminal=json.loads((CONTROL/'status.json').read_text()) if (CONTROL/'status.json').exists() else {}
        stopped=terminal.get('stop_returncode')==0
        done=all(progress[r].get('completed',0)==len(rows) for r in progress)
        jev_terminal=CONTROL/'jev-terminal.json'
        jev_done=progress['jev'].get('status')=='complete' or (jev_terminal.exists() and json.loads(jev_terminal.read_text()).get('status') in ['complete','failed','timeout'])
        status={'status':'waiting','counts':{r:progress[r].get('completed',0) for r in progress},'gpu_stopped':stopped,'time':time.time()}
        (CONTROL/'report-status.json').write_text(json.dumps(status,indent=2))
        if (done and stopped) or (stopped and jev_done) or time.time()-started>7*3600 or not args.watch:break
        time.sleep(45)
    report=publish(manifest,rows,terminal)
    if args.push:
        paths=['results/final-eval/comparison.json','results/final-eval/comparison.md']
        subprocess.run(['git','add','--',*paths],check=True)
        result=subprocess.run(['git','diff','--cached','--quiet','--',*paths])
        if result.returncode:
            subprocess.run(['git','commit','--only','-m','Report frozen independent Gemma versus Jev final tests','--',*paths],check=True)
            subprocess.run(['git','push','origin','HEAD'],check=True)
    (CONTROL/'report-status.json').write_text(json.dumps({'status':report['status'],'receipts':report['receipts'],'published':args.push,'time':time.time()},indent=2))

if __name__=='__main__':main()
