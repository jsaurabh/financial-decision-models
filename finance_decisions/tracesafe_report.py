"""Privately collect a separate addendum and publish only aggregate evidence."""
import argparse,hashlib,json,subprocess,time
from pathlib import Path
from collections import Counter
from huggingface_hub import HfApi
from . import final_report as collector
from .tracesafe_eval import DATA,REPO
from .final_eval import read,valid
from .tracesafe_metrics import compare

CONTROL=Path('/workspace/tracesafe-eval-control')
CACHE=Path('data/private/tracesafe-eval-report')
OUT=Path('results/tracesafe-eval')

def publish(rows,manifest,terminal):
    systems={};byid={r['id']:r for r in rows}
    for role in ['gemma','jev']:
        file=CACHE/role/'predictions.jsonl';items=read(file) if file.exists() else [];pred={}
        cfg=CACHE/role/'config.json'
        if cfg.exists() and json.loads(cfg.read_text())['rows_sha256']!=manifest['rows_sha256']:raise ValueError('Collected addendum differs from the frozen corpus')
        for p in items:
            valid(byid[p['id']],p)
            if p['id'] in pred:raise ValueError('Duplicate addendum receipt')
            pred[p['id']]=p
        systems[role]=pred
    report=compare(rows,systems['gemma'],systems['jev'])
    report.update(status='complete' if all(len(systems[r])==len(rows) for r in systems) else 'partial',
        receipts={r:len(p) for r,p in systems.items()},checkpoint=manifest['gemma'],rows_sha256=manifest['rows_sha256'],
        manifest_sha256=hashlib.sha256(Path('data/tracesafe-eval-manifest.json').read_bytes()).hexdigest(),
        jev_resolved_providers=dict(Counter(p.get('provider','missing') for p in systems['jev'].values())),
        source=manifest['source'],terminal=terminal,limitations=manifest['limits'])
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'comparison.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    p=report['binary_primary'];lines=['# TraceSafe addendum: frozen Gemma versus Jev','',f"Status: **{report['status']}**. The original final corpus and four-domain claim gates are unchanged.",'',
        'Same full observed trajectories/toolsets, native reference labels, frozen weights/temperature/prompts and no truncation. Annotation and source-model fields were excluded from model inputs. This is a decision-interface adaptation, not an exact paper-prompt reproduction.','']
    if 'systems' in p:
        lines+=['| System | Balanced primary score | Accuracy | Coverage | Benign false blocks (valid only) |','|---|---:|---:|---:|---:|']
        for name,m in p['systems'].items():
            fpr=m['benign_false_block_rate_valid_only'];fpr=f'{fpr:.1%}' if fpr is not None else 'unavailable'
            lines.append(f"| {name} | {m['balanced_score']:.1%} | {m['metrics']['accuracy_missing_as_wrong']:.1%} | {m['metrics']['coverage']:.1%} | {fpr} |")
        ci=p['ci95'];interval=f'[{ci[0]*100:+.1f}, {ci[1]*100:+.1f}] pp' if ci else 'unavailable'
        lines+=['',f"Gemma minus Jev balanced score: **{p['difference']*100:+.1f} pp**, paired family95% interval {interval}; p={p['p_two_sided']:.4f}.",
            f"Guardrail-superiority gate: **{'passed' if p['supports_superiority'] else 'not passed'}**.",
            f"Always-unsafe baseline: {p['always_unsafe_accuracy']:.1%} source-prevalence accuracy and50.0% balanced score.",'',
            'The balanced score weights benign specificity at50% and macro-recall over12 mutation categories at50%. Missing/unsupported decisions count wrong; valid-only false blocks are disclosed separately.']
    else:lines+=['Primary claim blocked: not all13 native categories remain eligible.']
    lines+=['','| Binary source category | Cases | Gemma correctness | Jev correctness | Coverage G/J |','|---|---:|---:|---:|---|']
    for name,m in report['binary_per_category'].items():
        a,b=m['gemma'],m['jev'];lines.append(f"| {name} | {a['requested']} | {a['accuracy_missing_as_wrong']:.1%} | {b['accuracy_missing_as_wrong']:.1%} | {a['coverage']:.1%}/{b['coverage']:.1%} |")
    lines+=['','## Limits','',*['- '+v for v in report['limitations']],
        '- Unsupported long traces stay in the requested denominator. Common-valid sensitivity results are separate and cannot replace the full-cohort result.',
        '- The thirteen-way fine-grained task exceeds the frozen candidate capacity and was not run.',
        '- Offline trace classification does not establish interactive agent security or validate production financial workflows.','',
        'Full coverage/error counts, paired uncertainty, exploratory Holm tests, five-way confusion/macro-F1/root recalls, and common-valid sensitivity: [comparison.json](comparison.json).',
        'Frozen protocol: [TRACESAFE_EVAL_PROTOCOL.md](../../TRACESAFE_EVAL_PROTOCOL.md). [Source/count/hash manifest](../../data/tracesafe-eval-manifest.json).']
    (OUT/'comparison.md').write_text('\n'.join(lines)+'\n')
    HfApi().upload_folder(repo_id=REPO,repo_type='dataset',folder_path=str(OUT),path_in_repo='aggregate',commit_message='Frozen TraceSafe addendum, including losses and coverage gaps')
    return report

def main():
    import fcntl
    parser=argparse.ArgumentParser();parser.add_argument('--watch',action='store_true');parser.add_argument('--push',action='store_true');args=parser.parse_args()
    CONTROL.mkdir(exist_ok=True);lock=(CONTROL/'report.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    manifest=json.loads(Path('data/tracesafe-eval-manifest.json').read_text());assert hashlib.sha256(DATA.read_bytes()).hexdigest()==manifest['rows_sha256']
    rows=read(DATA);collector.CACHE=CACHE;collector.REPO=REPO;started=time.time()
    while True:
        progress={r:collector.collect(r) for r in ['gemma','jev']}
        terminal=json.loads((CONTROL/'status.json').read_text()) if (CONTROL/'status.json').exists() else {}
        cpu=CONTROL/'jev-terminal.json';cpu_done=cpu.exists()
        stopped=terminal.get('stop_returncode')==0
        done=all(progress[r].get('completed',0)==len(rows) for r in progress)
        (CONTROL/'report-status.json').write_text(json.dumps({'status':'waiting','counts':{r:progress[r].get('completed',0) for r in progress},'gpu_stopped':stopped,'time':time.time()},indent=2))
        if (stopped and (done or cpu_done)) or time.time()-started>4*3600 or not args.watch:break
        time.sleep(45)
    report=publish(rows,manifest,terminal)
    if args.push:
        paths=['results/tracesafe-eval/comparison.json','results/tracesafe-eval/comparison.md']
        subprocess.run(['git','add','--',*paths],check=True)
        result=subprocess.run(['git','diff','--cached','--quiet','--',*paths])
        if result.returncode:
            subprocess.run(['git','commit','--only','-m','Report independent TraceSafe Gemma versus Jev addendum','--',*paths],check=True)
            subprocess.run(['git','push','origin','HEAD'],check=True)
    (CONTROL/'report-status.json').write_text(json.dumps({'status':report['status'],'receipts':report['receipts'],'published':args.push,'time':time.time()},indent=2))

if __name__=='__main__':main()
