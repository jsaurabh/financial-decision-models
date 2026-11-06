"""Matched Jev baseline for the reported calibration fold, not a final test."""
import concurrent.futures
import hashlib
import json
from collections import Counter
from pathlib import Path

from huggingface_hub import HfApi
from sklearn.metrics import f1_score

from .evaluate_jev import call, digest
from .schema import options

PRIVATE = Path('data/private/phase3-jev-calibration')
OUTPUT = Path('results/phase3-pilot')


def read(path):
    with Path(path).open() as f:
        return [json.loads(line) for line in f if line.strip()]


def validate(row, prediction):
    if prediction['id'] != row['id'] or prediction['input_sha256'] != digest(row):
        raise ValueError('Cached prediction does not match the canonical input')
    if prediction['labels'] != options(row['question'])[0]:
        raise ValueError('Cached candidate order differs')


def summarize(rows, predictions, gemma):
    domains = {}
    for domain in sorted({r['domain'] for r in rows}):
        subset = [r for r in rows if r['domain'] == domain]
        valid = [(r, predictions[r['id']]) for r in subset
                 if r['id'] in predictions and predictions[r['id']].get('status') == 'ok']
        for r, p in valid:
            validate(r, p)
        accuracy = sum(r['target'] == p['prediction'] for r, p in valid)/len(subset)
        entry = {'requested': len(subset), 'source_groups':len({r['family'] for r in subset}), 'jev_valid': len(valid), 'jev_coverage': len(valid)/len(subset),
                 'gemma_accuracy': gemma[domain]['accuracy'], 'jev_accuracy_missing_as_wrong': accuracy,
                 'gemma_minus_jev_accuracy': gemma[domain]['accuracy']-accuracy if len(valid)==len(subset) else None,
                 'gemma_macro_f1': gemma[domain]['macro_f1']}
        assert gemma[domain]['n'] == len(subset)
        if valid:
            entry['jev_macro_f1_valid_only'] = float(f1_score([r['target'] for r,p in valid], [p['prediction'] for r,p in valid], average='macro', zero_division=0))
            if all(r['question']['type']=='noul' for r,p in valid):
                positive = sum(r['target']=='true' for r,p in valid)
                negative = len(valid)-positive
                entry['jev_positive_recall'] = sum(r['target']=='true' and p['prediction']=='true' for r,p in valid)/max(1,positive)
                entry['jev_false_positive_rate'] = sum(r['target']=='false' and p['prediction']=='true' for r,p in valid)/max(1,negative)
                entry['gemma_positive_recall'] = gemma[domain]['positive_recall']
                entry['gemma_false_positive_rate'] = gemma[domain]['false_positive_rate']
        domains[domain] = entry
    return domains


def main():
    import fcntl
    PRIVATE.mkdir(parents=True, exist_ok=True)
    lock = (PRIVATE/'worker.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    data = Path('data/private/phase3/calibration.jsonl')
    manifest = json.loads(Path('data/phase3-manifest.json').read_text())
    config = json.loads((OUTPUT/'run_config.json').read_text())
    assert hashlib.sha256(data.read_bytes()).hexdigest() == manifest['splits']['calibration']['sha256']
    assert hashlib.sha256(Path('data/phase3-manifest.json').read_bytes()).hexdigest() == config['manifest_sha256']
    gemma = json.loads((OUTPUT/'summary.json').read_text())
    artifact = json.loads((OUTPUT/'artifacts.json').read_text())
    rows = read(data); assert len(rows) == len({r['id'] for r in rows}) == 700
    provenance = {'split': 'calibration', 'rows': len(rows), 'data_sha256': manifest['splits']['calibration']['sha256'],
        'gemma_artifact': artifact, 'gemma_selected_step': gemma['selected_step'], 'jev_alias': 'jev-latest',
        'request_fields': 'model, state, questions only; references attached after API return',
        'scope': 'Post-training diagnostic on the reported calibration cases; not a reserved final test. No training or threshold fitting.'}
    config_path = PRIVATE/'config.json'
    if config_path.exists(): assert json.loads(config_path.read_text()) == provenance
    config_path.write_text(json.dumps(provenance, indent=2)+'\n')
    cache = PRIVATE/'predictions.jsonl'; predictions = {}
    byid = {r['id']: r for r in rows}
    if cache.exists():
        for p in read(cache):
            validate(byid[p['id']], p)
            if p.get('status') == 'ok': predictions[p['id']] = p
    pending = [r for r in rows if r['id'] not in predictions]
    print(json.dumps({'status':'started','cached':len(predictions),'pending':len(pending),'rows':len(rows)}),flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        futures = {pool.submit(call, r, 'jev-latest', timeout=120): r for r in pending}
        for n, future in enumerate(concurrent.futures.as_completed(futures),1):
            row = futures[future]
            try:
                p = future.result(); validate(row, p); p['status'] = 'ok'
            except Exception as exc:
                p = {'id':row['id'],'input_sha256':digest(row),'labels':options(row['question'])[0],
                     'status':'error','error_type':type(exc).__name__}
            p.update(target=row['target'],family=row['family'],domain=row['domain'])
            with cache.open('a') as f: f.write(json.dumps(p,allow_nan=False)+'\n')
            predictions[row['id']] = p
            if n%50==0 or n==len(pending):
                progress={'status':'running','completed':len(predictions),'total':len(rows),'errors':sum(p['status']!='ok' for p in predictions.values())}
                (PRIVATE/'progress.json').write_text(json.dumps(progress));print(json.dumps(progress),flush=True)
    domains = summarize(rows,predictions,gemma['calibration']['domains'])
    complete = all(m['jev_valid']==m['requested'] for m in domains.values())
    report = {'status':'complete' if complete else 'partial','provenance':provenance,'domains':domains,
        'resolved_providers':dict(Counter(p['provider'] for p in predictions.values() if p.get('status')=='ok')),
        'limits':['100 decisions per task; source-group independence is smaller than the row count.',
                  'Native source labels have not been independently expert-adjudicated for this comparison. When2Call preferences are synthetic source supervision, not the native four-way action test; banking uses the earlier eight-candidate adaptation.',
                  'Gemma calibration metrics are from the recorded frozen training artifact, not a new GPU replay.',
                  'No per-case Gemma outputs were saved by the trainer; paired significance/discordance is not computed.',
                  'Temperature was fitted on this fold for Gemma. We compare top-1 decisions, which positive scalar temperature does not change.',
                  'This comparison is not an untouched final-test or production-quality claim.']}
    (OUTPUT/'jev-calibration-comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    lines=['# Matched Jev calibration diagnostic','',provenance['scope'],'',
           '| Task | Cases | Gemma | Jev | Gemma minus Jev |','|---|---:|---:|---:|---:|']
    for name,m in domains.items():
        gap=f"{m['gemma_minus_jev_accuracy']*100:+.1f} pp" if m['gemma_minus_jev_accuracy'] is not None else 'incomplete'
        lines.append(f"| {name} | {m['requested']} | {m['gemma_accuracy']:.1%} | {m['jev_accuracy_missing_as_wrong']:.1%} | {gap} |")
    lines+=['',f"Resolved providers: `{json.dumps(report['resolved_providers'],sort_keys=True)}`.",'',
            'Identical source inputs, question text, candidate order and reference labels; input hashes verified. Gemma is the development-selected step1200 artifact with its saved temperature. Reference labels never enter Jev requests.','',
            '## Limits','',*['- '+v for v in report['limits']],'',
            'See [metrics and pinned provenance](jev-calibration-comparison.json). Per-case Jev probabilities stay private.']
    (OUTPUT/'jev-calibration-comparison.md').write_text('\n'.join(lines)+'\n')
    api=HfApi();repo='jsaurabh/financial-decision-phase3-pilot';assert api.dataset_info(repo).private
    for local,name in [(cache,'predictions.jsonl'),(config_path,'config.json'),(OUTPUT/'jev-calibration-comparison.json','summary.json')]:
        api.upload_file(repo_id=repo,repo_type='dataset',path_or_fileobj=str(local),path_in_repo='comparisons/jev-calibration/'+name,commit_message='Matched calibration diagnostic; no training or final benchmark evaluation')
    (PRIVATE/'progress.json').write_text(json.dumps({'status':report['status'],'completed':len(predictions),'total':len(rows)}))
    print(json.dumps({'status':report['status'],'domains':domains,'providers':report['resolved_providers']}),flush=True)


if __name__=='__main__':main()
