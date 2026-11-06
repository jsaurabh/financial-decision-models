"""Live Jev on the sealed phase-2 test; references never enter requests."""
import concurrent.futures
import json
from collections import Counter
from pathlib import Path

from .evaluate_jev import call, digest
from .phase2_eval import aggregate
from .phase2_train import verify_data


def main():
    rows=verify_data()['test'];out=Path('results/phase2/jev');out.mkdir(parents=True,exist_ok=True)
    cache=out/'cache.jsonl';cached={}
    if cache.exists():
        for line in cache.read_text().splitlines():
            item=json.loads(line);cached[(item['id'],item['input_sha256'])]=item
    pending=[r for r in rows if (r['id'],digest(r)) not in cached];errors=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        futures={pool.submit(call,r,'jev-latest'):r for r in pending}
        for n,f in enumerate(concurrent.futures.as_completed(futures),1):
            row=futures[f]
            try:
                item=f.result();cached[(item['id'],item['input_sha256'])]=item
                with cache.open('a') as file:file.write(json.dumps(item)+'\n')
            except Exception as e:errors.append({'id':row['id'],'error_type':type(e).__name__,'error':str(e)})
            if n%100==0:print(json.dumps({'completed':n,'pending_initial':len(pending),'errors':len(errors)}),flush=True)
    (out/'errors.json').write_text(json.dumps(errors,indent=2))
    if errors:raise RuntimeError(f'{len(errors)} API failures; rerun to resume without repeating cached requests')
    records=[]
    for row in rows:
        item=dict(cached[(row['id'],digest(row))]);item.update(target=row['target'],family=row['family'],split='test',correct=item['prediction']==row['target']);records.append(item)
    (out/'test-predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    summary=aggregate(records);summary['providers']=dict(Counter(r['provider'] for r in records));summary['api_failures']=len(errors)
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps({'status':'complete','providers':summary['providers'],'accuracy':summary['accuracy']}),flush=True)

if __name__=='__main__':main()
