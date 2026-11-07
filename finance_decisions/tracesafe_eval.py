"""Configure the frozen final-test engine for a separately sealed addendum."""
import argparse,hashlib,json
from pathlib import Path
from huggingface_hub import HfApi
from . import final_eval as engine

REPO='jsaurabh/financial-decision-tracesafe-eval'
DATA=Path('data/private/tracesafe-eval/rows.jsonl')
PRIVATE=Path('data/private/tracesafe-eval-results')

def main():
    import fcntl
    parser=argparse.ArgumentParser();parser.add_argument('--role',choices=['gemma','jev'],required=True);args=parser.parse_args()
    manifest=json.loads(Path('data/tracesafe-eval-manifest.json').read_text())
    assert hashlib.sha256(DATA.read_bytes()).hexdigest()==manifest['rows_sha256']
    assert hashlib.sha256(Path('TRACESAFE_EVAL_PROTOCOL.md').read_bytes()).hexdigest()==manifest['protocol_sha256']
    rows=engine.read(DATA);assert len(rows)==manifest['rows']==len({r['id'] for r in rows})
    assert HfApi().dataset_info(REPO).private
    engine.REPO=REPO;out=PRIVATE/args.role;out.mkdir(parents=True,exist_ok=True)
    lock=(out/'worker.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    cfg=engine.config(args.role,manifest)
    cfg['addendum_runner_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    records=engine.restore(out,cfg,rows)
    if args.role=='gemma':
        # Exact pinned-tokenizer preflight, no input truncation or capacity change.
        for row in rows:
            if row['id'] not in records and row['packed_tokens']>8192:
                engine.append(out,records,engine.meta(row)|{'status':'unsupported','error_type':'over_context_limit','input_tokens':row['packed_tokens']})
        engine.gemma(out,records,rows,manifest)
    else:engine.jev(out,records,rows,manifest)

if __name__=='__main__':main()
