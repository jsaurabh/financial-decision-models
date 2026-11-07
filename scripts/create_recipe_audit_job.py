"""Seal development-only data/code before allocating a GPU."""
import json
import subprocess
import tarfile
from pathlib import Path

from huggingface_hub import HfApi

ROOT=Path('/workspace/recipe-audit-control')
REPO='jsaurabh/financial-decision-gemma-recipe-audit'


def main():
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--seal-only',action='store_true');args=parser.parse_args()
    from hashlib import sha256
    def sha(path):return sha256(Path(path).read_bytes()).hexdigest()
    ROOT.mkdir(exist_ok=True)
    path=Path('data/recipe-audit-manifest.json');m=json.loads(path.read_text())
    assert m['status'] in ['data_prepared','sealed_before_fitting']
    for split,entry in m['splits'].items():
        assert sha(Path('data/private/recipe-audit',split+'.jsonl'))==entry['sha256']
    m['protocol_sha256']=sha('data/recipe-audit-protocol.json')
    m['code_sha256']={str(p):sha(p) for p in sorted(Path('finance_decisions').glob('*.py'))}
    m['status']='sealed_before_fitting';path.write_text(json.dumps(m,indent=2)+'\n')
    if args.seal_only:
        print('Sealed development-only manifest locally; no GPU or remote mutation')
        return
    with tarfile.open(ROOT/'recipe-code.tar.gz','w:gz') as archive:
        for p in sorted(Path('finance_decisions').glob('*.py')):archive.add(p,arcname=str(p))
        for name in ['data/recipe-audit-manifest.json','data/recipe-audit-protocol.json',
                     'data/phase4-answer-codes.json','requirements-recipe-audit.txt',
                     'requirements-gemma-eval.txt']:archive.add(name,arcname=name)
    with tarfile.open(ROOT/'recipe-data.tar.gz','w:gz') as archive:
        for split in m['splits']:
            p=Path('data/private/recipe-audit',split+'.jsonl');archive.add(p,arcname=str(p))
    with tarfile.open(ROOT/'recipe-data.tar.gz') as archive:
        assert all(name.startswith('data/private/recipe-audit/') for name in archive.getnames())
        assert not any('test' in name or 'phase4/' in name for name in archive.getnames())
    ready={'manifest_sha256':sha(path),'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
           'sha256':{n:sha(ROOT/n) for n in ['recipe-code.tar.gz','recipe-data.tar.gz']}}
    (ROOT/'recipe-ready.json').write_text(json.dumps(ready,indent=2)+'\n')
    api=HfApi()
    for kind,name in [('model',REPO),('dataset',REPO+'-data')]:
        api.create_repo(name,repo_type=kind,private=True,exist_ok=True)
        info=api.model_info(name) if kind=='model' else api.dataset_info(name)
        assert info.private
    for p,name in [(ROOT/'recipe-code.tar.gz','code.tar.gz'),(ROOT/'recipe-ready.json','ready.json'),
                   (path,'manifest.json'),(Path('data/recipe-audit-protocol.json'),'protocol.json')]:
        api.upload_file(repo_id=REPO+'-data',repo_type='dataset',path_or_fileobj=str(p),
            path_in_repo='sealed-job/'+name,commit_message='Seal Gemma-only training and development recipe comparison')
    print(json.dumps({'sealed':True,'packages':{n:(ROOT/n).stat().st_size for n in ready['sha256']}}))


if __name__=='__main__':main()
