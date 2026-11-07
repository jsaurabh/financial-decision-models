"""Verify the final split seal and package fitting inputs separately from tests."""
import hashlib,json,subprocess,tarfile
from pathlib import Path
from huggingface_hub import HfApi

ROOT=Path('/workspace/phase4-control');PROJECT=Path.cwd()
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    manifest_path=Path('data/phase4-manifest.json');m=json.loads(manifest_path.read_text());assert m['status']=='sealed_before_training'
    assert m['protocol_sha256']==sha('PHASE4_TRAINING_PROTOCOL.md')
    assert m['answer_code_vocabulary_sha256']==sha('data/phase4-answer-codes.json')
    for name,value in m['source_code_sha256'].items():assert sha(Path('finance_decisions',name))==value,(name,'Source changed after seal')
    for split,entry in m['splits'].items():assert sha(Path('data/private/phase4',split+'.jsonl'))==entry['sha256']
    assert sha('data/private/phase4/selection_validation.jsonl')==m['selection_validation']['sha256']
    assert sha('data/private/phase4/transfer_test.jsonl')==m['transfer_test']['sha256']
    with tarfile.open(ROOT/'phase4-code.tar.gz','w:gz') as archive:
        for path in sorted(Path('finance_decisions').glob('*.py')):archive.add(path,arcname=str(path))
        for name in ['data/phase4-manifest.json','data/phase4-answer-codes.json','PHASE4_TRAINING_PROTOCOL.md','requirements-phase4.txt','requirements-gemma-eval.txt']:archive.add(name,arcname=name)
    for kind,splits in [('training',['train','validation','calibration','selection_validation']),('tests',['core_test','transfer_test'])]:
        with tarfile.open(ROOT/f'phase4-{kind}.tar.gz','w:gz') as archive:
            for split in splits:
                path=Path('data/private/phase4',split+'.jsonl');archive.add(path,arcname=str(path))
    with tarfile.open(ROOT/'phase4-training.tar.gz') as archive:assert not any('core_test' in n or 'transfer_test' in n for n in archive.getnames())
    ready={'sha256':{name:sha(ROOT/name) for name in ['phase4-code.tar.gz','phase4-training.tar.gz','phase4-tests.tar.gz']},
        'manifest_sha256':sha(manifest_path),'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()}
    (ROOT/'phase4-ready.json').write_text(json.dumps(ready,indent=2)+'\n')
    api=HfApi();repo='jsaurabh/financial-decision-phase4';assert api.dataset_info(repo).private
    for path,name in [(ROOT/'phase4-code.tar.gz','code.tar.gz'),(ROOT/'phase4-ready.json','ready.json'),(manifest_path,'manifest.json'),(Path('PHASE4_TRAINING_PROTOCOL.md'),'protocol.md')]:
        api.upload_file(repo_id=repo,repo_type='dataset',path_or_fileobj=str(path),path_in_repo='sealed-job/'+name,commit_message='Seal aligned native-label core splits and separate transfer before fitting')
    print(json.dumps({'sealed_job':True,'source_commit':ready['source_commit'],'packages':{n:(ROOT/n).stat().st_size for n in ready['sha256']}}),flush=True)

if __name__=='__main__':main()
