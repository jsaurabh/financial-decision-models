"""Allocate/stage one owned G4; start its independent lifecycle immediately."""
import argparse,json,subprocess,time
from pathlib import Path

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--family',choices=['gemma','qwen'],required=True);args=parser.parse_args();family=args.family
    root=Path('/workspace/phase4-control');control=root/family;control.mkdir(exist_ok=True)
    cli=['/workspace/.venv/bin/python','scripts/colab_public_cli.py','phase4-'+family];session='phase4-'+family
    p=subprocess.run(cli+['new','-s',session,'--gpu','G4'],capture_output=True,text=True,timeout=240)
    print(json.dumps({'family':family,'allocate_returncode':p.returncode,'detail':p.stdout[-500:]}),flush=True)
    if p.returncode:raise RuntimeError('G4 allocation failed')
    (control/'start.json').write_text(json.dumps({'time':time.time()}))
    with (control/'watcher.log').open('a') as log:
        watcher=subprocess.Popen(['/workspace/.venv/bin/python','scripts/watch_phase4.py','--family',family],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    (control/'watcher.pid').write_text(str(watcher.pid));ok=False
    try:
        uploads=[(control/'role.txt','/content/phase4-role.txt'),(Path('/workspace/.credentials',f'phase4-{family}-hf-token'),'/content/.phase4-hf-token')]
        uploads += [(root/name,'/content/'+name) for name in ['phase4-code.tar.gz','phase4-training.tar.gz','phase4-tests.tar.gz','phase4-ready.json']]
        for source,target in uploads:
            p=subprocess.run(cli+['upload','-s',session,str(source),target],capture_output=True,text=True,timeout=300)
            print(json.dumps({'family':family,'file':source.name,'returncode':p.returncode}),flush=True)
            if p.returncode:raise RuntimeError('Sealed job staging failed: '+p.stdout[-300:])
        p=subprocess.run(cli+['exec','-s',session,'-f','scripts/colab_phase4_bootstrap.py','--timeout','60'],capture_output=True,text=True,timeout=120)
        print(json.dumps({'family':family,'bootstrap_returncode':p.returncode,'detail':p.stdout[-500:]}),flush=True)
        if p.returncode:raise RuntimeError('GPU pipeline bootstrap failed')
        ok=True
    finally:
        if not ok:(control/'staging-failure.json').write_text(json.dumps({'status':'failed','error':'GPU staging/bootstrap failed; controller must release session'}))

if __name__=='__main__':main()
