"""Fetch pinned public inputs; private working copies only, never execute data."""
import argparse,concurrent.futures,json,requests,tarfile,zipfile,io
from pathlib import Path
from huggingface_hub import hf_hub_download
parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,default=Path('/workspace/public-eval-source'));root=parser.parse_args().out;root.mkdir(parents=True,exist_ok=True)
jobs=[('nvidia/When2Call','0582f7749df63a96fdc3070932e83e72396ace53','test/when2call_test_mcq.jsonl'),('MurrayTom/TS-Bench','19047a0c4a54f54cc1ddd4a2c597105d8453b819','data/eval/AgentHarm-Traj.jsonl'),('MurrayTom/TS-Bench','19047a0c4a54f54cc1ddd4a2c597105d8453b819','data/eval/ASB-Traj.jsonl'),('MurrayTom/TS-Bench','19047a0c4a54f54cc1ddd4a2c597105d8453b819','data/eval/AgentDojo-Traj.jsonl'),('withmartian/routerbench','784021482c3f320c6619ed4b3bb3b41a21424fcb','routerbench_0shot.pkl'),('withmartian/routerbench','784021482c3f320c6619ed4b3bb3b41a21424fcb','routerbench_5shot.pkl')]
def hf(job):
 repo,rev,name=job;p=hf_hub_download(repo,name,revision=rev,repo_type="dataset");print(json.dumps({'source':repo,'file':name,'path':p}),flush=True);return '/'.join([repo,name]),p
paths=dict(concurrent.futures.ThreadPoolExecutor(5).map(hf,jobs))
urls={'rjudge.tar.gz':'https://codeload.github.com/Lordog/R-Judge/tar.gz/83ce301da3ad50dd8b397e772863f5411c3d3dc2','contract-nli.zip':'https://raw.githubusercontent.com/stanfordnlp/contract-nli/eced6528dd3c1d14d73f9a87df8f7bdbc03126f9/resources/contract-nli.zip','rag-responses.jsonl':'https://raw.githubusercontent.com/ParticleMedia/RAGTruth/c103204b9ce28d6bbad859304bf30de72b8ed8fe/dataset/response.jsonl','rag-sources.jsonl':'https://raw.githubusercontent.com/ParticleMedia/RAGTruth/c103204b9ce28d6bbad859304bf30de72b8ed8fe/dataset/source_info.jsonl','decision-index.tar.gz':'https://codeload.github.com/apolinario/decision-index/tar.gz/25c6ec7b6b5f44a216c63c3a700de3a9c6d6699d'}
def direct(item):
 name,url=item;r=requests.get(url,timeout=90);r.raise_for_status();p=root/name;p.write_bytes(r.content);print(json.dumps({'downloaded':name,'bytes':len(r.content)}),flush=True);return name,str(p)
paths.update(dict(concurrent.futures.ThreadPoolExecutor(5).map(direct,urls.items())))
(root/'paths.json').write_text(json.dumps(paths,indent=2));print('source downloads complete',flush=True)

for name in ['rjudge.tar.gz','decision-index.tar.gz']:
 with tarfile.open(root/name) as archive:archive.extractall(root,filter='data')
