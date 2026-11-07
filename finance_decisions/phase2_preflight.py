"""Freeze shared, untruncated input eligibility before any model scoring."""
import hashlib
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from transformers import AutoTokenizer
from torch import nn
from .candidate_model import CandidateDecisionModel
from .model import BASE, REVISION
from .phase2_data import ROOT


def main():
    tokenizer=AutoTokenizer.from_pretrained(BASE,revision=REVISION)
    backbone=nn.Linear(1,1);backbone.config=SimpleNamespace(hidden_size=1,sliding_window=512)
    model=CandidateDecisionModel(backbone,tokenizer,max_length=2048)
    manifest=json.loads(Path('data/phase2-manifest.json').read_text())
    for split,entry in manifest['splits'].items():
        path=ROOT/f'{split}.jsonl';rows=[json.loads(x) for x in path.read_text().splitlines()]
        accepted=[];seen={};excluded=Counter();lengths=[]
        for row in rows:
            key=json.dumps([row['state'],row['question']],sort_keys=True,ensure_ascii=False)
            if key in seen:
                assert seen[key]==row['target'];excluded['duplicate identical decision input']+=1;continue
            seen[key]=row['target']
            try:n=model.token_length(row)
            except ValueError:excluded['schema input length']+=1;continue
            if n>2048:excluded['over 2048 packed tokens']+=1;continue
            accepted.append(row);lengths.append(n)
        path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in accepted))
        entry.update(rows=len(accepted),unique_states=len({json.dumps(r['state'],sort_keys=True) for r in accepted}),
            families=len({r['family'] for r in accepted}),domains=dict(Counter(r['domain'] for r in accepted)),
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),max_packed_tokens=max(lengths),
            exclusions_before_training=dict(excluded),requested_rows=len(rows))
    manifest['eligibility']={'tokenizer':BASE,'revision':REVISION,'max_packed_tokens':2048,'truncation':False,'frozen_before_scoring':True}
    Path('data/phase2-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2))

if __name__=='__main__':main()
