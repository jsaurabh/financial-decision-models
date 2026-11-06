"""Seal new public test examples before the candidate-head quality run."""
import hashlib
import json
import random
from collections import defaultdict,Counter
from pathlib import Path
from datasets import load_dataset
from .build_data import SOURCES,record,sampled_options


def normalized(text):
    return ' '.join(text.lower().split())


def main():
    rng=random.Random(20261107); seen=set(); used_ids=set()
    for split in ['train','calibration','test']:
        for line in Path(f'data/private/{split}.jsonl').read_text().splitlines():
            row=json.loads(line);seen.add(normalized(row['state']));used_ids.add(row['id'])
    result=[]
    for alias,subset,domain,field,n in [('banking77',None,'banking_intent','label',6),
        ('lex_glue','ledgar','legal_contract','label',6),('lex_glue','unfair_tos','legal_terms','labels',400)]:
        source=SOURCES[alias];ds=load_dataset(source['repo'],subset,revision=source['revision'])
        native=ds['test'];features=native.features[field]
        names=features.feature.names if field=='labels' else features.names
        available=[]
        for i,row in enumerate(native):
            old_id=f'{alias}-{subset}-test-{i}'
            if old_id not in used_ids and normalized(row['text']) not in seen:
                available.append((i,row))
        if field=='label':
            groups=defaultdict(list)
            for i,row in available:groups[row['label']].append((i,row))
            chosen=[]
            for label,rows in sorted(groups.items()):rng.shuffle(rows);chosen.extend(rows[:n])
        else:
            rng.shuffle(available);chosen=available[:n]
        for i,row in chosen:
            text=normalized(row['text'])
            if text in seen:continue
            seen.add(text);rid=f'public-fresh-{alias}-{subset}-{i}'
            if field=='label':
                target=names[row['label']]
                r=record(rid,domain,row['text'],'Choose the best matching banking intent.' if alias=='banking77' else 'Choose the contract clause topic.',
                    sampled_options(names,target,rng),target)
            else:
                positive=list(row['labels']);k=rng.choice(positive) if positive and rng.random()<.5 else rng.randrange(len(names))
                r=record(rid,domain,row['text'],f'Does this clause contain {names[k].lower()}?',None,'true' if k in positive else 'false',kind='noul')
            r['source']={**source,'subset':subset,'native_split':'test','row':i};result.append(r)
    rng.shuffle(result);path=Path('data/private/public-fresh.jsonl')
    path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in result))
    manifest={'seed':20261107,'rows':len(result),'domains':dict(Counter(r['domain'] for r in result)),
        'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'banking_intents_represented':len({r['target'] for r in result if r['domain']=='banking_intent'}),
        'contract_topics_represented':len({r['target'] for r in result if r['domain']=='legal_contract'}),
        'split':'official test rows; prior train, calibration, and inspected test rows/states excluded',
        'use':'final-only; no training/calibration/checkpoint selection','limits':'Eight sampled candidates including the reference for banking/LEDGAR; not full-taxonomy classification.'}
    Path('data/public-fresh-manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest,indent=2))


if __name__=='__main__':main()
