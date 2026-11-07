"""Seal model-specific context support before any fitting or test predictions."""
import json,hashlib
from collections import Counter,defaultdict
from pathlib import Path
from transformers import AutoTokenizer
from .phase4_data import OUT,SPLITS,CAPS
from .phase3_data import read,sha
from .code_model import SPECS,messages
from .schema import render

def main():
    file=Path('data/phase4-manifest.json');manifest=json.loads(file.read_text())
    if manifest['status']!='sealed_data_before_training':raise RuntimeError('Preflight already sealed')
    rows={s:read(OUT/(s+'.jsonl')) for s in SPLITS};support=defaultdict(dict)
    for family,spec in SPECS.items():
        tokenizer=AutoTokenizer.from_pretrained(spec['repo'],revision=spec['revision'])
        for split in SPLITS:
            for index,row in enumerate(rows[split]):
                try:
                    render(row['state'],row['question'],max_choices=128,max_chars=2000000)
                    text=tokenizer.apply_chat_template(messages(row),tokenize=False,add_generation_prompt=True,enable_thinking=False)
                    n=len(tokenizer(text,add_special_tokens=False)['input_ids'])
                except ValueError:n=None
                row.setdefault('token_lengths',{})[family]=n
                if index%2000==0:print(json.dumps({'model':family,'split':split,'tokenized':index,'total':len(rows[split])}),flush=True)
    excluded={r['family'] for r in rows['train'] if any(n is None or n>8192 for n in r['token_lengths'].values())}
    removed=[r for r in rows['train'] if r['family'] in excluded];rows['train']=[r for r in rows['train'] if r['family'] not in excluded]
    assert set(r['task'] for r in rows['train'])==set(CAPS)
    training=defaultdict(set)
    for r in rows['train']:
        key=(r['task'],r['source'].get('review_question',''))
        value=r['question']['criteria'][r['target']] if r['task']=='legal_maud' else r['target'];training[key].add(value)
    for split in SPLITS[1:]:
        for r in rows[split]:
            value=r['question']['criteria'][r['target']] if r['task']=='legal_maud' else r['target']
            assert value in training[(r['task'],r['source'].get('review_question',''))],f'Untrained core reference class: {split}/{r["task"]}'
    all_families=set()
    for split,part in rows.items():
        families={r['family'] for r in part};assert not families&all_families;all_families|=families
        payload=''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in part).encode();(OUT/(split+'.jsonl')).write_bytes(payload)
        manifest['splits'][split].update(rows=len(part),families=len(families),sha256=sha(payload),domains=dict(Counter(r['domain'] for r in part)),tasks=dict(Counter(r['task'] for r in part)),
            class_counts={t:dict(Counter(r['target'] for r in part if r['task']==t)) for t in CAPS})
        support[split]={family:{task:{'requested':len(task_rows),'supported_8192':sum(r['token_lengths'][family] is not None and r['token_lengths'][family]<=8192 for r in task_rows)} for task in CAPS if (task_rows:=[r for r in part if r['task']==task])} for family in SPECS}
    # Frozen selection subset comes ONLY from validation, not core/transfer tests.
    selection=[]
    for task in CAPS:
        candidates=sorted([r for r in rows['validation'] if r['task']==task],key=lambda r:sha(r['id'].encode()))
        buckets=defaultdict(list)
        for r in candidates:buckets[(r['source'].get('review_question',''),r['target'])].append(r)
        cap=128 if task=='legal_maud' else 64;part=[]
        while len(part)<cap and any(buckets.values()):
            for key in sorted(buckets):
                if buckets[key]:part.append(buckets[key].pop())
                if len(part)==cap:break
        selection.extend(part)
    payload=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in selection).encode();(OUT/'selection_validation.jsonl').write_bytes(payload)
    paths=json.loads(Path('data/private/phase4-sources/paths.json').read_text())
    used_repos={r['source']['repo'] for part in rows.values() for r in part}
    hashes={key:sha(Path(path).read_bytes()) for key,path in paths.items() if any(key.startswith(repo+'/') for repo in used_repos)}
    original=json.loads(Path('/workspace/public-eval-source/paths.json').read_text())
    for shot in [0,5]:
        key=f'withmartian/routerbench/routerbench_{shot}shot.pkl';hashes[key]=sha(Path(original[key]).read_bytes())
    manifest.update(status='preflight_sealed_before_training',model_specs=SPECS,max_tokens=8192,
        source_file_sha256=hashes,source_pins={k:v for k,v in manifest['source_pins'].items() if any(k==repo.replace('/','__') for repo in used_repos)},
        context_support=support,training_context_exclusion={'families':len(excluded),'rows':len(removed),'by_task':dict(Counter(r['task'] for r in removed)),'policy':'Same training rows for both models; exclude whole source families if any selected training view exceeds either tokenizer envelope. All validation/calibration/test inputs remain requested, without truncation.'},
        selection_validation={'rows':len(selection),'sha256':sha(payload),'parent_split':'validation'},
        source_code_sha256={name:sha(Path('finance_decisions',name).read_bytes()) for name in ['phase4_data.py','phase4_numeric.py','phase4_preflight.py','code_model.py']})
    file.write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'preflight':'sealed','splits':{s:manifest['splits'][s]['rows'] for s in SPLITS},'excluded_training_rows':len(removed),'selection_validation':len(selection)}),flush=True)

if __name__=='__main__':main()
