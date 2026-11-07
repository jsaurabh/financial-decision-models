"""Evaluation-only source/task transfer, never a substitute for core testing."""
import json
from collections import Counter,defaultdict
from pathlib import Path
from .phase4_data import OUT,hashed,SPLITS
from .phase3_data import read,sha
from .final_data import audit

def main():
    manifest_path=Path('data/phase4-manifest.json');manifest=json.loads(manifest_path.read_text());assert manifest['status']=='preflight_sealed_before_training'
    if (OUT/'transfer_test.jsonl').exists():raise RuntimeError('Transfer corpus already sealed')
    old=read('data/private/final-eval/rows.jsonl');traces=read('data/private/tracesafe-eval/rows.jsonl');chosen=[]
    for suite in ['finqa_verification','insurance_policy_interpretation','cuad_cap_on_liability','cuad_audit_rights','cuad_anti-assignment','cuad_termination_for_convenience','routerarena_quality','routerarena_costaware']:
        part=[r for r in old if r['suite']==suite]
        if suite=='finqa_verification':
            pairs=defaultdict(list)
            for r in part:pairs[r['source']['id']].append(r)
            part=[r for key in sorted(pairs,key=hashed)[:200] for r in pairs[key]]
        elif suite.startswith('routerarena'):
            part=sorted(part,key=lambda r:hashed(r['source']['global_index']))[:120]
        else:part=sorted(part,key=lambda r:hashed(r['id']))[:100]
        chosen.extend(part)
    for category in sorted({r['metadata']['category'] for r in traces}):
        native=defaultdict(list)
        for r in traces:
            if r['metadata']['category']==category:native[(r['source']['file'],r['source']['row'])].append(r)
        chosen.extend(r for key in sorted(native,key=lambda x:hashed(str(x)))[:20] for r in native[key])
    rows=[]
    for original in chosen:
        r=json.loads(json.dumps(original));r['id']='phase4-transfer:'+r['id'];r['cohort']='transfer';r['split']='transfer_test';r['task']='transfer/'+r['suite'];r['suite']=r['task']
        if original['suite']=='finqa_verification':r['domain']='financial';core={k:v for k,v in r['state'].items() if k not in ['question','proposed_answer']}
        elif original['suite'].startswith('routerarena'):r['domain']='model_routing';core=r['state']['question']
        elif original['suite'].startswith('tracesafe'):r['domain']='tool_safety';core=r['state']['user_query']
        else:r['domain']='legal';core=r['state']['policy'] if isinstance(r['state'],dict) else r['state']
        r['_core']=core;r['family']='phase4-transfer:'+original['family'];rows.append(r)
    known=[r for split in SPLITS for r in read(OUT/(split+'.jsonl'))]
    retained,info=audit(rows,known)
    assert {r['domain'] for r in retained}=={'financial','legal','model_routing','tool_safety'}
    from transformers import AutoTokenizer
    from .code_model import SPECS,messages
    from .schema import render
    for family,spec in SPECS.items():
        tokenizer=AutoTokenizer.from_pretrained(spec['repo'],revision=spec['revision'])
        for r in retained:
            try:
                render(r['state'],r['question'],max_choices=128,max_chars=2000000)
                text=tokenizer.apply_chat_template(messages(r),tokenize=False,add_generation_prompt=True,enable_thinking=False)
                n=len(tokenizer(text,add_special_tokens=False)['input_ids'])
            except ValueError:n=None
            r.setdefault('token_lengths',{})[family]=n
    payload=''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in retained).encode();(OUT/'transfer_test.jsonl').write_bytes(payload)
    entry={'rows':len(retained),'families':len({r['family'] for r in retained}),'sha256':sha(payload),'domains':dict(Counter(r['domain'] for r in retained)),
        'tasks':dict(Counter(r['task'] for r in retained)),'overlap_audit':info,
        'policy':'Separate source/task-transfer reporting. Exclude source families touching any core split by exact/near/long containment audit. No transfer cases are used in training, checkpoint selection or calibration.',
        'historical_exposure':'These published transfer sources/cases were evaluated by earlier models; they are evaluation-only for phase4, not claimed never inspected during the project.'}
    entry['context_support']={family:sum(r['token_lengths'][family] is not None and r['token_lengths'][family]<=8192 for r in retained) for family in SPECS}
    manifest['transfer_test']=entry;manifest['status']='sealed_before_training'
    manifest['source_code_sha256']['phase4_transfer.py']=sha(Path(__file__).read_bytes())
    manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'transfer':'sealed',**{k:v for k,v in entry.items() if k not in ['overlap_audit']}}),flush=True)

if __name__=='__main__':main()
