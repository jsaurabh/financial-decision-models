"""Development-only controlled recipe subsets. Never read final-test cases."""
import hashlib
import json
import random
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path

from transformers import AutoTokenizer

from .code_model import CODES, SPECS, messages
from .phase3_data import read, sha
from .phase4_metrics import accepted, labels

ROOT = Path('data/private/recipe-audit')
SEED = 20261107
CAPS = {
    'train': {'financial_verification':512, 'banking_intent':256, 'legal_maud':384,
        'legal_topic':384, 'tool_action_safety':256, 'prompt_injection':256,
        'malicious_request':256, 'routing_quality':384, 'routing_costaware':384},
    'development': {'financial_verification':128, 'banking_intent':64, 'legal_maud':128,
        'legal_topic':64, 'tool_action_safety':64, 'prompt_injection':64,
        'malicious_request':64, 'routing_quality':96, 'routing_costaware':96},
    'memorization': {'financial_verification':32, **{t:4 for t in [
        'banking_intent','legal_maud','legal_topic','tool_action_safety',
        'prompt_injection','malicious_request','routing_quality','routing_costaware']}},
}


def order(value):
    return hashlib.sha256(('gemma-recipe-v1:' + str(value)).encode()).hexdigest()


def semantic(row, target=None):
    target = row['target'] if target is None else target
    return (row['task'], row['source'].get('review_question', ''),
            row['question']['criteria'][target] if row['task']=='legal_maud' else target)


def choose(rows, caps, balanced=False):
    selected = []
    for task, count in caps.items():
        part = sorted((r for r in rows if r['task']==task), key=lambda r:order(r['id']))
        if task=='financial_verification':
            pairs = defaultdict(list)
            for row in part:
                pairs[row['metadata']['pair']].append(row)
            complete = [v for _, v in sorted(pairs.items(), key=lambda kv:order(kv[0]))
                        if len(v)==2 and {r['target'] for r in v}=={'false','true'}]
            chosen = [r for pair in complete[:count//2] for r in pair]
        elif balanced:
            buckets = defaultdict(list)
            for row in part:
                buckets[semantic(row)].append(row)
            chosen = []
            while len(chosen)<count and any(buckets.values()):
                for key in sorted(buckets):
                    if buckets[key]:
                        chosen.append(buckets[key].pop())
                    if len(chosen)==count:
                        break
        else:
            chosen = part[:count]
        if len(chosen)!=count:
            raise ValueError(f'Insufficient {task}: {len(chosen)}/{count}')
        selected.extend(deepcopy(chosen))
    return selected


def controls(rows):
    result = []
    anchors = choose(rows, {t:16 for t in CAPS['train']})
    for row in anchors:
        keys = labels(row)
        from .schema import options
        descriptions = options(row['question'], max_choices=128)[1]
        positions = list(range(len(keys)))
        random.Random(int(order(row['id'])[:12],16)).shuffle(positions)
        if positions==list(range(len(keys))):
            positions = positions[1:]+positions[:1]
        for kind in ['permutation','aliases']:
            clone = deepcopy(row)
            clone['id'] = kind+':'+row['id']
            mapping = {('candidate_'+str(i) if kind=='aliases' else keys[i]):keys[i]
                       for i in positions}
            reverse = {old:new for new,old in mapping.items()}
            clone['question']['type'] = 'choice'
            clone['question']['criteria'] = {
                new: (old+': '+descriptions[keys.index(old)] if kind=='aliases'
                      else descriptions[keys.index(old)]) for new,old in mapping.items()}
            clone['target'] = reverse[row['target']]
            clone['metadata']['accepted'] = [reverse[k] for k in accepted(row)]
            if 'outcomes' in clone['metadata']:
                clone['metadata']['outcomes'] = {reverse[k]:v for k,v in row['metadata']['outcomes'].items()}
                clone['state']['training_model_profiles'] = {
                    reverse[k]:v for k,v in row['state']['training_model_profiles'].items()}
            clone['metadata']['control'] = {'kind':kind,'parent_id':row['id'],
                                             'to_parent_label':mapping}
            result.append(clone)
        if row['task']=='financial_verification':
            clone = deepcopy(row)
            clone['id'] = 'predicate_reversal:'+row['id']
            clone['target'] = 'false' if row['target']=='true' else 'true'
            clone['metadata']['accepted'] = [clone['target']]
            clone['question'] = {'type':'noul',
                'instructions':'Is the proposed answer numerically incorrect for the question, using the supplied financial report context? Respect the displayed units, scale and rounding.',
                'criteria':{'false':'The proposed answer is numerically correct.',
                            'true':'The proposed answer is numerically incorrect.'}}
            clone['metadata']['control'] = {'kind':'predicate_reversal','parent_id':row['id'],
                    'to_parent_label':{'false':'true','true':'false'}}
            result.append(clone)
    return result


def main():
    if (ROOT/'train.jsonl').exists():
        raise RuntimeError('Recipe audit data already sealed')
    phase4 = json.loads(Path('data/phase4-manifest.json').read_text())
    parent = {}
    for name in ['train','validation']:
        path = Path('data/private/phase4',name+'.jsonl')
        assert sha(path.read_bytes())==phase4['splits'][name]['sha256']
        parent[name] = read(path)
    train = choose(parent['train'],CAPS['train'],balanced=True)
    development = choose(parent['validation'],CAPS['development'])
    assert {semantic(r) for r in development}<={semantic(r) for r in train}
    assert {r['family'] for r in train}.isdisjoint({r['family'] for r in development})
    profiles = defaultdict(lambda:defaultdict(list))
    for row in train:
        if row['task']=='routing_quality':
            for model, outcome in row['metadata']['outcomes'].items():
                profiles[row['state']['shot_regime']][model].append(outcome)
    fitted = {shot:{model:{'mean_quality':sum(v['quality'] for v in values)/len(values),
                'mean_cost_usd':sum(v['cost_usd'] for v in values)/len(values),
                'training_cases':len(values)} for model,values in models.items()}
                for shot,models in profiles.items()}
    for row in train+development:
        if row['domain']=='model_routing':
            row['state']['training_model_profiles'] = fitted[row['state']['shot_regime']]
    splits = {'train':train,'development':development,
              'memorization':choose(train,CAPS['memorization'],balanced=True),
              'controls':controls(development)}
    tokenizer = AutoTokenizer.from_pretrained(**{'pretrained_model_name_or_path':SPECS['gemma']['repo'],
                                                'revision':SPECS['gemma']['revision']})
    ROOT.mkdir(parents=True,exist_ok=True)
    entries = {}
    for split,rows in splits.items():
        for row in rows:
            text = tokenizer.apply_chat_template(messages(row),tokenize=False,
                        add_generation_prompt=True,enable_thinking=False)
            length = len(tokenizer(text,add_special_tokens=False)['input_ids'])
            assert length<=8192
            row['token_lengths'] = {'gemma':length}
        random.Random(SEED).shuffle(rows)
        data = ''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in rows).encode()
        (ROOT/(split+'.jsonl')).write_bytes(data)
        entries[split] = {'rows':len(rows),'sha256':sha(data),
            'tasks':dict(Counter(r['task'] for r in rows)),
            'families':len({r['family'] for r in rows}),
            'max_tokens':max(r['token_lengths']['gemma'] for r in rows)}
    manifest = {'status':'data_prepared','base':SPECS['gemma'],'seed':SEED,'splits':entries,
        'parent_phase4_manifest_sha256':sha(Path('data/phase4-manifest.json').read_bytes()),
        'parent_splits':['train','validation'],'test_cases_accessed':False,
        'routing_profiles':'Fitted only from selected pilot training quality outcomes, identical in all arms.',
        'development_exposure':'Existing project validation; used for recipe comparison only, not a fresh superiority claim.',
        'answer_vocabulary_sha256':sha(Path('data/phase4-answer-codes.json').read_bytes())}
    Path('data/recipe-audit-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(entries,indent=2))


if __name__=='__main__':
    main()
