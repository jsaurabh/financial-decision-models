"""Freeze new group-disjoint quality data; old inspected cases are exclusion-only.

All policies are invented. Numerical/Boolean references are computed, never
provided by Jev. Template/register and counterfactual family groups stay together.
"""
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from datasets import load_dataset
from .build_data import SOURCES, record, sampled_options

SEED = 20261108
SPLITS = ('train', 'development', 'calibration', 'test')
ROOT = Path('data/private/phase2')


def sha(value):
    return hashlib.sha256(value).hexdigest()


def norm(state):
    if not isinstance(state, str):
        state = json.dumps(state, sort_keys=True, ensure_ascii=False)
    return ' '.join(state.lower().split())


def synthetic():
    result = {s: [] for s in SPLITS}
    registers = {
        'train': ('Case facts', 'Operations record', 'Assessment notes', 'Control worksheet'),
        'development': ('Reviewer briefing',),
        'calibration': ('Evidence summary',),
        'test': ('Independent adjudication packet',),
    }
    # Draw parameter families from separate RNG streams; every variant of a
    # scenario shares its split. Text registers are also held out as groups.
    for split in SPLITS:
        rng = random.Random(SEED + SPLITS.index(split))
        families = 160 if split == 'train' else 24
        for task in range(6):
            for f in range(families):
                family = f'phase2-{split}-task{task}-{f}'
                cap = rng.randrange(8, 81)
                tag = rng.choice(registers[split])
                entity = rng.choice(['supplier', 'borrower', 'custodian', 'counterparty', 'payment agent', 'fund'])
                for variant in range(3):
                    # 0/1/2 are labels balanced within every counterfactual family.
                    obs = cap + (-1 if variant == 0 else 0)
                    criteria = {'standard':'Proceed under the ordinary process.',
                                'enhanced':'Apply the additional control.',
                                'review':'Request clarification or a qualified human review.'}
                    target = ('standard', 'enhanced', 'review')[variant]
                    kind = 'choice'
                    if task == 0:
                        a, b, offset = rng.randrange(10, 400), rng.randrange(10, 400), rng.randrange(1, 30)
                        # Dollars, no floating point arithmetic.
                        threshold = a+b-offset + (1 if variant == 0 else 0)
                        missing = variant == 2
                        state = f'{tag}. {entity.title()} exposures: ${a} and ${b}. Eligible collateral: ' + ('not verified.' if missing else f'${offset}.')
                        instructions = f'Supplied policy: net exposure is the sum of both exposures minus eligible collateral. Net exposure below ${threshold} uses the ordinary process; at or above ${threshold} needs the additional control. Missing any input requires clarification.'
                        domain = 'net_exposure_arithmetic'
                        oracle = {'a':a,'b':b,'collateral':None if missing else offset,'threshold':threshold}
                    elif task == 1:
                        day = rng.randrange(1, 21); lag = cap
                        start = rng.randrange(0, 7)
                        # Count working days after receipt; weekends Sat/Sun, no holidays.
                        days = [d for d in range(1, lag+1) if (start+d)%7 < 5]
                        limit = len(days) + (1 if variant == 0 else 0)
                        state = f'{tag}. A notice to the {entity} arrived on weekday {start} (Monday=0). Today is {lag} calendar days after receipt. ' + ('The receipt date is disputed.' if variant == 2 else 'The receipt date is verified.')
                        instructions = f'Under this supplied policy, count working days strictly after receipt through today. Only Saturday=5 and Sunday=6 are nonworking; there are no holidays. Below {limit} working days uses ordinary processing; at least {limit} requires the additional control. A disputed receipt date requires review.'
                        domain = 'deadline_arithmetic'
                        oracle = {'start_weekday':start,'elapsed_days':lag,'working_days':len(days),'limit':limit,'disputed':variant==2}
                    elif task == 2:
                        total = rng.choice([100,200,400,1000]); pct = rng.choice([10,20,25,40,50,60,75])
                        amount = total*pct//100 - (1 if variant == 0 else 0)
                        state = f'{tag}. The {entity} portfolio has ${total} total value and ${amount} in one issuer. ' + ('Valuations are conflicting.' if variant == 2 else 'Both amounts are independently verified.')
                        instructions = f'Supplied policy: concentration is issuer value divided by total portfolio value. Below {pct}% uses the ordinary process; concentration of {pct}% or higher needs the additional control. Conflicting valuations require review. Equality is not below the limit.'
                        domain = 'concentration_arithmetic'
                        oracle = {'total':total,'issuer':amount,'percent_limit':pct,'conflict':variant==2}
                    elif task == 3:
                        # Vary the names of conditions and exception direction.
                        clause = rng.choice(['transfer', 'disclosure', 'assignment', 'subcontracting'])
                        exempt = rng.choice(['wholly owned affiliate', 'licensed auditor', 'named successor'])
                        requires = rng.choice([True, False])
                        event = variant != 2
                        exception = variant == 0 if requires else variant == 1
                        state = f'{tag}. Proposed {clause} by the {entity}. Counterparty is ' + (exempt if exception else 'an unrelated third party') + '. ' + ('Status is verified.' if event else 'Counterparty status cannot be established.')
                        instructions = f'Apply only this supplied contract rule: {clause} ' + ('requires' if requires else 'does not require') + ' prior written consent, except a transaction with a ' + exempt + ', which ' + ('does not require' if requires else 'requires') + ' consent. Use ordinary processing if consent is not required, additional control if required, and review if the exception status is unknown.'
                        domain = 'contract_exception'
                        oracle = {'default_requires_consent':requires,'exception_applies':exception,'unknown':not event}
                    elif task == 4:
                        older = cap + 4
                        fresh = obs
                        state = f'{tag}. Archived {entity} assessment: value {older}. Latest signed independent assessment: ' + ('unavailable; the draft contains conflicting estimates.' if variant == 2 else f'value {fresh}.') + ' A draft email suggests a different value. No approved override exists.'
                        instructions = f'Supplied evidence hierarchy: use only the latest signed independent assessment. Ignore archived values and draft emails. A verified value below {cap} uses ordinary processing; at least {cap} requires the additional control. If the signed assessment is unavailable or internally contradictory, request review.'
                        domain = 'evidence_precedence'
                        oracle = {'verified':None if variant==2 else fresh,'limit':cap}
                    else:
                        # Route labels describe declared capability, not measured vendor quality.
                        mode = (f+variant)%4
                        topics = ['fee schedule','settlement instructions','account terms','liquidity report','collateral schedule','covenant definitions','claim history','privacy notice']
                        topic = rng.choice(topics)
                        verb = rng.choice(['Locate','Retrieve','Find','Quote'])
                        analysis = rng.choice(['compare two inconsistent versions and explain the differences','calculate a scenario and justify the assumptions','reconcile the conflicting records and identify the cause','assess three alternatives against the supplied criteria'])
                        code = rng.choice(['write a tested Python function','debug the failing SQL query','implement a validator with tests','repair the data transformation script'])
                        action = rng.choice(['approve an exception','release a blocked payment','make the final legal determination','authorize an account restriction'])
                        text = [f'{verb} the current {topic}; return the documented passage with a citation.',f'Use the {topic} to {analysis}; return analysis only, without an approval.',f'For {topic}, {code}; return source code, without executing a transaction.',f'Review the {topic} and {action}; a binding authorization is requested.'][mode]
                        # Confound keywords: quoted background isn't the requested action.
                        background = rng.choice(['The file mentions Python and a prior approval, but those are background notes.', 'A colleague already retrieved the source documents.', 'The project title says analysis; select using the requested deliverable.'])
                        state = f'{tag}. {text} {background}'
                        criteria = {'lookup':'Retrieves documented facts and citations only; no synthesis or code.', 'analysis':'Performs comparisons, calculations and reasoning; cannot write source code or authorize actions.', 'code':'Writes or debugs software; cannot grant binding authorization.', 'human':'Qualified human with authority for approvals and binding decisions.'}
                        instructions = 'Choose the least-capable adequate route under these supplied capabilities. Route by the requested deliverable. Binding approvals require the human route; an analytical recommendation alone is not an approval. Background mentions do not change the request.'
                        target = ('lookup','analysis','code','human')[mode]
                        domain = 'capability_routing'
                        oracle = {'requested_deliverable':('retrieval','analysis','source code','binding approval')[mode]}
                    r = record(f'{family}-{variant}',domain,state,instructions,criteria,target,kind=kind,family=family)
                    r['split'] = split
                    r['source'] = {'generator':'phase2-rules-v1','human_reviewed':False,'oracle':oracle,'register':tag}
                    result[split].append(r)
    return result


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    historical = []
    for name in ['train','calibration','test','public-fresh','contrastive-banking-train','legacy-banking-128']:
        historical += [json.loads(x) for x in Path(f'data/private/{name}.jsonl').read_text().splitlines()]
    seen = {norm(r['state']) for r in historical}
    used = {(r['source']['repo'],r['source'].get('subset'),r['source']['native_split'],r['source']['row']) for r in historical if 'repo' in r.get('source',{})}
    result = synthetic(); rng = random.Random(SEED)
    for alias,subset,field,domain in [('banking77',None,'label','banking_intent'),('lex_glue','ledgar','label','legal_contract'),('lex_glue','unfair_tos','labels','legal_terms')]:
        source=SOURCES[alias]; ds=load_dataset(source['repo'],subset,revision=source['revision'])
        feature=ds['train'].features[field];names=feature.feature.names if field=='labels' else feature.names
        # Public test stays official test; dev/cal from remaining official validation
        # or remaining train where the source has no native validation.
        for split in SPLITS:
            native='test' if split=='test' else 'validation' if split!='train' and 'validation' in ds else 'train'
            pool=defaultdict(list)
            for i,row in enumerate(ds[native]):
                key=(source['repo'],subset,native,i)
                if key in used or norm(row['text']) in seen:continue
                group=row[field] if field=='label' else bool(row[field])
                pool[group].append((i,row))
            selected=[]
            # Stratify labels; for TOS include all predicates for a balanced sample
            # of clauses with / without any annotated unfairness. Not natural prevalence.
            per_label=(18 if split=='train' else 2 if split!='test' else 3) if field=='label' else (140 if split=='train' else 45)
            for _,rows in sorted(pool.items()):rng.shuffle(rows);selected.extend(rows[:per_label])
            for i,row in selected:
                if norm(row['text']) in seen:continue
                seen.add(norm(row['text']));used.add((source['repo'],subset,native,i))
                rid=f'phase2-{alias}-{subset}-{native}-{i}'
                if field=='label':
                    target=names[row[field]]
                    r=record(rid,domain,row['text'],'Choose the best matching banking intent.' if alias=='banking77' else 'Choose the contract clause topic.',sampled_options(names,target,rng),target)
                else:
                    positives=list(row[field]); k=rng.choice(positives) if positives and rng.random()<.7 else rng.randrange(len(names))
                    r=record(rid,domain,row['text'],f'Does this clause contain {names[k].lower()}?',None,'true' if k in positives else 'false',kind='noul')
                r['source']={**source,'subset':subset,'native_split':native,'row':i};r['split']=split;result[split].append(r)
    manifest={'seed':SEED,'historical_state_exclusions':len({norm(r['state']) for r in historical}),'splits':{},'limitations':['Rule-generated cases are not human reviewed or statements of law.','Public test contamination in pretrained models cannot be excluded.','Banking/LEDGAR are eight-candidate tasks; routing uses declared capabilities, not empirical model utility.','TOS selection is stratified by presence of annotations; it is not natural prevalence.','Template registers differ by split, but generators and logical operators are shared.']}
    all_seen=set();all_families=set()
    for split,rows in result.items():
        rng.shuffle(rows)
        states={norm(r['state']) for r in rows};families={r['family'] for r in rows}
        assert not states&all_seen and not families&all_families
        assert not states&{norm(r['state']) for r in historical}
        all_seen|=states;all_families|=families
        path=ROOT/f'{split}.jsonl';path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
        manifest['splits'][split]={'rows':len(rows),'unique_states':len(states),'families':len(families),'domains':dict(Counter(r['domain'] for r in rows)),'sha256':sha(path.read_bytes())}
    Path('data/phase2-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2))

if __name__=='__main__':main()
