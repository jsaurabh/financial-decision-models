"""Paired source-family inference with fixed mixture and multiplicity control."""
import math
from collections import Counter
import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score

SEED=20261114
DOMAINS=['legal','oos','financial_verification','injection_detection']

def hit(row,p):
    return p is not None and p.get('status')=='ok' and p['prediction'] in row.get('metadata',{}).get('accepted',[row['target']])

def holm(values):
    ordered=sorted(values,key=values.get);adjusted={};previous=0
    for i,name in enumerate(ordered):
        previous=max(previous,min(1.,values[name]*(len(ordered)-i)));adjusted[name]=previous
    return adjusted

def paired_cluster(rows,delta,reps=10000,seed=SEED,task_weights=None):
    """Resample shared source families, preserving paired rows and task mixing."""
    tasks=sorted({r['suite'] for r in rows});families=sorted({r['family'] for r in rows})
    if not rows:return {'difference':None,'ci95':None,'p_two_sided':None,'families':0}
    ti={t:i for i,t in enumerate(tasks)};fi={f:i for i,f in enumerate(families)}
    sums=np.zeros((len(families),len(tasks)));counts=np.zeros_like(sums)
    for row,value in zip(rows,delta,strict=True):
        i,j=fi[row['family']],ti[row['suite']];sums[i,j]+=value;counts[i,j]+=1
    weights=np.array([task_weights[t] if task_weights else 1/len(tasks) for t in tasks]);weights/=weights.sum()
    observed=float((sums.sum(0)/counts.sum(0))@weights)
    if len(families)<2:return {'difference':observed,'ci95':None,'p_two_sided':1.,'families':len(families),'reason':'insufficient_independent_families'}
    rng=np.random.default_rng(seed);draws=[];extreme=0;accepted=0
    family_contribution=(sums/counts.sum(0))@weights
    for begin in range(0,reps,100):
        n=min(100,reps-begin)
        sampled=rng.multinomial(len(families),np.full(len(families),1/len(families)),size=n)
        denom=sampled@counts;ok=(denom>0).all(axis=1)
        draws.extend(((sampled[ok]@sums/denom[ok])@weights).tolist());accepted+=int(ok.sum())
        signs=2*rng.integers(0,2,size=(n,len(families)),dtype=np.int8)-1
        extreme+=int((np.abs(signs@family_contribution)>=abs(observed)-1e-14).sum())
    return {'difference':observed,'ci95':np.quantile(draws,[.025,.975]).tolist() if draws else None,
        'p_two_sided':(extreme+1)/(reps+1),'families':len(families),'bootstrap_replicates':accepted,'requested_replicates':reps,
        'method':'paired source-family cluster bootstrap; equal task weights unless declared; two-sided family sign randomization'}

def system_metrics(rows,predictions):
    valid=[(r,predictions[r['id']]) for r in rows if predictions.get(r['id'],{}).get('status')=='ok']
    result={'requested':len(rows),'valid':len(valid),'coverage':len(valid)/len(rows),
        'accuracy_missing_as_wrong':sum(hit(r,predictions.get(r['id'])) for r in rows)/len(rows),
        'errors':dict(Counter(predictions.get(r['id'],{}).get('error_type','missing') for r in rows if predictions.get(r['id'],{}).get('status')!='ok'))}
    if not valid:return result
    y=[r['target'] for r,p in valid];guess=[p['prediction'] for r,p in valid]
    keys=sorted(set(y+guess));result['confusion_valid_only']={'labels':keys,'matrix':confusion_matrix(y,guess,labels=keys).tolist()}
    result['macro_f1_valid_only']=float(f1_score(y,guess,average='macro',zero_division=0))
    result['accuracy_valid_only']=float(np.mean([hit(r,p) for r,p in valid]))
    result['nll_valid_only']=float(np.mean([-math.log(max(sum(v for k,v in zip(p['labels'],p['probabilities'],strict=True) if k in r.get('metadata',{}).get('accepted',[r['target']])),1e-12)) for r,p in valid]))
    if all(p['labels']==['false','true'] for r,p in valid):
        truth=np.array([r['target']=='true' for r,p in valid]);pred=np.array([p['prediction']=='true' for r,p in valid])
        result['binary_valid_only']={'positive':int(truth.sum()),'negative':int((~truth).sum()),
            'recall':float(pred[truth].mean()) if truth.any() else None,
            'false_positive_rate':float(pred[~truth].mean()) if (~truth).any() else None,
            'average_precision':float(average_precision_score(truth,[p['probabilities'][1] for r,p in valid])) if truth.any() else None}
    confidence=np.array([max(p['probabilities']) for r,p in valid]);correct=np.array([hit(r,p) for r,p in valid])
    result['selective_valid_only']={str(t):{'coverage_of_valid':float((confidence>=t).mean()),'accuracy':float(correct[confidence>=t].mean()) if (confidence>=t).any() else None} for t in [.7,.8,.9,.95]}
    return result

def routing(rows,g,j,reps=10000):
    pairs=[r for r in rows if g.get(r['id'],{}).get('status')==j.get(r['id'],{}).get('status')=='ok']
    result={'requested':len(rows),'common_valid':len(pairs),'coverage':len(pairs)/len(rows)}
    if not pairs:return result
    values={name:{key:[] for key in ['quality','cost_usd','utility','oracle_regret']} for name in ['gemma','jev']}
    for row in pairs:
        outcomes=row['metadata']['outcomes'];penalty=row['metadata']['cost_penalty']
        oracle=max(v['quality']-penalty*v['cost_usd'] for v in outcomes.values())
        for name,pred in [('gemma',g),('jev',j)]:
            o=outcomes[pred[row['id']]['prediction']];utility=o['quality']-penalty*o['cost_usd']
            for key,value in [('quality',o['quality']),('cost_usd',o['cost_usd']),('utility',utility),('oracle_regret',max(0,oracle-utility))]:values[name][key].append(value)
    result['common_valid_means']={name:{key:float(np.mean(v)) for key,v in metrics.items()} for name,metrics in values.items()}
    result['paired_differences']={key:paired_cluster(pairs,np.array(values['gemma'][key])-values['jev'][key],reps=reps) for key in values['gemma']}
    result['note']='Restricted three-model historical cached replay. Missing routing decisions are excluded from cost/utility, never counted as zero-cost successes.'
    return result

def compare(rows,g,j,reps=10000):
    suites={}
    for name in sorted({r['suite'] for r in rows}):
        part=[r for r in rows if r['suite']==name]
        entry={'cohort':part[0]['cohort'],'domain':part[0]['domain'],'families':len({r['family'] for r in part}),
            'gemma':system_metrics(part,g),'jev':system_metrics(part,j)}
        delta=[int(hit(r,g.get(r['id'])))-int(hit(r,j.get(r['id']))) for r in part]
        entry['paired_accuracy']=paired_cluster(part,delta,reps=reps)
        if all('outcomes' in r.get('metadata',{}) and 'cost_penalty' in r['metadata'] for r in part):
            entry['routing']=routing(part,g,j,reps=reps)
            # Arbitrary canonical tie labels are not semantic routing F1.
            for system in ['gemma','jev']:
                entry[system].pop('macro_f1_valid_only',None);entry[system].pop('confusion_valid_only',None)
        suites[name]=entry
    domains={}
    for name in DOMAINS:
        part=[r for r in rows if r['cohort']=='fresh' and r['domain']==name]
        if not part:domains[name]={'status':'no_eligible_cases','supports_superiority':False};continue
        entry=paired_cluster(part,[int(hit(r,g.get(r['id'])))-int(hit(r,j.get(r['id']))) for r in part],reps=reps)
        entry.update(requested=len(part),gemma_valid=sum(g.get(r['id'],{}).get('status')=='ok' for r in part),jev_valid=sum(j.get(r['id'],{}).get('status')=='ok' for r in part))
        domains[name]=entry
    adjusted=holm({name:domains[name].get('p_two_sided',1.) for name in DOMAINS})
    for name,entry in domains.items():
        entry['holm_p']=adjusted[name]
        entry['supports_superiority']=bool(entry.get('ci95') and entry['ci95'][0]>0 and adjusted[name]<.05 and entry['gemma_valid']>=entry['jev_valid'])
    fresh=[r for r in rows if r['cohort']=='fresh'];weights={}
    available={r['domain'] for r in fresh}
    for domain in available:
        tasks={r['suite'] for r in fresh if r['domain']==domain}
        for task in tasks:weights[task]=1/(len(available)*len(tasks))
    mixture=paired_cluster(fresh,[int(hit(r,g.get(r['id'])))-int(hit(r,j.get(r['id']))) for r in fresh],reps=reps,task_weights=weights)
    return {'suites':suites,'confirmatory_domains':domains,'equal_domain_research_mixture':mixture,
        'broad_classification_superiority_supported':all(domains[name]['supports_superiority'] for name in DOMAINS),
        'claim_scope':'Four represented public classification domains only. Routing, attack-only recall and inspected regressions are separate. No action-safety or production-financial-services superiority claim.'}
