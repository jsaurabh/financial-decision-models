"""Per-task metrics and complete-coverage accounting for public evaluation."""
from collections import Counter,defaultdict
import math
import numpy as np
from sklearn.metrics import f1_score,precision_recall_fscore_support,average_precision_score


def correct(row,pred):return pred in row['metadata'].get('accepted',[row['target']])

def score(rows,records):
    byid={r['id']:r for r in records};result={}
    for suite in sorted({r['suite'] for r in rows}):
        expected=[r for r in rows if r['suite']==suite]
        pairs=[(r,byid[r['id']]) for r in expected if r['id'] in byid and byid[r['id']].get('status')=='ok']
        isrouter=suite.startswith('routerbench')
        m={'requested':len(expected),'valid':len(pairs),'coverage':len(pairs)/len(expected),
           'accuracy_missing_as_wrong':sum(correct(r,p['prediction']) for r,p in pairs)/len(expected),
           'errors':dict(Counter(byid.get(r['id'],{}).get('error_type','missing') for r in expected if byid.get(r['id'],{}).get('status')!='ok'))}
        if pairs:
            m['accuracy_valid_only']=float(np.mean([correct(r,p['prediction']) for r,p in pairs]))
            nll=[]
            for r,p in pairs:
                accepted=set(r['metadata'].get('accepted',[r['target']]))
                nll.append(-math.log(max(sum(v for k,v in zip(p['labels'],p['probabilities']) if k in accepted),1e-12)))
            m['set_aware_nll_valid_only']=float(np.mean(nll))
            if isrouter:
                utility=[];quality=[];cost=[];regret=[]
                for r,p in pairs:
                    o=r['metadata']['outcomes'];chosen=o[p['prediction']];penalty=r['metadata']['cost_penalty']
                    value=chosen['quality']-penalty*chosen['cost_usd'];best=max(x['quality']-penalty*x['cost_usd'] for x in o.values())
                    quality.append(chosen['quality']);cost.append(chosen['cost_usd']);utility.append(value);regret.append(max(0,best-value))
                m['routing_valid_only']={'chosen_quality':float(np.mean(quality)),'chosen_cost_usd':float(np.mean(cost)),
                    'utility':float(np.mean(utility)),'oracle_regret':float(np.mean(regret)),
                    'note':'Historical released outcomes; missing decisions excluded from utility/cost and counted wrong in oracle-optimal accuracy.'}
            else:
                y=[r['target'] for r,p in pairs];pred=[p['prediction'] for r,p in pairs]
                m['macro_f1_valid_only']=float(f1_score(y,pred,average='macro',zero_division=0))
                if suite.startswith('tsbench'):
                    for name,unsafe in [('strict',{'controversial','unsafe'}),('loose',{'unsafe'})]:
                        truth=[r['target'] in unsafe for r,p in pairs];guess=[p['prediction'] in unsafe for r,p in pairs]
                        pp,rr,ff,_=precision_recall_fscore_support(truth,guess,average='binary',zero_division=0)
                        m[name+'_binary_valid_only']={'unsafe_precision':float(pp),'unsafe_recall':float(rr),'unsafe_f1':float(ff),
                            'unsafe_miss_rate':sum(t and not g for t,g in zip(truth,guess))/max(1,sum(truth)),
                            'safe_false_block_rate':sum(not t and g for t,g in zip(truth,guess))/max(1,sum(not t for t in truth))}
                if all(p['labels']==['false','true'] for r,p in pairs):
                    truth=[r['target']=='true' for r,p in pairs];guess=[p['prediction']=='true' for r,p in pairs]
                    pp,rr,ff,_=precision_recall_fscore_support(truth,guess,average='binary',zero_division=0)
                    m['positive_valid_only']={'precision':float(pp),'recall':float(rr),'f1':float(ff),
                        'average_precision':float(average_precision_score(truth,[p['probabilities'][1] for r,p in pairs]))}
            confidence=np.array([max(p['probabilities']) for r,p in pairs]);hit=np.array([correct(r,p['prediction']) for r,p in pairs])
            m['selective_valid_only']={str(t):{'coverage_of_valid':float(np.mean(confidence>=t)),
                'accuracy':float(np.mean(hit[confidence>=t])) if (confidence>=t).any() else None} for t in [.7,.8,.9,.95]}
        if suite=='contractnli':
            groups=defaultdict(list)
            for r in expected:groups[r['family']].append(r)
            exact=sum(all(byid.get(r['id'],{}).get('status')=='ok' and correct(r,byid[r['id']]['prediction']) for r in group) for group in groups.values())
            m['document_exact_match_missing_as_wrong']=exact/len(groups)
        result[suite]=m
    return result
