"""Semantic classification and measured routing metrics for matched tasks."""
import math
from collections import Counter,defaultdict
import numpy as np
from sklearn.metrics import f1_score,confusion_matrix
from .schema import options

def labels(row):return options(row['question'],max_choices=128)[0]
def accepted(row):return row['metadata'].get('accepted',[row['target']])
def correct(row,p):return p is not None and p.get('status')=='ok' and p['prediction'] in accepted(row)
def meaning(row,label):return row['question']['criteria'][label] if row.get('task')=='legal_maud' else label

def metrics(rows,predictions):
    result={};domain_scores=defaultdict(list)
    for task in sorted({r['task'] for r in rows}):
        part=[r for r in rows if r['task']==task];pairs=[(r,predictions[r['id']]) for r in part if predictions.get(r['id'],{}).get('status')=='ok']
        m={'requested':len(part),'valid':len(pairs),'coverage':len(pairs)/len(part),'source_families':len({r['family'] for r in part}),
            'accuracy_missing_as_wrong':sum(correct(r,predictions.get(r['id'])) for r in part)/len(part),
            'errors':dict(Counter(predictions.get(r['id'],{}).get('error_type','missing') for r in part if predictions.get(r['id'],{}).get('status')!='ok'))}
        if part[0]['domain']=='model_routing':
            values=[];baseline=[];all_regret=[]
            for r in part:
                outcomes=r['metadata']['outcomes'];penalty=r['metadata']['cost_penalty'];utilities={k:v['quality']-penalty*v['cost_usd'] for k,v in outcomes.items()}
                best,worst=max(utilities.values()),min(utilities.values());p=predictions.get(r['id'],{})
                if p.get('status')!='ok':all_regret.append(1.);continue
                o=outcomes[p['prediction']];u=utilities[p['prediction']];regret=best-u
                all_regret.append(regret/max(best-worst,1e-12));values.append([o['quality'],o['cost_usd'],u,regret])
                profiles=r['state'].get('training_model_profiles',{})
                if profiles:
                    static=max(profiles,key=lambda k:profiles[k]['mean_quality']-penalty*profiles[k]['mean_cost_usd']);b=outcomes[static]
                    baseline.append([b['quality'],b['cost_usd'],b['quality']-penalty*b['cost_usd']])
            m['selection_score']=1-float(np.mean(all_regret))
            if values:m['routing_valid_only']=dict(zip(['quality','cost_usd','utility','oracle_regret'],np.mean(values,axis=0).tolist()))
            if baseline:m['training_profile_router_valid_only']=dict(zip(['quality','cost_usd','utility'],np.mean(baseline,axis=0).tolist()))
        else:
            units=defaultdict(list)
            for r in part:units[r['source'].get('review_question','')].append(r)
            fs=[]
            for unit,items in units.items():
                valid=[r for r in items if predictions.get(r['id'],{}).get('status')=='ok']
                if not valid:fs.append(0.);continue
                truth=[meaning(r,r['target']) for r in valid];pred=[meaning(r,predictions[r['id']]['prediction']) for r in valid]
                fs.append(float(f1_score(truth,pred,average='macro',zero_division=0))*len(valid)/len(items))
            m['semantic_macro_f1_coverage_penalized']=float(np.mean(fs));m['selection_score']=m['semantic_macro_f1_coverage_penalized']
            if pairs and all(p['labels']==['false','true'] for r,p in pairs):
                truth=np.array([r['target']=='true' for r,p in pairs]);pred=np.array([p['prediction']=='true' for r,p in pairs])
                m['binary_valid_only']={'recall':float(pred[truth].mean()) if truth.any() else None,'false_positive_rate':float(pred[~truth].mean()) if (~truth).any() else None}
            if task=='tool_action_safety' and pairs:
                truth=np.array([r['target']!='safe' for r,p in pairs]);pred=np.array([p['prediction']!='safe' for r,p in pairs])
                m['strict_safety_valid_only']={'unsafe_or_review_recall':float(pred[truth].mean()) if truth.any() else None,'safe_false_block_rate':float(pred[~truth].mean()) if (~truth).any() else None}
        if pairs:m['set_nll_valid_only']=float(np.mean([-math.log(max(sum(v for k,v in zip(p['labels'],p['probabilities']) if k in accepted(r)),1e-12)) for r,p in pairs]))
        result[task]=m;domain_scores[part[0]['domain']].append(m['selection_score'])
    domains={d:float(np.mean(v)) for d,v in domain_scores.items()}
    return {'tasks':result,'domain_selection_scores':domains,'selection_score':float(np.mean(list(domains.values()))),
        'selection_rule':'Equal four-domain score; classification uses coverage-penalized semantic macro-F1 (MAUD averaged across review questions); routing uses one minus normalized oracle regret. Validation only, not production traffic weighting.'}
