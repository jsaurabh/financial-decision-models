"""Matched evaluation and family-cluster uncertainty; no test-based selection."""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import binomtest
from .train import read
from .compare_quality import pair_records
from .phase2_eval import aggregate


def paired(a,b):
    if len(a) != len(b) or {r['id'] for r in a} != {r['id'] for r in b}:
        raise ValueError('Paired comparison requires identical complete coverage')
    expected = len(a)
    a,b=pair_records(a,b)
    if len(a) != expected:
        raise ValueError('Unexpected missing or duplicate comparison inputs')
    groups=defaultdict(list)
    for x,y in zip(a,b):groups[x['family']].append(int(y['correct'])-int(x['correct']))
    sums=np.array([sum(v) for v in groups.values()]);sizes=np.array([len(v) for v in groups.values()])
    rng=np.random.default_rng(20261108);stats=[]
    for _ in range(10000):
        ix=rng.integers(0,len(groups),len(groups));stats.append(float(sums[ix].sum()/sizes[ix].sum()))
    bw=sum(not x['correct'] and y['correct'] for x,y in zip(a,b));aw=sum(x['correct'] and not y['correct'] for x,y in zip(a,b))
    return {'n':len(a),'families':len(groups),'b_minus_a_accuracy':float(sums.sum()/sizes.sum()),
            'family_bootstrap_95pct':np.quantile(stats,[.025,.975]).tolist(),'a_only_correct':aw,'b_only_correct':bw,
            'case_level_exact_mcnemar_p':float(binomtest(bw,aw+bw).pvalue) if aw+bw else 1.,
            'limits':'Related rule templates and public pretraining exposure limit interpretation; case-level p ignores family clustering.'}


def main():
    root=Path('results/phase2');names=['v4','causal','joint','qwen','jev']
    predictions={name:read(root/name/'test-predictions.jsonl') for name in names}
    results={name:aggregate(rows) for name,rows in predictions.items()}
    results['paired']={f'{a}_to_{b}':paired(predictions[a],predictions[b]) for a,b in [('causal','joint'),('v4','causal'),('v4','joint'),('joint','qwen'),('joint','jev'),('qwen','jev')]}
    # Fixed selection rule: uncalibrated development NLL, never test accuracy.
    # Preserve the choice frozen by training; inference kernels can produce
    # small BF16/TF32 differences and must not reselect after test access.
    dev={n:json.loads((root/n/'selection.json').read_text())['selected_dev_nll'] for n in ['causal','joint']}
    results['gemma_selection']={'criterion':'uncalibrated development NLL','development_nll':dev,'selected':min(dev,key=dev.get)}
    (root/'comparison.json').write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps({'accuracy':{n:results[n]['accuracy'] for n in names},'selection':results['gemma_selection'],'causal_joint':results['paired']['causal_to_joint']},indent=2))

if __name__=='__main__':main()
