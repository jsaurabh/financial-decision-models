"""Measure exact untruncated Gemma lengths; no model inference or label filtering."""
import json
from pathlib import Path
from types import SimpleNamespace
from collections import defaultdict
from transformers import AutoTokenizer
from torch import nn
from finance_decisions.candidate_model import CandidateDecisionModel
from finance_decisions.model import BASE,REVISION
from finance_decisions.public_eval import read
b=nn.Linear(1,1);b.config=SimpleNamespace(hidden_size=1,sliding_window=512)
m=CandidateDecisionModel(b,AutoTokenizer.from_pretrained(BASE,revision=REVISION),8192);m.max_choices=11;m.max_chars=1000000
stats=defaultdict(list)
for r in read('data/private/public-eval/rows.jsonl'):stats[r['suite']].append(m.token_length(r))
result={k:{'n':len(v),'within_2048':sum(n<=2048 for n in v),'within_8192':sum(n<=8192 for n in v),'max_tokens':max(v)} for k,v in stats.items()}
Path('data/public-eval-lengths.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
