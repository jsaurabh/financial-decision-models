"""Run against a local merged artifact, with FastAPI's in-process test client."""
import json
import os
from pathlib import Path

from fastapi.testclient import TestClient
from finance_decisions.serve import app

with TestClient(app) as client:
    health = client.get('/health'); assert health.status_code == 200
    payload = {'model':health.json()['model'], 'state':'The file contains two verified required documents out of two.', 'questions':{
        'complete':{'type':'noul','instructions':'Are both required documents verified?'},
        'score':{'type':'score','instructions':'Score how many required documents are verified.',
                 'criteria':['No documents verified','One document verified','Both documents verified']},
        'route':{'type':'choice','instructions':'Select the route appropriate for this complete file.',
                 'criteria':{'process':'Required evidence is complete.','review':'Required evidence is missing.'}}}}
    response = client.post('/v1/systemone',json=payload); assert response.status_code == 200, response.text
    answers = response.json()['answers']; assert set(answers)=={'complete','score','route'}
    assert 0 <= answers['complete']['noul'] <= 1
    assert 0 <= answers['score']['score'] <= 2
    assert answers['route']['choice'] in ['process','review']
    for name in ['score','route']:
        assert abs(sum(answers[name]['probabilities'].values()) - 1) < 1e-5
    questions = []
    for name, q in payload['questions'].items():
        mapped = {'type':'predicate' if q['type']=='noul' else q['type'], 'name':name, 'instructions':q['instructions']}
        if q['type']=='choice':mapped['choices']=[{'value':k,'description':v} for k,v in q['criteria'].items()]
        elif q['type']=='score':mapped['levels']=[{'label':v} for v in q['criteria']]
        questions.append(mapped)
    openai = client.post('/v1/decisions',json={'model':payload['model'],'input':payload['state'],'questions':questions})
    assert openai.status_code == 200, openai.text
    assert [a['type'] for a in openai.json()['answers']] == ['predicate','score','choice']
    assert 0 <= openai.json()['answers'][0]['probability'] <= 1
    for value in openai.json()['answers'][1:]:
        assert abs(sum(p['probability'] for p in value['probabilities']) - 1) < 1e-5
    assert openai.json()['usage']['output_tokens'] == 0
    legacy = client.post('/v1/classify',json=payload); assert legacy.status_code == 200
    assert isinstance(legacy.json()['answers']['complete']['value'],bool)
    invalid = client.post('/v1/systemone',json={'state':'text','questions':{'bad':{'type':'unsupported','instructions':'Do something'}}})
    assert invalid.status_code == 422
path = Path(os.environ.get('DECISION_RESULT_DIR','results/gemma4-finance-v3'))/'api-smoke.json'; path.parent.mkdir(parents=True,exist_ok=True)
result = {'status':'passed','health':health.json(),'systemone_response':response.json(),'decisions_response':openai.json(),
          'invalid_schema_status':invalid.status_code,'artifact':os.environ.get('DECISION_ARTIFACT')}
path.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
