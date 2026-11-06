"""Run against a local merged artifact, with FastAPI's in-process test client."""
import json
import os
from pathlib import Path

from fastapi.testclient import TestClient
from finance_decisions.serve import app

client = TestClient(app)
health = client.get('/health'); assert health.status_code == 200
payload = {'state':'The file contains two verified required documents out of two.', 'questions':{
    'complete':{'type':'noul','instructions':'Are both required documents verified?'},
    'score':{'type':'score','instructions':'Score how many required documents are verified.',
             'criteria':['No documents verified','One document verified','Both documents verified']},
    'route':{'type':'choice','instructions':'Select the route appropriate for this complete file.',
             'criteria':{'process':'Required evidence is complete.','review':'Required evidence is missing.'}}}}
response = client.post('/v1/decisions',json=payload); assert response.status_code == 200, response.text
answers = response.json()['answers']; assert set(answers)=={'complete','score','route'}
assert isinstance(answers['complete']['value'],bool)
assert 0 <= answers['score']['value'] <= 2
assert answers['route']['value'] in ['process','review']
for answer in answers.values():
    assert abs(sum(answer['probabilities'].values()) - 1) < 1e-5
invalid = client.post('/v1/decisions',json={'state':'text','questions':{'bad':{'type':'unsupported','instructions':'Do something'}}})
assert invalid.status_code == 422
path = Path(os.environ.get('DECISION_RESULT_DIR','results/gemma4-finance-v3'))/'api-smoke.json'; path.parent.mkdir(parents=True,exist_ok=True)
result = {'status':'passed','health':health.json(),'typed_multi_question_response':response.json(),
          'invalid_schema_status':invalid.status_code,'artifact':os.environ.get('DECISION_ARTIFACT')}
path.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
