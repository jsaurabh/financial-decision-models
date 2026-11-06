import copy
import math
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

from finance_decisions.api_protocol import decisions_request, decisions_response, systemone_answer
from finance_decisions.schema import options
from finance_decisions.serve import create_app


class Backend:
    device = 'cpu'
    max_length = 2048

    def __init__(self):
        self.rows = []

    def predict(self, rows):
        self.rows.extend(copy.deepcopy(rows))
        probabilities = {2: [.2, .8], 3: [.1, .7, .2], 4: [.05, .6, .1, .25]}
        return [probabilities[len(options(row['question'])[0])] for row in rows], len(rows) * 17


@pytest.fixture
def api():
    backend = Backend()
    with TestClient(create_app(backend, model_id='finance-test')) as client:
        yield client, backend


def native_body():
    return {'model': 'finance-test', 'state': 'The customer disputes a payment.', 'questions': {
        'fraud': {'type': 'noul', 'instructions': 'Is review required?'},
        'route': {'type': 'choice', 'instructions': 'Choose a route.',
                  'criteria': {'payments': 'Payment problem', 'security': 'Security problem', 'other': 'Other'}},
        'severity': {'type': 'score', 'instructions': 'Rate severity.', 'criteria': ['Low', 'Medium', 'High']}}}


def openai_body():
    return {'model': 'finance-test', 'input': 'The customer disputes a payment.', 'questions': [
        {'type': 'predicate', 'name': 'fraud', 'instructions': 'Is review required?'},
        {'type': 'choice', 'name': 'route', 'instructions': 'Choose a route.', 'choices': [
            {'value': 'payments', 'description': 'Payment problem'},
            {'value': 'security', 'description': 'Security problem'}, {'value': 'other', 'description': 'Other'}]},
        {'type': 'score', 'name': 'severity', 'instructions': 'Rate severity.',
         'levels': [{'label': label} for label in ['Low', 'Medium', 'High']]}]}


def test_systemone_exact_typed_answers_and_usage(api):
    client, _ = api
    response = client.post('/v1/systemone', json=native_body())
    assert response.status_code == 200, response.text
    result = response.json()
    assert set(result) == {'model', 'answers', 'usage'}
    assert result['usage'] == {'input_tokens': 51, 'output_tokens': 0}
    assert result['answers']['fraud'] == {'type': 'noul', 'noul': .8}
    route, score = result['answers']['route'], result['answers']['severity']
    assert set(route) == {'type', 'choice', 'probabilities', 'confidence'}
    assert route['choice'] == 'security'
    assert route['confidence'] == pytest.approx(.55)
    assert route['probabilities'] == pytest.approx({'payments': .1, 'security': .7, 'other': .2})
    assert set(score) == {'type', 'score', 'legend', 'probabilities', 'confidence'}
    assert score['score'] == pytest.approx(1.1)  # Expected index, not the most likely level.
    assert score['confidence'] == pytest.approx(.55)
    assert score['legend'] == {'0': 'Low', '1': 'Medium', '2': 'High'}


def test_openai_exact_schema_and_both_routes_use_identical_model_inputs(api):
    client, backend = api
    native = client.post('/v1/systemone', json=native_body()).json()
    response = client.post('/v1/decisions', json=openai_body())
    assert response.status_code == 200, response.text
    result = response.json()
    assert backend.rows[:3] == backend.rows[3:]
    assert set(result) == {'model', 'answers', 'usage'}
    assert result['model'] == 'finance-test'
    assert result['usage'] == {'input_tokens': 51, 'output_tokens': 0, 'total_tokens': 51,
        'input_tokens_details': {'cached_tokens': 0, 'cache_write_tokens': 0},
        'output_tokens_details': {'reasoning_tokens': 0}}
    assert result['answers'][0] == {'name': 'fraud', 'type': 'predicate', 'probability': .8}
    route, score = result['answers'][1:]
    assert set(route) == {'name', 'type', 'choice', 'probabilities', 'confidence'}
    assert route['choice'] == native['answers']['route']['choice']
    assert route['probabilities'] == [dict(value=k, probability=v) for k, v in native['answers']['route']['probabilities'].items()]
    assert score['probabilities'] == [dict(value=i, label=label, probability=p)
        for i, (label, p) in enumerate(zip(['Low', 'Medium', 'High'], [.1, .7, .2]))]
    assert score['score'] == native['answers']['severity']['score']


def test_typed_choices_never_collapse_bool_and_string_values(api):
    client, backend = api
    body = openai_body(); body['questions'] = [{'type': 'choice', 'instructions': 'Select.',
        'choices': [{'value': v} for v in [True, 'true', False, 'false']]}]
    answer = client.post('/v1/decisions', json=body).json()['answers'][0]
    assert answer['name'] is None
    assert answer['choice'] == 'true' and type(answer['choice']) is str
    assert [p['value'] for p in answer['probabilities']] == [True, 'true', False, 'false']
    assert len(backend.rows[0]['question']['criteria']) == 4
    body['questions'][0]['choices'] = [{'value': False}, {'value': True}]
    answer = client.post('/v1/decisions', json=body).json()['answers'][0]
    assert answer['choice'] is True


def test_text_json_content_and_partial_noul_criteria(api):
    client, backend = api
    body = openai_body()
    body['input'] = [{'role': 'user', 'content': [{'type': 'input_text', 'text': 'Evidence:'},
        {'type': 'input_json', 'data': {'verified': True}}]}]
    body['questions'] = [{'type': 'predicate', 'name': '', 'instructions': [
        {'type': 'input_text', 'text': 'Is it verified?'}]}]
    result = client.post('/v1/decisions', json=body)
    assert result.status_code == 200
    assert backend.rows[-1]['state'] == 'Evidence:\n{"verified": true}'
    assert result.json()['answers'][0]['name'] == ''
    body = native_body()
    body['state'] = {'verified': True}
    body['questions'] = {'q': {'type': 'noul', 'instructions': {'question': 'Verified?'}, 'criteria': {'true': 'Verified'}}}
    assert client.post('/v1/systemone', json=body).status_code == 200
    assert backend.rows[-1]['question']['criteria'] == {'true': 'Verified', 'false': 'The predicate is false.'}


@pytest.mark.parametrize('change', [
    lambda b: b.update(extra=True),
    lambda b: b.update(prompt_cache_key='unsupported'),
    lambda b: b.update(questions=[]),
    lambda b: b.update(questions=b['questions'] * 3),
    lambda b: b.update(input='x' * 13000),
    lambda b: b.update(input=[{'role': 'user', 'content': [{'type': 'input_image', 'image_url': 'https://example.com/a.png'}]}]),
    lambda b: b.update(input=[{'type': 'function_call', 'name': 'transfer'}]),
    lambda b: b['questions'][0].update(name='route'),
    lambda b: b['questions'][0].update(choices=[]),
    lambda b: b['questions'][1].update(choices=[{'value': 'same'}, {'value': 'same'}]),
    lambda b: b['questions'][1].update(choices=[{'value': 1}, {'value': 2}]),
    lambda b: b['questions'][1].update(choices=[{'value': str(i)} for i in range(9)]),
    lambda b: b['questions'][2].update(levels=[{'label': 'low', 'value': 7}, {'label': 'high'}]),
])
def test_unsupported_requests_rejected_before_inference(api, change):
    client, backend = api
    body = openai_body(); change(body)
    assert client.post('/v1/decisions', json=body).status_code == 422
    assert not backend.rows


def test_unknown_model_and_native_extensions_are_explicit(api):
    client, backend = api
    body = openai_body(); body['model'] = 'gpt-6-luna'
    assert client.post('/v1/decisions', json=body).status_code == 404
    body = native_body(); body['assets'] = {'image': {'type': 'image'}}
    assert client.post('/v1/systemone', json=body).status_code == 422
    body = native_body(); body['questions']['route']['criteria'] = {'a': 'one'}
    assert client.post('/v1/systemone', json=body).status_code == 422
    assert not backend.rows


def test_legacy_review_fields_remain_available_on_classify(api):
    client, _ = api
    body = native_body(); body['review_threshold'] = .9
    response = client.post('/v1/classify', json=body)
    assert response.status_code == 200
    assert response.json()['answers']['fraud']['value'] is True
    assert response.json()['answers']['fraud']['needs_review'] is True
    assert response.json()['shared_prefill'] is False


@pytest.mark.parametrize('distribution', [[math.nan, .5], [-.1, 1.1], [.1, .1], [.2, .8, 0], ['invalid', .5]])
def test_invalid_backend_result_is_500_and_capacity_is_released(api, distribution):
    client, backend = api
    original = backend.predict
    backend.predict = lambda rows: ([distribution] * len(rows), 0)
    assert client.post('/v1/systemone', json=native_body()).status_code == 500
    backend.predict = original
    assert client.post('/v1/systemone', json=native_body()).status_code == 200


def test_gemma_backend_applies_temperature_once_and_counts_packed_tokens():
    import torch
    from finance_decisions.serve import GemmaBackend

    class Model:
        temperature = 2.
        def prepare(self, rows):
            return {'input_ids': torch.ones((1, 13), dtype=torch.long)}
        def __call__(self, batch, counts):
            assert not torch.is_grad_enabled()
            assert counts == [2]
            return torch.tensor([[0., 2 * math.log(4)]])

    backend = GemmaBackend.__new__(GemmaBackend)
    backend.model = Model()
    body = {'model': 'finance-test', 'state': 'Evidence', 'questions': {
        'a': {'type': 'noul', 'instructions': 'Verified?'},
        'b': {'type': 'choice', 'instructions': 'Choose.', 'criteria': {'no': 'No', 'yes': 'Yes'}}}}
    with TestClient(create_app(backend, model_id='finance-test')) as client:
        result = client.post('/v1/systemone', json=body).json()
    assert result['answers']['a']['noul'] == pytest.approx(.8)
    assert result['answers']['b']['probabilities']['yes'] == pytest.approx(.8)
    assert result['usage'] == {'input_tokens': 26, 'output_tokens': 0}


def test_busy_request_does_not_start_another_inference(api):
    client, backend = api
    started, release = threading.Event(), threading.Event()
    original = backend.predict
    def slow(rows):
        started.set(); assert release.wait(5)
        return original(rows)
    backend.predict = slow
    with ThreadPoolExecutor(max_workers=1) as pool:
        first = pool.submit(client.post, '/v1/systemone', json=native_body())
        try:
            assert started.wait(5)
            assert client.post('/v1/decisions', json=openai_body()).status_code == 503
        finally:
            release.set()
        assert first.result().status_code == 200


def test_confidence_boundaries_expected_score_and_refusal_bridge():
    choice = {'type': 'choice', 'criteria': {'a': 'A', 'b': 'B', 'c': 'C'}}
    assert systemone_answer(choice, [1/3] * 3)['confidence'] == pytest.approx(0)
    assert systemone_answer(choice, [0, 1, 0])['confidence'] == 1
    score = {'type': 'score', 'criteria': ['low', 'medium', 'high']}
    assert systemone_answer(score, [.5, 0, .5])['confidence'] == 0
    assert systemone_answer(score, [0, .5, .5])['confidence'] == pytest.approx(.25)
    _, mapping = decisions_request(openai_body())
    response = decisions_response({'model': 'finance-test', 'usage': {},
        'answers': {m['key']: {'type': 'refusal'} for m in mapping}}, mapping)
    assert all(a['type'] == 'refusal' and set(a) == {'name', 'type'} for a in response['answers'])
