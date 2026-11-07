"""Text/JSON decision contracts; confidence uses TypeSafe's published formulas.

https://docs.typesafe.ai/confidence. OpenAI hosted confidence is not reproduced.
"""
import json
import math

from .schema import MAX_CHOICES, options, render

MAX_QUESTIONS = 8


def fields(value, allowed, required=()):
    if not isinstance(value, dict):
        raise ValueError('Expected an object')
    extra, missing = set(value) - set(allowed), set(required) - set(value)
    if extra or missing:
        raise ValueError(f'Unsupported fields: {sorted(extra)}; missing fields: {sorted(missing)}')


def text(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError('Expected a nonempty string')
    return value


def json_content(value):
    if not isinstance(value, (str, dict, list)):
        raise ValueError('Expected text or a JSON object/array')
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def systemone_request(body):
    fields(body, ('model', 'state', 'questions'), ('state', 'questions'))
    if 'model' in body:
        text(body['model'])
    json_content(body['state'])
    questions = body['questions']
    if not isinstance(questions, dict) or not 1 <= len(questions) <= MAX_QUESTIONS:
        raise ValueError(f'Provide 1–{MAX_QUESTIONS} named questions')
    normalized = {}
    for name, question in questions.items():
        text(name)
        fields(question, ('type', 'instructions', 'criteria'), ('type', 'instructions'))
        kind, criteria = question['type'], question.get('criteria')
        q = {'type': kind, 'instructions': text(json_content(question['instructions']))}
        if kind == 'choice':
            if not isinstance(criteria, dict) or not 2 <= len(criteria) <= MAX_CHOICES:
                raise ValueError(f'Choice requires 2–{MAX_CHOICES} criteria')
            q['criteria'] = {text(k): json_content(v) if v is not None else k for k, v in criteria.items()}
        elif kind == 'score':
            if not isinstance(criteria, list) or not 2 <= len(criteria) <= MAX_CHOICES:
                raise ValueError(f'Score requires 2–{MAX_CHOICES} ordered criteria')
            q['criteria'] = [json_content(v) if v is not None else f'Level {i}' for i, v in enumerate(criteria)]
        elif kind == 'noul':
            if criteria is not None:
                fields(criteria, ('true', 'false'))
                q['criteria'] = {k: json_content(criteria[k]) if criteria.get(k) is not None else default
                                 for k, default in [('false', 'The predicate is false.'), ('true', 'The predicate is true.')]}
        else:
            raise ValueError('Supported question types: choice, score, noul')
        render(body['state'], q)  # Validate every question before any model work.
        normalized[name] = q
    return {**body, 'questions': normalized}


def content_parts(value):
    if isinstance(value, str):
        return value
    if not isinstance(value, list) or not value:
        raise ValueError('Provide text or nonempty input_text/input_json content parts')
    parts = []
    for part in value:
        if not isinstance(part, dict):
            raise ValueError('Expected an input_text or input_json object')
        if part.get('type') == 'input_text':
            fields(part, ('type', 'text'), ('type', 'text'))
            if not isinstance(part['text'], str):
                raise ValueError('input_text.text must be a string')
            parts.append(part['text'])
        elif part.get('type') == 'input_json':
            fields(part, ('type', 'data'), ('type', 'data'))
            data = part['data']
            if not isinstance(data, dict) and not (isinstance(data, list) and all(isinstance(v, dict) for v in data)):
                raise ValueError('input_json.data must be an object or array of objects')
            parts.append(json_content(data))
        else:
            raise ValueError('Only input_text/input_json parts are supported; images/caching are not implemented')
    return '\n'.join(parts)


def decisions_request(body):
    fields(body, ('model', 'input', 'questions'), ('model', 'input', 'questions'))
    text(body['model'])
    state = body['input']
    if not isinstance(state, str):
        if not isinstance(state, list) or not state:
            raise ValueError('input must be text or a nonempty array of user messages')
        messages = []
        for message in state:
            fields(message, ('type', 'role', 'content'), ('content',))
            if message.get('type', 'message') != 'message' or message.get('role', 'user') != 'user':
                raise ValueError('Only user messages are supported; tool/conversation items are not implemented')
            messages.append(content_parts(message['content']))
        state = messages[0] if len(messages) == 1 else messages
    questions = body['questions']
    if not isinstance(questions, list) or not 1 <= len(questions) <= MAX_QUESTIONS:
        raise ValueError(f'Provide 1–{MAX_QUESTIONS} questions')
    mapped, mapping, names = {}, [], set()
    for index, question in enumerate(questions):
        fields(question, ('type', 'name', 'instructions', 'choices', 'levels'), ('type', 'instructions'))
        name = question.get('name')
        if 'name' in question:
            if not isinstance(name, str) or name in names:
                raise ValueError('Supplied question names must be unique strings')
            names.add(name)
        kind, key = question['type'], f'q{index}'
        q = {'type': 'noul' if kind == 'predicate' else kind, 'instructions': text(content_parts(question['instructions']))}
        meta = {'key': key, 'name': name, 'type': kind}
        if kind == 'predicate':
            if 'choices' in question or 'levels' in question:
                raise ValueError('Predicate questions have no choices or levels')
        elif kind == 'choice':
            if 'levels' in question:
                raise ValueError('Choice questions have no levels')
            choices = question.get('choices')
            if not isinstance(choices, list) or not 2 <= len(choices) <= MAX_CHOICES:
                raise ValueError(f'Provide 2–{MAX_CHOICES} choices')
            values, descriptions, seen = [], [], set()
            for choice in choices:
                fields(choice, ('value', 'description'), ('value',))
                value = choice['value']
                if type(value) not in (str, bool) or (type(value), value) in seen:
                    raise ValueError('Choice values must be unique typed strings or booleans')
                seen.add((type(value), value)); values.append(value)
                description = choice.get('description')
                if description is not None and not isinstance(description, str):
                    raise ValueError('Choice descriptions must be strings')
                descriptions.append(description)
            # Preserve string prompts; alias typed values so true and "true" cannot collide.
            keys = values if all(type(v) is str and v.strip() for v in values) else [f'option_{i}' for i in range(len(values))]
            q['criteria'] = {k: (d if d is not None else k) if keys == values else json.dumps(v) + (f': {d}' if d else '')
                             for k, v, d in zip(keys, values, descriptions, strict=True)}
            meta.update(keys=keys, values=values)
        elif kind == 'score':
            if 'choices' in question:
                raise ValueError('Score questions have no choices')
            levels = question.get('levels')
            if not isinstance(levels, list) or not 2 <= len(levels) <= MAX_CHOICES:
                raise ValueError(f'Provide 2–{MAX_CHOICES} score levels')
            labels, criteria = [], []
            for level in levels:
                fields(level, ('label', 'description'), ('label',))
                label, description = level['label'], level.get('description')
                if not isinstance(label, str) or (description is not None and not isinstance(description, str)):
                    raise ValueError('Score labels and descriptions must be strings')
                labels.append(label); criteria.append(label + (f': {description}' if description else ''))
            q['criteria'] = criteria; meta['labels'] = labels
        else:
            raise ValueError('Supported question types: predicate, choice, score')
        mapped[key] = q; mapping.append(meta)
    return systemone_request({'model': body['model'], 'state': state, 'questions': mapped}), mapping


def systemone_answer(question, probabilities):
    labels, descriptions = options(question)
    try:
        p = [float(v) for v in probabilities]
    except (ValueError, TypeError, OverflowError) as exc:
        raise RuntimeError('Backend returned invalid probabilities') from exc
    if len(p) != len(labels) or any(not math.isfinite(v) or not 0 <= v <= 1 for v in p) or not math.isclose(sum(p), 1, abs_tol=1e-5):
        raise RuntimeError('Backend returned an invalid probability distribution')
    total = sum(p); p = [v / total for v in p]
    kind = question['type']
    if kind == 'noul':
        return {'type': 'noul', 'noul': p[1]}
    best = max(range(len(p)), key=p.__getitem__)
    result = {'type': kind, 'probabilities': dict(zip(labels, p, strict=True))}
    if kind == 'choice':
        confidence = (p[best] - 1 / len(p)) / (1 - 1 / len(p))
        result.update(choice=labels[best], confidence=max(0., min(1., confidence)))
    else:
        baseline = sum(abs(i - (len(p) - 1) / 2) for i in range(len(p))) / len(p)
        confidence = 1 - sum(v * abs(i - best) for i, v in enumerate(p)) / baseline
        result.update(score=sum(i * v for i, v in enumerate(p)), confidence=max(0., min(1., confidence)),
                      legend=dict(zip(labels, descriptions, strict=True)))
    return result


def decisions_response(response, mapping):
    answers = []
    for meta in mapping:
        source = response['answers'][meta['key']]
        result = {'name': meta['name'], 'type': meta['type']}
        if source['type'] == 'refusal':
            result['type'] = 'refusal'
        elif meta['type'] == 'predicate':
            result['probability'] = source['noul']
        elif meta['type'] == 'choice':
            result.update(choice=meta['values'][meta['keys'].index(source['choice'])], confidence=source['confidence'],
                          probabilities=[{'value': v, 'probability': source['probabilities'][k]}
                                         for k, v in zip(meta['keys'], meta['values'], strict=True)])
        else:
            result.update(score=source['score'], confidence=source['confidence'], probabilities=[
                {'value': i, 'label': label, 'probability': source['probabilities'][str(i)]}
                for i, label in enumerate(meta['labels'])])
        answers.append(result)
    return {'model': response['model'], 'answers': answers, 'usage': response['usage']}
