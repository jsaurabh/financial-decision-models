"""A deliberately bounded text-only choice/noul/score decision interface."""
import json

MAX_CHOICES = 8
MAX_CHARS = 12000


def options(question, max_choices=MAX_CHOICES):
    kind = question.get('type')
    if kind == 'choice':
        values = question.get('criteria')
        if not isinstance(values, dict) or not 2 <= len(values) <= max_choices:
            raise ValueError(f'choice requires 2–{max_choices} named criteria')
        if not all(isinstance(k, str) and isinstance(v, str) for k, v in values.items()):
            raise ValueError('criteria must map string labels to descriptions')
        return list(values), list(values.values())
    if kind == 'noul':
        criteria = question.get('criteria')
        if criteria is None:
            return ['false', 'true'], ['The predicate is false.', 'The predicate is true.']
        if not isinstance(criteria, dict) or set(criteria) != {'false', 'true'} or not all(isinstance(v,str) for v in criteria.values()):
            raise ValueError('noul criteria must contain string descriptions for false and true')
        return ['false', 'true'], [criteria['false'], criteria['true']]
    if kind == 'score':
        values = question.get('criteria')
        if not isinstance(values, list) or not 2 <= len(values) <= max_choices:
            raise ValueError(f'score requires 2–{max_choices} ordered string descriptions')
        if not all(isinstance(v, str) for v in values):
            raise ValueError('score descriptions must be strings')
        return [str(i) for i in range(len(values))], values
    raise ValueError('supported types: choice, noul, score')


def render(state, question, max_choices=MAX_CHOICES, max_chars=MAX_CHARS):
    labels, descriptions = options(question, max_choices=max_choices)
    instructions = question.get('instructions')
    if not isinstance(instructions, str) or not instructions.strip():
        raise ValueError('instructions are required')
    state = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False, sort_keys=True)
    if len(state) + len(instructions) + sum(map(len, descriptions)) + sum(map(len, labels)) > max_chars:
        raise ValueError('request exceeds character budget')
    choices = '\n'.join(f'{i}: {label}: {desc}' for i, (label, desc) in enumerate(zip(labels, descriptions)))
    return f'State:\n{state}\nQuestion: {instructions}\nOptions:\n{choices}\nDecision:', labels


def answer(question, probabilities, abstain_threshold=0.8):
    labels, _ = options(question)
    if len(probabilities) != len(labels):
        raise ValueError('probability dimension mismatch')
    index = max(range(len(labels)), key=lambda i: probabilities[i])
    result = {'value': labels[index], 'probabilities': dict(zip(labels, probabilities)),
              'max_probability': probabilities[index],
              'needs_review': probabilities[index] < abstain_threshold}
    if question['type'] == 'noul':
        result['value'] = bool(index)
    elif question['type'] == 'score':
        result['value'] = sum(i * p for i, p in enumerate(probabilities))
        result['most_likely_level'] = index
    return result
