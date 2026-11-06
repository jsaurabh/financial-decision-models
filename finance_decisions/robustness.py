"""Label-preserving training paraphrases and a separate fresh robustness probe."""
import copy
import json
import random
import re
from pathlib import Path

PATTERN = re.compile(r'^Case (\d+): observed (.*): (unknown|\d+)\. Evidence is current\. No override applies\.$')
MISSING = ['pending verification', 'not provided', 'unavailable', 'not recorded',
           'missing from the file', 'not measurable from available data']
ROUTING_PARAPHRASES = {
    'lookup': ['Retrieve the published limits for a savings transfer.', 'Locate the mortgage servicing policy.',
               'Find the documented escalation contact.', 'Fetch the documented eligibility checklist.',
               'Find the published deposit protection terms.', 'Locate our written policy on account closure.'],
    'reasoning': ['Explain why these sets of financial positions disagree.', 'Evaluate competing readings of a loan covenant.',
                  'Analyze the cause of a mismatch in settlement records.', 'Assess how several risk factors interact.',
                  'Compare two proposed controls and explain their tradeoffs.', 'Investigate the implications of inconsistent contract evidence.'],
    'code': ['Produce a program to validate payment-message fields.', 'Fix a bug in the cash reconciliation service.',
             'Add automated checks for a sanctions screening function.', 'Refactor a script that parses broker statements.',
             'Implement property-based testing for a transfer function.', 'Develop a program to compute loan amortization.'],
    'human': ['Sign off a regulatory exception.', 'Authorize an override of a hard compliance block.',
              'Make the binding decision to end a customer relationship.', 'Exercise final approval on a disputed legal obligation.',
              'Approve the release of a frozen account.', 'Provide the mandatory human approval for this transaction.'],
}


def augment(row, rng):
    row = copy.deepcopy(row)
    if row.get('domain') == 'model_routing':
        row['state'] = rng.choice(ROUTING_PARAPHRASES[row['target']])
        return row
    if not isinstance(row['state'], str):
        return row
    match = PATTERN.match(row['state'])
    if match:
        case, noun, value = match.groups()
        value = rng.choice(MISSING) if value == 'unknown' else value
        templates = [
            f'Account file {case}. The {noun} reading is {value}. No exception is documented.',
            f'Operational evidence: {noun} -> {value}. Evaluate the supplied rule on this current evidence.',
            f'Latest audit for record {case}: {noun} = {value}; overrides: none.',
            f'Input ({noun}): {value}. Apply the stated policy boundary exactly.',
        ]
        row['state'] = rng.choice(templates)
    return row


def fresh_probe():
    """Final-only probe: unseen numeric limits, domains, and state wordings.

    This small synthetic probe is not used for checkpoint selection or calibration.
    It supplements the earlier test that has already informed the development plan.
    """
    rng = random.Random(6112026); rows = []
    domains = ['treasury_settlement', 'loan_servicing', 'brokerage_operations', 'claims_controls']
    unseen_missing = ['cannot be determined from the record', 'the record contains no usable measurement',
                      'the data feed did not return a result', 'unresolved']
    for domain in domains:
        for i in range(60):
            cap = rng.choice([2,4,8,12,17,33,64])
            for role, value in [('below',cap-1),('equal',cap),('above',cap+1),('missing',None)]:
                actual = rng.choice(unseen_missing) if value is None else str(value)
                state = f'You are reviewing a {domain.replace("_"," ")} file. The measured quantity is {actual}. The applicable limit is {cap}; there are no exception instructions.'
                criteria = {'routine': f'The quantity is known and strictly less than {cap}.',
                            'additional_checks': f'The quantity is known and equal to or greater than {cap}.',
                            'investigate': 'The quantity is not known well enough to compare with the limit.'}
                pairs = list(criteria.items()); rng.shuffle(pairs)
                rows.append({'id':f'fresh-{domain}-{i}-{role}', 'family':f'fresh-{domain}-{i}',
                    'domain':domain, 'state':state,
                    'question':{'type':'choice','instructions':'Follow the supplied hypothetical policy and choose the warranted workflow. Treat absent evidence as unknown.', 'criteria':dict(pairs)},
                    'target':'investigate' if value is None else 'routine' if value<cap else 'additional_checks',
                    'source':{'generator':'fresh-robustness-v1','human_reviewed':False,'role':role}})
    path = Path('data/private/fresh-robustness.jsonl')
    path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    return rows


if __name__ == '__main__':
    fresh_probe()
