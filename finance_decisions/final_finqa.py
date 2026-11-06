"""Conservative FinQA program validation and declared verification adaptation.

No Python eval, external functions or execution of source-provided code.
Unsupported programs/answer displays are adapter exclusions, never negatives.
"""
import math
import re
from decimal import Decimal, ROUND_HALF_UP

NUMBER=re.compile(r'[-+]?(?:\d[\d,]*)(?:\.\d+)?')

def number(value):
    s=str(value).strip().replace('$','').replace(',','').replace('%','')
    if s.startswith('(') and s.endswith(')'):s='-'+s[1:-1]
    return float(s)

def execute(program,table):
    results=[];pos=0
    # Native linear DSL has nested parentheses only inside table row labels.
    for item in re.split(r'\),\s*(?=[a-z_]+\()',program.strip()):
        item=item.strip().rstrip(',').strip()
        if not item.endswith(')'):item+=')'
        match=re.fullmatch(r'([a-z_]+)\((.*)\)',item)
        if not match:raise ValueError('invalid_program_syntax')
        op,args=match.groups()
        if ', ' not in args:raise ValueError('invalid_program_arguments')
        a,b=args.rsplit(', ',1)
        def arg(s):
            if re.fullmatch(r'#\d+',s):return results[int(s[1:])]
            if s.startswith('const_'):
                return number(s[6:].replace('m','-').replace('_','.'))
            return number(s)
        if op.startswith('table_'):
            matches=[row[1:] for row in table if str(row[0]).strip().lower()==a.strip().lower()]
            if len(matches)!=1 or b!='none':raise ValueError('unsupported_table_lookup')
            values=[number(v) for v in matches[0] if str(v).strip()]
            fn={'table_sum':sum,'table_average':lambda x:sum(x)/len(x),'table_max':max,'table_min':min}.get(op)
            if fn is None:raise ValueError('unsupported_operator')
            value=fn(values)
        else:
            x,y=arg(a),arg(b)
            if op=='add':value=x+y
            elif op=='subtract':value=x-y
            elif op=='multiply':value=x*y
            elif op=='divide':value=x/y
            elif op=='exp':
                if abs(y)>100 or abs(x)>1e20:raise ValueError('unsafe_exponent')
                value=x**y
            elif op=='greater':value='yes' if x>y else 'no'
            else:raise ValueError('unsupported_operator')
        if isinstance(value,(float,int)) and not math.isfinite(value):raise ValueError('nonfinite_program')
        results.append(value);pos+=1
    if not results:raise ValueError('empty_program')
    return results[-1]

def verification_answers(qa,table):
    result=execute(qa['program'],table);gold=qa['exe_ans'];answer=str(qa['answer']).strip()
    if isinstance(result,str):
        if result!=str(gold).lower() or answer.lower() not in ['yes','no'] or answer.lower()!=result:
            raise ValueError('boolean_reference_mismatch')
        return answer,('no' if answer.lower()=='yes' else 'yes')
    if not math.isclose(result,float(gold),rel_tol=1e-5,abs_tol=1e-5):raise ValueError('program_reference_mismatch')
    spans=list(NUMBER.finditer(answer))
    if len(spans)!=1:raise ValueError('ambiguous_answer_display')
    match=spans[0];literal=match.group();display=number(literal)
    if '(' in answer and ')' in answer:raise ValueError('parenthesized_answer_display')
    decimals=len(literal.rsplit('.',1)[1]) if '.' in literal else 0
    expected=result*100 if '%' in answer else result
    # Native human answer may round; permit its explicit display precision only.
    tolerance=.51*10**(-decimals)
    if abs(display-expected)>tolerance:raise ValueError('answer_execution_display_mismatch')
    delta=max(abs(display)*.1,10*10**(-decimals))
    counter=Decimal(str(display+delta)).quantize(Decimal(1).scaleb(-decimals),rounding=ROUND_HALF_UP)
    replacement=f'{counter:,.{decimals}f}' if ',' in literal else f'{counter:.{decimals}f}'
    wrong=answer[:match.start()]+replacement+answer[match.end():]
    if wrong==answer or abs(float(counter)-expected)<=tolerance:raise ValueError('invalid_counterfactual')
    return answer,wrong
