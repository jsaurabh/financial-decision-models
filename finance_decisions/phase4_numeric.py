"""Check published numerical derivations without executing source code."""
import ast,math,re

def arithmetic(text):
    text=str(text).replace(',','').replace('−','-').replace('×','*').strip()
    if len(text)>2000:raise ValueError('derivation_too_long')
    tree=ast.parse(text,mode='eval')
    def visit(node):
        if isinstance(node,ast.Expression):return visit(node.body)
        if isinstance(node,ast.Constant) and type(node.value) in [int,float]:return float(node.value)
        if isinstance(node,ast.UnaryOp) and isinstance(node.op,(ast.UAdd,ast.USub)):
            value=visit(node.operand);return -value if isinstance(node.op,ast.USub) else value
        if isinstance(node,ast.BinOp):
            a,b=visit(node.left),visit(node.right)
            if isinstance(node.op,ast.Add):return a+b
            if isinstance(node.op,ast.Sub):return a-b
            if isinstance(node.op,ast.Mult):return a*b
            if isinstance(node.op,ast.Div):return a/b
        raise ValueError('unsupported_derivation_syntax')
    value=visit(tree)
    if not math.isfinite(value):raise ValueError('nonfinite_derivation')
    return value

def answers(question,direction=1):
    literal=str(question['answer']).replace(',','').strip()
    if not re.fullmatch(r'[-+]?\d+(?:\.\d+)?',literal):raise ValueError('non_numeric_answer')
    value=float(literal);scale=question.get('scale','');derived=arithmetic(question['derivation'])
    places=min(8,len(literal.split('.')[1]) if '.' in literal else 0);tolerance=max(.51*10**-places,1e-5)
    if not any(abs(x-value)<=tolerance for x in ([derived,derived*100] if scale=='percent' else [derived])):
        raise ValueError('derivation_answer_mismatch')
    delta=max(abs(value)*.1,10*10**-places);wrong=round(value+direction*delta,places)
    if not math.isfinite(wrong) or abs(wrong-value)<=tolerance:raise ValueError('invalid_counterfactual')
    suffix=(' '+scale) if scale else ''
    return literal+suffix,f'{wrong:.{places}f}'+suffix
