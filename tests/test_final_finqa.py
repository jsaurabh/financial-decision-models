import pytest
from finance_decisions.final_finqa import execute, verification_answers

def test_safe_dsl_reference_and_percent_display():
    qa={'program':'subtract(110, 100), divide(#0, 100)','exe_ans':.1,'answer':'10.0%'}
    assert execute(qa['program'],[])==.1
    assert verification_answers(qa,[])==('10.0%','11.0%')

def test_native_rounded_answer_and_currency():
    qa={'program':'divide(10000, 3)','exe_ans':3333.33333,'answer':'$3,333.33 million'}
    good,bad=verification_answers(qa,[])
    assert good=='$3,333.33 million' and bad=='$3,666.66 million'

def test_table_operations_and_boolean():
    assert execute('table_sum(revenue, none)',[['revenue','1,000','($200)']])==800
    assert verification_answers({'program':'greater(4, 3)','exe_ans':'yes','answer':'yes'},[])==('yes','no')

@pytest.mark.parametrize('program',['__import__(os, none)','exec(1, 2)','add(#0, 1)','exp(10, 10000)'])
def test_reject_invalid_or_unbounded_program(program):
    with pytest.raises((ValueError,IndexError)):execute(program,[])

def test_reject_ambiguous_or_mismatched_native_answers():
    for answer in ['10 to 20','30%','($10)']:
        with pytest.raises(ValueError):verification_answers({'program':'divide(1, 10)','exe_ans':.1,'answer':answer},[])
