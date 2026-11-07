import json,math
from types import SimpleNamespace
import pytest
import torch
from torch import nn
from transformers import Gemma4TextConfig,Gemma4ForCausalLM
from finance_decisions.code_model import CodeDecisionModel,messages
from finance_decisions.phase4_numeric import arithmetic,answers
from finance_decisions.phase4_data import partition,split_and_deduplicate
from finance_decisions.phase4_train import set_loss,permuted
from finance_decisions.phase4_metrics import metrics

def test_native_lm_code_readout_matches_gemma_logits_including_softcap():
    config=Gemma4TextConfig(vocab_size=128,vocab_size_per_layer_input=128,hidden_size=32,
        intermediate_size=64,num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=1,
        head_dim=8,global_head_dim=8,hidden_size_per_layer_input=8,num_kv_shared_layers=0,
        layer_types=['sliding_attention','full_attention'],sliding_window=16,max_position_embeddings=512,
        use_double_wide_mlp=False,final_logit_softcapping=5.,rope_parameters={'sliding_attention':{'rope_type':'default','rope_theta':10000.},'full_attention':{'rope_type':'proportional','rope_theta':10000.,'partial_rotary_factor':.5}})
    torch.manual_seed(12);original=Gemma4ForCausalLM(config).eval();readout=nn.Linear(32,11,bias=False)
    with torch.no_grad():readout.weight.copy_(original.lm_head.weight[1:12])
    model=CodeDecisionModel(original.model,SimpleNamespace(pad_token_id=0,padding_side='left'),readout,{},'gemma',5.).eval()
    inputs={'input_ids':torch.tensor([[3,5,6],[0,4,5]]),'attention_mask':torch.tensor([[1,1,1],[0,1,1]])}
    with torch.inference_mode():
        expected=original(**inputs,use_cache=False).logits[:,-1,1:12]
        actual=model(inputs|{'_counts':torch.tensor([2,3])})
    assert torch.allclose(actual[0,:2],expected[0,:2],atol=1e-6)
    assert torch.allclose(actual[1,:3],expected[1,:3],atol=1e-6)
    assert (actual[0,2:]<-1e8).all()

def test_native_derivation_and_counterfactual_units():
    good,bad=answers({'answer':17.7,'scale':'percent','derivation':'16.6/93.8'})
    assert good=='17.7 percent' and bad!='17.7 percent'
    assert arithmetic('(1,000 - 40) / 2')==480
    with pytest.raises(ValueError):arithmetic('__import__("os").system("echo bad")')
    with pytest.raises(ValueError):answers({'answer':99,'scale':'','derivation':'1 + 2'})

def test_whole_document_and_cross_fold_boilerplate_policy():
    rows=[{'id':'a','task':'legal','family':'MAUD:doc1','split':'train','target':'yes','state':'first','question':{},'_core':'first entirely distinct provision with enough words to form a source family'},
          {'id':'b','task':'legal','family':'MAUD:doc1','split':'core_test','target':'no','state':'second','question':{},'_core':'second entirely separate section about a totally different legal obligation'},
          {'id':'c','task':'legal','family':'MAUD:doc2','split':'train','target':'no','state':'second','question':{},'_core':'second entirely separate section about a totally different legal obligation'},
          {'id':'d','task':'legal','family':'MAUD:doc2','split':'train','target':'yes','state':'unrelated','question':{},'_core':'unrelated text about audited statements from an independently verified annual financial report'}]
    kept,excluded,_=split_and_deduplicate(rows)
    assert {r['id'] for r in kept}=={'a','b','d'}
    assert all(r['split']=='core_test' for r in kept if r['family']=='MAUD:doc1')
    assert next(r for r in kept if r['id']=='d')['split']=='train'

def test_set_loss_accepts_all_native_routing_ties_and_permutation():
    row={'question':{'type':'choice','instructions':'choose','criteria':{'a':'A','b':'B','c':'C'}},'metadata':{'accepted':['a','b']},'target':'a'}
    values=torch.tensor([[2.,2.,0.]],requires_grad=True);loss=set_loss(values,row).mean();loss.backward()
    assert values.grad[0,0]<0 and values.grad[0,1]<0 and values.grad[0,2]>0
    changed=permuted(row,1);assert changed['target']=='a' and changed['metadata']['accepted']==['a','b']

def test_reference_receipts_and_future_outcomes_never_enter_prompt():
    row={'state':{'request':'calculate a ratio'},'question':{'type':'choice','instructions':'choose a model','criteria':{'a':'model A','b':'model B'}},
        'target':'GOLD_REFERENCE_SECRET','metadata':{'outcomes':'FUTURE_OUTCOMES_SECRET'},'source':{'gold':'ANNOTATION_SECRET'}}
    text=json.dumps(messages(row))
    assert all(x not in text for x in ['GOLD_REFERENCE_SECRET','FUTURE_OUTCOMES_SECRET','ANNOTATION_SECRET'])

def test_group_partition_is_shared_across_views_and_objectives():
    assert partition('RouterBench:query1')==partition('RouterBench:query1')
    assert {partition('family:'+str(i)) for i in range(100)}=={'train','validation','calibration','core_test'}

def test_full_native_label_spaces_are_not_silently_shortlisted():
    criteria={str(i):f'class {i}' for i in range(100)}
    row={'state':'A clause.','question':{'type':'choice','instructions':'Choose its topic.','criteria':criteria},'target':'99','metadata':{}}
    prompt=messages(row)[0]['content'];assert 'class 99' in prompt and 'class 0' in prompt
    values=torch.zeros(1,128,requires_grad=True);mask=torch.arange(128)>=100
    loss=set_loss(values.masked_fill(mask[None,:],-1e9),row).mean();loss.backward()
    assert loss.item()==pytest.approx(math.log(100)) and values.grad[0,99]<0
