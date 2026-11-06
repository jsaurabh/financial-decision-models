import pytest

from finance_decisions.phase3_data import Groups, balanced, packed_length, preference_question


def test_heldout_near_duplicate_excludes_entire_transitive_source_group():
    groups = Groups()
    text = ' '.join('documentword'+str(i) for i in range(80))
    first = groups.add(text)
    second = groups.add(text.replace('documentword40','modifiedword40'))
    assert groups.find(first) == groups.find(second)
    groups.add(text.upper(), 'excluded')
    assert groups.split(first) is None
    assert groups.split(second) is None


def test_validation_groups_cannot_be_promoted_to_training():
    groups = Groups()
    key = groups.add('Source document containing complete evidence for the entire transaction.', 'train')
    groups.add('Source document containing complete evidence for the entire transaction.', 'development')
    assert groups.split(key) == 'development'
    groups.add('Source document containing complete evidence for the entire transaction.', 'calibration')
    assert groups.split(key) == 'calibration'


def test_balanced_sampling_keeps_small_classes_without_repeating_rows():
    rows = [{'id': str(i), 'domain':'a' if i<12 else 'b', 'target':'rare' if i==0 else 'common'} for i in range(24)]
    result = balanced(rows, 12)
    assert len(result)==len({r['id'] for r in result})==12
    assert len([r for r in result if r['domain']=='a'])==6
    assert any(r['target']=='rare' for r in result)


def test_packed_token_preflight_matches_candidate_encoder():
    from finance_decisions.candidate_model import CandidateDecisionModel
    class Tokenizer:
        def __call__(self,text,add_special_tokens=True):
            return {'input_ids':list(range(len(text.split())+int(add_special_tokens)))}
    class Encoder:
        tokenizer=Tokenizer();max_choices=11;max_chars=1000000
    row={'state':{'evidence':'Verified values'},'question':{'type':'choice','instructions':'Select the correct candidate.',
          'criteria':{str(i):'Candidate '+str(i) for i in range(11)}}}
    prefix,blocks=CandidateDecisionModel.encode(Encoder(),row)
    assert packed_length(Encoder.tokenizer,row)==len(prefix)+sum(map(len,blocks))


def test_preference_targets_follow_content_and_do_not_reveal_the_chosen_label():
    import json
    row={'chosen_response':{'content':'Correct response'},'rejected_response':{'content':'Incorrect response'}}
    labels=set()
    for index in range(20):
        question,target=preference_question(row,index)
        assert json.loads(question['criteria'][target])==row['chosen_response']
        labels.add(target)
    assert labels=={'A','B'}


def test_saved_candidate_capacity_restores_for_training_resume(tmp_path,monkeypatch):
    import json
    import torch
    from types import SimpleNamespace
    from safetensors.torch import save_file
    from finance_decisions.model import DecisionModel
    class Backbone(torch.nn.Module):
        def __init__(self):
            super().__init__();self.config=SimpleNamespace(hidden_size=8)
    tokenizer=SimpleNamespace(padding_side='right',pad_token_id=0)
    monkeypatch.setattr(DecisionModel,'pretrained_gemma',classmethod(lambda cls,dtype:SimpleNamespace(backbone=Backbone())))
    monkeypatch.setattr('peft.PeftModel.from_pretrained',lambda backbone,path:backbone)
    monkeypatch.setattr('finance_decisions.model.AutoTokenizer.from_pretrained',lambda path:tokenizer)
    config={'encoder':False,'merged':False,'architecture':'packed-independent-candidate-readout',
            'max_length':8192,'max_choices':11,'max_chars':1000000,'temperature':1.}
    (tmp_path/'decision_config.json').write_text(json.dumps(config))
    save_file({'weight':torch.zeros((1,8))},str(tmp_path/'readout.safetensors'))
    model=DecisionModel.load(tmp_path)
    assert (model.max_choices,model.max_chars,model.max_length)==(11,1000000,8192)
