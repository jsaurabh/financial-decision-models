import json
from types import SimpleNamespace
from finance_decisions import final_report

def test_collector_pins_immutable_revision_and_uses_separate_cache(tmp_path,monkeypatch):
    remote=tmp_path/'remote';remote.mkdir();calls=[]
    for name,body in [('config.json',{}),('progress.json',{'completed':10}),('predictions.jsonl',{})]:
        (remote/name).write_text(json.dumps(body))
    monkeypatch.setattr(final_report,'CACHE',tmp_path/'collector-cache')
    monkeypatch.setattr(final_report,'HfApi',lambda:SimpleNamespace(dataset_info=lambda repo:SimpleNamespace(sha='frozen-commit')))
    def download(repo,file,**kw):
        calls.append(kw)
        return str(remote/file.rsplit('/',1)[-1])
    monkeypatch.setattr(final_report,'hf_hub_download',download)
    assert final_report.collect('gemma')['completed']==10
    assert len(calls)==3 and all(c['revision']=='frozen-commit' for c in calls)
    assert (tmp_path/'collector-cache/gemma/predictions.jsonl').exists()
    assert (remote/'predictions.jsonl').read_text()=='{}'
