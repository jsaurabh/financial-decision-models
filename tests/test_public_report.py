from finance_decisions import public_report


def test_collector_never_overwrites_live_evaluator_output(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    live = tmp_path/'results/public-eval/jev'
    live.mkdir(parents=True)
    active = live/'predictions.jsonl'
    active.write_text('{"id":"new-local-result"}\n')
    remote = tmp_path/'remote'; remote.mkdir()
    for name, content in [('predictions.jsonl','{"id":"older-uploaded-result"}\n'),
                          ('config.json','{}'), ('summary.json','{"completed_records":1}')]:
        (remote/name).write_text(content)
    monkeypatch.setattr(public_report, 'hf_hub_download',
        lambda repo, path, **kwargs: str(remote/path.rsplit('/',1)[-1]))
    records, summaries = public_report.download(None, 'pinned-revision')
    assert records['jev'] == [{'id':'older-uploaded-result'}]
    assert summaries['jev']['completed_records'] == 1
    assert active.read_text() == '{"id":"new-local-result"}\n'
    assert (public_report.CACHE/'jev/predictions.jsonl').exists()


def test_duplicate_predictions_cannot_count_as_complete():
    rows=[{'id':'one'},{'id':'two'}]
    records={n:[{'id':'one'},{'id':'two'}] for n in public_report.NAMES}
    assert public_report.complete_records(rows,records)
    records['jev']=[{'id':'one'},{'id':'one'}]
    assert not public_report.complete_records(rows,records)
