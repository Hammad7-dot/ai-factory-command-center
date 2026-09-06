from streamlit.testing.v1 import AppTest
from factory.data import ROOT

def test_dashboard_analysis_and_reject(tmp_path, monkeypatch):
    import factory.workflow as workflow
    original=workflow.save_decision
    monkeypatch.setattr(workflow,'save_decision',lambda incident,choice,reason,action: original(incident,choice,reason,action,db_path=tmp_path/'audit.sqlite'))
    app=AppTest.from_file(str(ROOT/'app.py'),default_timeout=90).run()
    assert not app.exception
    next(b for b in app.button if b.label=='Analyze incident').click().run()
    assert not app.exception
    assert app.session_state['incident']['status']=='pending'
    next(r for r in app.radio if r.label=='Decision').set_value('reject')
    next(b for b in app.button if b.label=='Record human decision').click().run()
    assert not app.exception
    assert app.session_state['decision']['status']=='rejected'
