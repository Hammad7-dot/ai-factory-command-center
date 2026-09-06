import json
from datetime import timedelta
import pandas as pd
import pytest
from factory.data import FEATURES
from factory.workflow import save_decision
from factory.learning import save_outcome,load_outcomes,candidate_splits

def incident():
    return {'incident_id':'test','recommendation':{'action':'continue'},'maintenance':{'machine_id':'M','observation_timestamp':'2026-01-01T00:00:00Z','features':[{'feature':f,'value':1.0} for f in FEATURES]}}

def test_outcome_requires_decision_full_horizon_and_evidence(tmp_path):
    db=tmp_path/'audit.sqlite'
    with pytest.raises(ValueError,match='decision'): save_outcome('test',True,'2026-01-02','log',db_path=db)
    save_decision(incident(),'approve','checked',db_path=db)
    with pytest.raises(ValueError,match='six-hour'): save_outcome('test',True,'2026-01-01T02:00Z','log',db_path=db)
    with pytest.raises(ValueError,match='reason'): save_outcome('test',True,'2026-01-02','',db_path=db)
    save_outcome('test',True,'2026-01-02','verified log',db_path=db)
    with pytest.raises(ValueError,match='already'): save_outcome('test',False,'2026-01-02','other',db_path=db)
    rows=load_outcomes(db)
    assert len(rows)==1 and rows.failure_next_6h.iloc[0]==1

def test_candidate_split_excludes_history_interventions_and_purges():
    original=pd.DataFrame({'timestamp':pd.date_range('2026-01-01',periods=100,freq='h',tz='UTC')})
    feedback=pd.DataFrame({'timestamp':pd.date_range('2026-03-01',periods=100,freq='h',tz='UTC'),'incident_id':[str(i) for i in range(100)],'machine_id':'M','intervention':'none','failure_next_6h':[i%2 for i in range(100)]})
    feedback.loc[0,'timestamp']=original.timestamp.min()
    feedback.loc[1,'intervention']='maintenance'
    tr,va,te=candidate_splits(feedback,original)
    assert tr.timestamp.max()+pd.Timedelta(hours=6)<va.timestamp.min()
    assert va.timestamp.max()+pd.Timedelta(hours=6)<te.timestamp.min()
    assert tr.timestamp.min()>original.timestamp.max()
    assert set(pd.concat([tr,va,te]).intervention)=={'none'}
    with pytest.raises(ValueError,match='60'): candidate_splits(feedback.head(20),original)
