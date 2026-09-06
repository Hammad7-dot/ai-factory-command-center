import json
from io import BytesIO
import numpy as np
import pandas as pd
from pypdf import PdfReader
from factory.data import ROOT, FEATURES, prepare, temporal_split
from factory.models import maintenance_predict, vision_predict
from factory.workflow import run_agents
from factory.reporting import incident_pdf
from PIL import Image

def test_split_purges_future_labels():
    df,_=prepare(pd.read_csv(ROOT/'data/sensors.csv'),pd.read_csv(ROOT/'data/production.csv'))
    tr,va,te=temporal_split(df)
    assert tr.timestamp.max()+pd.Timedelta(hours=6)<va.timestamp.min()
    assert va.timestamp.max()+pd.Timedelta(hours=6)<te.timestamp.min()
    assert not df.duplicated(['machine_id','timestamp']).any()
    assert not (df.pressure<0).any()

def test_pipeline_report_and_probability_response():
    df,_=prepare(pd.read_csv(ROOT/'data/sensors.csv'),pd.read_csv(ROOT/'data/production.csv'))
    row=df.iloc[610]
    m=maintenance_predict(row)
    assert 0<=m['probability']<=1 and len(m['features'])==len(FEATURES)
    other=maintenance_predict(df.iloc[5])
    assert abs(other['probability']-m['probability'])>.01
    vision,cam=vision_predict(Image.open(ROOT/'data/images/test/defect/0001.png'))
    assert cam.size==(64,64) and 0<=vision['confidence']<=1
    incident=run_agents(m,vision,'Rising vibration and bearing noise')
    assert len(incident['messages'])==4 and incident['status']=='pending'
    assert len(incident['scenarios'])==3 and incident['evidence']
    pdf=incident_pdf(incident)
    text=' '.join(p.extract_text() for p in PdfReader(BytesIO(pdf)).pages)
    assert 'PENDING' in text and incident['incident_id'] in text
    assert 'Retrieved evidence' in text and 'Digital twin' in text

def test_missing_and_invalid_csv():
    import pytest
    with pytest.raises(ValueError,match='Missing sensor columns'):
        prepare(pd.DataFrame({'timestamp':['bad']}))
