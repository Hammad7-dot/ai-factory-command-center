"""Rehearse model-to-report flows using explicitly simulated supervisor events."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import json
import tempfile
from io import BytesIO
import pandas as pd
from PIL import Image
from pypdf import PdfReader
from factory.data import prepare
from factory.models import maintenance_predict,vision_predict
from factory.workflow import run_agents,save_decision
from factory.reporting import incident_pdf

df,_=prepare(pd.read_csv(ROOT/'data/sensors.csv'),pd.read_csv(ROOT/'data/production.csv'))
history=df[df.machine_id=='M-01']; results=[]
with tempfile.TemporaryDirectory(dir=ROOT/'work') as temp:
    for position,image_name,decision in [(5,'normal/0000.png','approve'),(610,'defect/0001.png','reject'),(610,'defect/0001.png','modify')]:
        row=history.iloc[position]; m=maintenance_predict(row)
        m['observed_reject_rate']=float(row.reject_rate);m['reject_rate_source']='production record'
        v,_=vision_predict(Image.open(ROOT/'data/images/test'/image_name))
        inc=run_agents(m,v,'Bearing inspection with vibration and quality review.')
        inc['provenance']='SYNTHETIC REHEARSAL: programmatically simulated supervisor event'
        outcome=save_decision(inc,decision,'SYNTHETIC REHEARSAL, not a real supervisor decision','reduce_load' if decision=='modify' else None,db_path=Path(temp)/'audit.sqlite')
        report=incident_pdf(inc,outcome)
        extracted=' '.join(p.extract_text() for p in PdfReader(BytesIO(report)).pages)
        assert outcome['status'].upper() in extracted and str(row.timestamp) in extracted
        assert inc['incident_id'] in extracted
        assert len(inc['messages'])==4 and len(inc['scenarios'])==3 and inc['evidence']
        (ROOT/'outputs'/f'rehearsal-{decision}.pdf').write_bytes(report)
        results.append(dict(observation=position,image=v['label'],failure_probability=m['probability'],human_status=outcome['status'],report_verified=True,source='Synthetic rehearsal only'))
(ROOT/'outputs/rehearsal-evidence.json').write_text(json.dumps(results,indent=2))
print(json.dumps(results,indent=2))
