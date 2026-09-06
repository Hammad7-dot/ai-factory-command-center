"""Synthetic rehearsal of controlled learning. Never writes the live human audit."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT)); sys.path.append(str(ROOT/'.deps'))
import json
import numpy as np
import pandas as pd
import joblib
from factory.data import FEATURES,prepare
from factory.workflow import save_decision
from factory.learning import save_outcome,train_candidate

def main():
    rng=np.random.default_rng(812)
    # Independent prospective synthetic cohort, not relabelled original test samples.
    wear=np.clip(.6+.35*np.sin(np.arange(186)*.29)+rng.normal(0,.1,186),0,1.3)
    rows=[]; production=[]
    for i in range(180):
        stamp=pd.Timestamp('2026-03-01',tz='UTC')+pd.Timedelta(hours=i)
        load=float(rng.uniform(.6,1))
        rows.append(dict(machine_id='SYNTHETIC-NEW',timestamp=stamp,temperature=43+wear[i]*32+load*7+rng.normal(0,3),vibration=.6+wear[i]*4+rng.normal(0,.45),pressure=6.5-wear[i]*.8+rng.normal(0,.25),rpm=1400+load*350+rng.normal(0,35),load=load,failure_next_6h=int(max(wear[i+1:i+7])+rng.normal(0,.09)>1.05)))
        production.append(dict(machine_id='SYNTHETIC-NEW',timestamp=stamp,units=120,rejects=int(rng.binomial(120,.012+wear[i]*.075))))
    df,_=prepare(pd.DataFrame(rows),pd.DataFrame(production))
    folder=ROOT/'artifacts/learning-demo';folder.mkdir(exist_ok=True)
    db=folder/'synthetic-audit.sqlite'
    # Each rehearsal gets its own database, preserving earlier evidence.
    if db.exists():
        from uuid import uuid4
        db=folder/('synthetic-audit-'+str(uuid4())+'.sqlite')
    model=joblib.load(ROOT/'artifacts/maintenance.joblib')['model']
    probabilities=model.predict_proba(df[FEATURES])[:,1]
    for i,(_,row) in enumerate(df.iterrows()):
        inc=dict(incident_id=f'synthetic-learning-{i}',status='pending',recommendation={'action':'continue'},maintenance={'machine_id':row.machine_id,'observation_timestamp':str(row.timestamp),'probability':float(probabilities[i]),'features':[{'feature':f,'value':float(row[f])} for f in FEATURES]},provenance='Synthetic rehearsal; simulated supervisor, not a human decision')
        save_decision(inc,'approve','SYNTHETIC REHEARSAL supervisor event; no real action',db_path=db)
        save_outcome(inc['incident_id'],bool(row.failure_next_6h),str(row.timestamp+pd.Timedelta(hours=6)),'SYNTHETIC generator future-wear outcome',db_path=db)
    df.to_csv(folder/'prospective-synthetic-cohort.csv',index=False)
    summary=train_candidate(db,provenance='Synthetic rehearsal: 180 independent generated observations and simulated supervisor events')
    (ROOT/'outputs/controlled-learning-evidence.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))

if __name__=='__main__': main()
