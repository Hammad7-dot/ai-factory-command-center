"""Controlled feedback learning. Supervisor choices are never failure labels."""
import hashlib
import json
import sqlite3
from contextlib import closing
from pathlib import Path
from datetime import datetime, timezone
from uuid import uuid4
import joblib
import numpy as np
import pandas as pd
from factory.data import ROOT, FEATURES, temporal_split


def _path(db_path):
    p=Path(db_path)
    return p if p.is_absolute() else ROOT/p


def save_outcome(incident_id, failed_within_6h, observed_at, reason, intervention='none', db_path='artifacts/decisions.sqlite'):
    """Store separately confirmed outcome only after the full horizon has elapsed."""
    if type(failed_within_6h) is not bool: raise ValueError('Outcome must be a confirmed yes/no value.')
    if not reason.strip(): raise ValueError('An observation source or reason is required.')
    if intervention not in {'none','reduced_load','maintenance','unknown'}: raise ValueError('Select a valid intervention.')
    observed=pd.to_datetime(observed_at,utc=True,errors='coerce')
    if pd.isna(observed) or observed>pd.Timestamp.now(tz='UTC'): raise ValueError('Outcome observation time must be valid and not in the future.')
    path=_path(db_path)
    if not path.exists(): raise ValueError('Record a supervisor decision first.')
    with closing(sqlite3.connect(path)) as db:
        try: entry=db.execute('SELECT incident_json FROM decisions WHERE incident_id=? ORDER BY timestamp DESC LIMIT 1',(incident_id,)).fetchone()
        except sqlite3.OperationalError: entry=None
        if not entry: raise ValueError('Record a supervisor decision for this incident first.')
        incident=json.loads(entry[0]); m=incident['maintenance']
        stamp=pd.to_datetime(m.get('observation_timestamp'),utc=True,errors='coerce')
        if pd.isna(stamp): raise ValueError('The incident has no valid sensor observation timestamp.')
        if observed<stamp+pd.Timedelta(hours=6): raise ValueError('Observe the complete six-hour horizon before recording an outcome.')
        features={f['feature']:f['value'] for f in m['features']}
        if set(FEATURES)-set(features): raise ValueError('The incident is missing training features.')
        db.execute('CREATE TABLE IF NOT EXISTS outcomes (incident_id TEXT PRIMARY KEY, machine_id TEXT, timestamp TEXT, observed_at TEXT, failure_next_6h INTEGER, intervention TEXT, reason TEXT, features_json TEXT, recorded_at TEXT)')
        try:
            db.execute('INSERT INTO outcomes VALUES (?,?,?,?,?,?,?,?,?)',(incident_id,m['machine_id'],stamp.isoformat(),observed.isoformat(),int(failed_within_6h),intervention,reason,json.dumps(features),datetime.now(timezone.utc).isoformat()))
        except sqlite3.IntegrityError: raise ValueError('This incident already has a confirmed outcome; duplicate labels are not accepted.') from None
        db.commit()
    return dict(incident_id=incident_id,outcome=int(failed_within_6h),observed_at=observed.isoformat(),eligible_without_intervention=intervention=='none')


def load_outcomes(db_path='artifacts/decisions.sqlite'):
    p=_path(db_path)
    if not p.exists(): return pd.DataFrame()
    with closing(sqlite3.connect(p)) as db:
        if not db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='outcomes'").fetchone(): return pd.DataFrame()
        rows=pd.read_sql_query('SELECT * FROM outcomes ORDER BY timestamp',db)
    if rows.empty: return rows
    expanded=pd.DataFrame([json.loads(s) for s in rows.pop('features_json')])
    rows=pd.concat([rows,expanded],axis=1)
    rows['timestamp']=pd.to_datetime(rows.timestamp,utc=True)
    return rows


def candidate_splits(feedback,original):
    """Use new outcomes only; purge six-hour windows around temporal boundaries."""
    if feedback.empty: raise ValueError('No confirmed outcomes are available.')
    cutoff=pd.to_datetime(original.timestamp,utc=True).max()+pd.Timedelta(hours=6)
    eligible=feedback[(feedback.intervention=='none')&(feedback.timestamp>cutoff)].copy()
    eligible=eligible.sort_values(['timestamp','incident_id']).drop_duplicates(['machine_id','timestamp'])
    if len(eligible)<60: raise ValueError(f'Need at least 60 unique post-training, no-intervention outcomes; found {len(eligible)}. Historical demo observations and intervention-affected outcomes are excluded.')
    parts=temporal_split(eligible)
    for name,part in zip(['train','validation','test'],parts):
        if len(part)<8 or part.failure_next_6h.nunique()<2: raise ValueError(f'{name} requires at least 8 outcomes and both outcome classes after the six-hour purge.')
    return parts


def train_candidate(db_path='artifacts/decisions.sqlite', provenance='supervisor-confirmed outcomes'):
    """Train one candidate, compare to the active model, register for review; never promote."""
    import sys
    sys.path.append(str(ROOT/'.deps'))
    import mlflow
    import mlflow.sklearn
    from sklearn.pipeline import make_pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score
    original=pd.read_csv(ROOT/'artifacts/cleaned_data.csv')
    original['timestamp']=pd.to_datetime(original.timestamp,utc=True)
    old_train,_,_=temporal_split(original)
    new_train,val,test=candidate_splits(load_outcomes(db_path),original)
    train=pd.concat([old_train,new_train],ignore_index=True)
    active_path=ROOT/'artifacts/maintenance.joblib'
    before=hashlib.sha256(active_path.read_bytes()).hexdigest()
    active=joblib.load(active_path)
    if active['kind']!='sklearn': raise ValueError('Candidate comparison currently requires the selected sklearn baseline.')
    candidate=make_pipeline(SimpleImputer(strategy='median'),RandomForestClassifier(n_estimators=160,max_depth=10,min_samples_leaf=3,random_state=73,n_jobs=2))
    candidate.fit(train[FEATURES],train.failure_next_6h)
    def evaluate(model,frame):
        p=model.predict_proba(frame[FEATURES])[:,1]; y=frame.failure_next_6h
        return dict(f1=float(f1_score(y,p>=.5,zero_division=0)),recall=float(recall_score(y,p>=.5,zero_division=0)),precision=float(precision_score(y,p>=.5,zero_division=0)),roc_auc=float(roc_auc_score(y,p)))
    cv=evaluate(candidate,val); av=evaluate(active['model'],val)
    ct=evaluate(candidate,test); at=evaluate(active['model'],test)
    # Non-regression alone must not let two useless models pass together.
    validation_gate=cv['f1']>=max(.65,av['f1']) and cv['recall']>=max(.70,av['recall'])
    folder=ROOT/'artifacts/candidates'/str(uuid4()); folder.mkdir(parents=True)
    joblib.dump(candidate,folder/'candidate.joblib')
    summary=dict(status='candidate_only',provenance=provenance,validation_gate_passed=validation_gate,promotion='Not promoted. Qualified review and additional prospective validation required.',rows=dict(historical_train=len(old_train),feedback_train=len(new_train),validation=len(val),test=len(test)),candidate_validation=cv,active_validation=av,candidate_test=ct,active_test=at,active_model_sha256=before,excluded='Interventions, historical timestamps, duplicate machine observations and boundary windows',split_ranges={name:[str(frame.timestamp.min()),str(frame.timestamp.max())] for name,frame in [('feedback_train',new_train),('validation',val),('test',test)]})
    mlflow.set_tracking_uri('sqlite:///'+(ROOT/'artifacts/mlflow.db').as_posix())
    mlflow.set_experiment('AI Factory - controlled feedback')
    with mlflow.start_run(run_name='feedback_candidate') as run:
        mlflow.log_params({'provenance':provenance,'model':'RandomForest160_depth10','promotion':'never automatic','purge_hours':6,**summary['rows']})
        mlflow.log_metrics({prefix+'_'+key:value for prefix,values in [('candidate_validation',cv),('active_validation',av),('candidate_test',ct),('active_test',at)] for key,value in values.items()})
        info=mlflow.sklearn.log_model(candidate,name='candidate',serialization_format='cloudpickle',registered_model_name='FactoryMaintenanceCandidate',pip_requirements=['scikit-learn','numpy','pandas','cloudpickle'])
        summary.update(run_id=run.info.run_id,model_uri=info.model_uri,minimum_validation_f1=.65,minimum_validation_recall=.70)
        (folder/'evaluation.json').write_text(json.dumps(summary,indent=2))
        mlflow.log_artifact(str(folder/'evaluation.json'))
    if hashlib.sha256(active_path.read_bytes()).hexdigest()!=before: raise RuntimeError('Active model unexpectedly changed.')
    (ROOT/'artifacts/latest_candidate.json').write_text(json.dumps(summary,indent=2))
    return summary
