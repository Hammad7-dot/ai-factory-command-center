"""Run once: generate data, train actual models and record MLflow evidence."""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent/'.deps'))
import copy
import json
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.inspection import permutation_importance
from PIL import Image
from factory.data import ROOT, FEATURES, generate, prepare, temporal_split
from factory.models import FailureANN, DefectCNN, image_tensor

def metrics(y,p):
    return dict(precision=float(precision_score(y,p>=.5,zero_division=0)),recall=float(recall_score(y,p>=.5,zero_division=0)),f1=float(f1_score(y,p>=.5,zero_division=0)),roc_auc=float(roc_auc_score(y,p)))

def fit_network(model,x,y,vx,vy,epochs=50,lr=.003):
    opt=torch.optim.Adam(model.parameters(),lr=lr)
    criterion=torch.nn.BCEWithLogitsLoss()
    best=None; best_loss=float('inf')
    for epoch in range(epochs):
        model.train()
        order=torch.randperm(len(x))
        for ids in order.split(64):
            opt.zero_grad(); loss=criterion(model(x[ids]),y[ids]); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad(): val=float(criterion(model(vx),vy))
        if val<best_loss: best_loss=val; best=copy.deepcopy(model.state_dict())
    model.load_state_dict(best); model.eval()
    return model

def main():
    import mlflow
    import mlflow.sklearn
    torch.manual_seed(42); np.random.seed(42)
    out=ROOT/'artifacts'; out.mkdir(exist_ok=True)
    print('Generating synthetic datasets...',flush=True); generate()
    df,audit=prepare(pd.read_csv(ROOT/'data/sensors.csv'),pd.read_csv(ROOT/'data/production.csv'))
    tr,va,te=temporal_split(df)
    df.to_csv(out/'cleaned_data.csv',index=False)
    df[FEATURES+['failure_next_6h']].describe().to_csv(out/'eda_summary.csv')
    (out/'preprocessing.json').write_text(json.dumps(dict(audit=audit,split_rows=dict(train=len(tr),validation=len(va),test=len(te)),purge_hours=6),indent=2))
    mlflow.set_tracking_uri('sqlite:///'+(out/'mlflow.db').as_posix())
    mlflow.set_experiment('AI Factory - synthetic bearing line')
    candidates={}; results=[]; run_ids={}
    for name,estimator in [('logistic',LogisticRegression(max_iter=1000)),('random_forest',RandomForestClassifier(n_estimators=120,max_depth=8,min_samples_leaf=4,random_state=42,n_jobs=2))]:
        print('Training '+name,flush=True)
        model=make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),estimator)
        model.fit(tr[FEATURES],tr.failure_next_6h)
        val=metrics(va.failure_next_6h,model.predict_proba(va[FEATURES])[:,1])
        test=metrics(te.failure_next_6h,model.predict_proba(te[FEATURES])[:,1])
        with mlflow.start_run(run_name=name) as run:
            mlflow.log_params(dict(model=name,seed=42,split='chronological_60_20_20_purge6h',rows_train=len(tr)))
            mlflow.log_metrics({**{'validation_'+k:v for k,v in val.items()},**{'test_'+k:v for k,v in test.items()}})
            info=mlflow.sklearn.log_model(model,name='model',serialization_format='cloudpickle')
            run_ids[name]=dict(run_id=run.info.run_id,model_uri=info.model_uri)
        candidates[name]=model; results.append(dict(model=name,task='maintenance',**{'validation_'+k:v for k,v in val.items()},**{'test_'+k:v for k,v in test.items()}))
    print('Training failure ANN',flush=True)
    prep=make_pipeline(SimpleImputer(strategy='median'),StandardScaler()).fit(tr[FEATURES])
    def tensor(frame): return torch.tensor(prep.transform(frame[FEATURES]),dtype=torch.float32)
    def labels(frame): return torch.tensor(frame.failure_next_6h.to_numpy(),dtype=torch.float32)
    ann=fit_network(FailureANN(),tensor(tr),labels(tr),tensor(va),labels(va),epochs=45)
    torch.save(ann.state_dict(),out/'ann.pt')
    with torch.no_grad():
        av=torch.sigmoid(ann(tensor(va))).numpy(); at=torch.sigmoid(ann(tensor(te))).numpy()
    val=metrics(va.failure_next_6h,av); test=metrics(te.failure_next_6h,at)
    with mlflow.start_run(run_name='failure_ann') as run:
        mlflow.log_params(dict(model='ANN_32_16',epochs=45,seed=42,learning_rate=.003))
        mlflow.log_metrics({**{'validation_'+k:v for k,v in val.items()},**{'test_'+k:v for k,v in test.items()}})
        mlflow.log_artifact(str(out/'ann.pt')); run_ids['ann']=dict(run_id=run.info.run_id)
    results.append(dict(model='ann',task='maintenance',**{'validation_'+k:v for k,v in val.items()},**{'test_'+k:v for k,v in test.items()}))
    selected=max(results,key=lambda r:r['validation_f1'])['model']
    # Registry artifact includes the exact preprocessing and predictor selected by validation.
    bundle=dict(kind='ann' if selected=='ann' else 'sklearn',name=selected,preprocessor=prep,reference=tr[FEATURES].median().tolist(),version='1')
    if selected!='ann': bundle['model']=candidates[selected]
    class FactoryPredictor(mlflow.pyfunc.PythonModel):
        def load_context(self,context):
            self.bundle=joblib.load(context.artifacts['bundle'])
            if self.bundle['kind']=='ann':
                self.net=FailureANN(); self.net.load_state_dict(torch.load(context.artifacts['ann'],weights_only=True)); self.net.eval()
        def predict(self,context,model_input,params=None):
            if self.bundle['kind']=='ann':
                with torch.no_grad(): return torch.sigmoid(self.net(torch.tensor(self.bundle['preprocessor'].transform(model_input[FEATURES]),dtype=torch.float32))).numpy()
            return self.bundle['model'].predict_proba(model_input[FEATURES])[:,1]
    joblib.dump(bundle,out/'maintenance.joblib')
    with mlflow.start_run(run_name='selected_model_registry') as run:
        mlflow.log_param('selection','highest validation F1, threshold 0.5')
        info=mlflow.pyfunc.log_model(name='selected',python_model=FactoryPredictor(),artifacts={'bundle':str(out/'maintenance.joblib'),'ann':str(out/'ann.pt')},registered_model_name='FactoryMaintenance',pip_requirements=['scikit-learn','torch','numpy','pandas','joblib'])
    client=mlflow.MlflowClient(); versions=client.search_model_versions("name='FactoryMaintenance'")
    version=max(int(v.version) for v in versions); bundle['version']=str(version); joblib.dump(bundle,out/'maintenance.joblib')
    selected_meta=dict(name=selected,version=version,registry='FactoryMaintenance',selection='validation F1',run_id=run.info.run_id,model_uri=info.model_uri,training_domain='synthetic',threshold=.5)
    (out/'selected_model.json').write_text(json.dumps(selected_meta,indent=2))
    # Global held-out permutation importance for the forest; separate from local sensitivity.
    imp=permutation_importance(candidates['random_forest'],te[FEATURES],te.failure_next_6h,n_repeats=5,random_state=42,scoring='f1',n_jobs=1)
    pd.DataFrame(dict(feature=FEATURES,importance=imp.importances_mean,std=imp.importances_std)).sort_values('importance',ascending=False).to_csv(out/'feature_importance.csv',index=False)
    pred=at if selected=='ann' else candidates[selected].predict_proba(te[FEATURES])[:,1]
    errors=te[['machine_id','timestamp','failure_next_6h',*FEATURES]].copy(); errors['probability']=pred; errors['incorrect']=(pred>=.5)!=te.failure_next_6h.to_numpy()
    errors[errors.incorrect].to_csv(out/'maintenance_errors.csv',index=False)
    (out/'confusion_matrix.json').write_text(json.dumps(confusion_matrix(te.failure_next_6h,pred>=.5).tolist()))
    print('Training defect CNN',flush=True)
    manifest=pd.read_csv(ROOT/'data/image_manifest.csv')
    def images(split):
        part=manifest[manifest.split==split]
        x=torch.cat([image_tensor(Image.open(ROOT/p)) for p in part.path]); y=torch.tensor(part.label.to_numpy(),dtype=torch.float32)
        return x,y
    ix,iy=images('train'); vx,vy=images('validation'); tx,ty=images('test')
    cnn=fit_network(DefectCNN(),ix,iy,vx,vy,epochs=35,lr=.002)
    torch.save(cnn.state_dict(),out/'cnn.pt')
    with torch.no_grad(): cv=torch.sigmoid(cnn(vx)).numpy(); ct=torch.sigmoid(cnn(tx)).numpy()
    val=metrics(vy.numpy(),cv); test=metrics(ty.numpy(),ct)
    with mlflow.start_run(run_name='defect_cnn'):
        mlflow.log_params(dict(model='CNN_12_24_32',epochs=35,seed=42,learning_rate=.002,domain='synthetic images'))
        mlflow.log_metrics({**{'validation_'+k:v for k,v in val.items()},**{'test_'+k:v for k,v in test.items()}}); mlflow.log_artifact(str(out/'cnn.pt'))
    results.append(dict(model='cnn',task='vision',**{'validation_'+k:v for k,v in val.items()},**{'test_'+k:v for k,v in test.items()}))
    pd.DataFrame(results).to_csv(out/'metrics.csv',index=False)
    image_errors=manifest[manifest.split=='test'].copy(); image_errors['probability']=ct; image_errors['incorrect']=(ct>=.5)!=image_errors.label.to_numpy(); image_errors.to_csv(out/'image_predictions.csv',index=False)
    with mlflow.start_run(run_name='evaluation_evidence'):
        for name in ['metrics.csv','feature_importance.csv','maintenance_errors.csv','image_predictions.csv','preprocessing.json','selected_model.json','confusion_matrix.json']: mlflow.log_artifact(str(out/name))
    print(pd.DataFrame(results).to_string(index=False),flush=True)
    print('Selected model: '+json.dumps(selected_meta),flush=True)

if __name__=='__main__': main()
