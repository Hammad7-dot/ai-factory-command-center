"""CPU-sized predictive networks, inference and genuine gradient explanations."""
import json
import joblib
import numpy as np
import torch
from torch import nn
from PIL import Image
from factory.data import ROOT, FEATURES

torch.set_num_threads(2)

class FailureANN(nn.Module):
    def __init__(self):
        super().__init__(); self.net=nn.Sequential(nn.Linear(len(FEATURES),32),nn.ReLU(),nn.Linear(32,16),nn.ReLU(),nn.Linear(16,1))
    def forward(self,x): return self.net(x).squeeze(-1)

class DefectCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features=nn.Sequential(nn.Conv2d(3,12,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),nn.Conv2d(12,24,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),nn.Conv2d(24,32,3,padding=1),nn.ReLU())
        self.head=nn.Sequential(nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Linear(32,1))
    def forward(self,x): return self.head(self.features(x)).squeeze(-1)

def image_tensor(img):
    a=np.asarray(img.convert('RGB').resize((64,64)),dtype=np.float32)/255
    return torch.from_numpy(a.transpose(2,0,1).copy()).unsqueeze(0)

def maintenance_predict(row):
    bundle=joblib.load(ROOT/'artifacts/maintenance.joblib')
    x=row[FEATURES].to_frame().T.astype(float)
    if bundle['kind']=='ann':
        model=FailureANN(); model.load_state_dict(torch.load(ROOT/'artifacts/ann.pt',weights_only=True)); model.eval()
        def predict(data):
            with torch.no_grad(): return torch.sigmoid(model(torch.tensor(bundle['preprocessor'].transform(data),dtype=torch.float32))).numpy()
    else:
        def predict(data): return bundle['model'].predict_proba(data)[:,1]
    probability=float(predict(x)[0])
    # Local counterfactual sensitivity, not additive SHAP values or causality.
    evidence=[]
    for col,reference in zip(FEATURES,bundle['reference']):
        altered=x.copy(); altered[col]=reference
        delta=probability-float(predict(altered)[0])
        evidence.append(dict(feature=col,value=None if not np.isfinite(x[col].iloc[0]) else float(x[col].iloc[0]),reference=float(reference),probability_delta=delta))
    return dict(machine_id=str(row.machine_id),observation_timestamp=str(row.timestamp),probability=probability,model=bundle['name'],model_version=bundle.get('version','local'),horizon='next 6 hours',explanation_method='One-feature replacement with training median (local sensitivity; not causal)',features=sorted(evidence,key=lambda v:abs(v['probability_delta']),reverse=True))

def vision_predict(img):
    img=img.copy()
    img.thumbnail((1024,1024))
    model=DefectCNN(); model.load_state_dict(torch.load(ROOT/'artifacts/cnn.pt',weights_only=True)); model.eval()
    x=image_tensor(img); activations=model.features(x); activations.retain_grad()
    logit=model.head(activations).squeeze(); p=float(torch.sigmoid(logit).detach())
    # Grad-CAM for the predicted class, using the final convolutional activation.
    score=logit if p>=.5 else -logit
    score.backward()
    weights=activations.grad.mean(dim=(2,3),keepdim=True)
    cam=torch.relu((weights*activations).sum(dim=1))[0].detach().numpy()
    cam=cam/(cam.max()+1e-9)
    heat=np.asarray(Image.fromarray((cam*255).astype('uint8')).resize(img.size))/255
    base=np.asarray(img.convert('RGB'),dtype=float)
    color=np.stack([255*heat,70*heat,np.zeros_like(heat)],axis=-1)
    overlay=Image.fromarray(np.clip(base*.65+color*.35,0,255).astype('uint8'))
    return dict(defect_probability=p,label='defect' if p>=.5 else 'normal',confidence=max(p,1-p),severity='review required' if p>=.5 else 'no model-detected defect',model='synthetic-bearing-cnn',explanation_method='Grad-CAM for predicted class',domain='synthetic bearing surfaces only'),overlay
