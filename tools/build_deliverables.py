"""Generate an example pending report, synthetic manual PDF and browser slide deck."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import json
from html import escape
import pandas as pd
from PIL import Image
from factory.data import prepare, MANUALS
from factory.models import maintenance_predict, vision_predict
from factory.workflow import run_agents
from factory.reporting import incident_pdf
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

out=ROOT/'outputs'; out.mkdir(exist_ok=True)
data,_=prepare(pd.read_csv(ROOT/'data/sensors.csv'),pd.read_csv(ROOT/'data/production.csv'))
row=data[data.machine_id=='M-01'].iloc[610]
m=maintenance_predict(row); m['observed_reject_rate']=float(row.reject_rate); m['reject_rate_source']='production record'
v,cam=vision_predict(Image.open(ROOT/'data/images/test/defect/0001.png'))
incident=run_agents(m,v,'Rising vibration and bearing noise. Inspect surface scratches.')
genai_path=out/'local-genai-evidence.json'
if genai_path.exists():
    recorded=json.loads(genai_path.read_text(encoding='utf-8')).get('incident_explanation',{})
    if recorded.get('available'):
        incident['explanation']=dict(recorded,provenance='Previously generated for the same synthetic M-01 observation 610 and defective image; see local-genai-evidence.json')
(out/'sample-incident.pdf').write_bytes(incident_pdf(incident))
(out/'sample-evidence.json').write_text(json.dumps(incident,indent=2),encoding='utf-8')
cam.save(out/'grad-cam.png')
styles=getSampleStyleSheet(); story=[]
for name,content in MANUALS.items():
    story.append(Paragraph(name,styles['Heading1']))
    for paragraph in content.split('\n\n'): story.extend([Paragraph(escape(paragraph).replace('\n','<br/>'),styles['BodyText']),Spacer(1,14)])
SimpleDocTemplate(str(out/'synthetic-manual.pdf')).build(story)
metrics=pd.read_csv(ROOT/'artifacts/metrics.csv')
selected=json.loads((ROOT/'artifacts/selected_model.json').read_text())
rows=''.join('<tr>'+''.join(f'<td>{escape(str(x))}</td>' for x in [r.model,r.task,f'{r.test_precision:.3f}',f'{r.test_recall:.3f}',f'{r.test_f1:.3f}',f'{r.test_roc_auc:.3f}'])+'</tr>' for r in metrics.itertuples())
slides=[
('AI Factory<br>Command Center','<p class="lead">From factory signals to a human-reviewed decision.</p><p>AI Factory 2.0 · Bearing manufacturing demonstration</p><p class="note">All included factory data and manuals are synthetic.</p>'),
('The factory problem','<p class="lead">Rising vibration. More surface defects. A production decision.</p><ul><li>Detect emerging machine risk before the next six hours.</li><li>Inspect product quality and maintenance context.</li><li>Compare the cost of three responses before a supervisor decides.</li></ul>'),
('Five input modalities','<div class="flow"><span>Production CSV</span><span>Sensor history</span><span>Product image</span><span>Maintenance note</span><span>PDF / text SOP</span></div><p>6 machines · 4,320 sensor observations before injected duplicates · 920 independently seeded images.</p><p class="note">Upload controls accept replacement evidence. Image predictions are only validated on the synthetic generator.</p>'),
('One integrated architecture','<div class="flow"><span>Inputs + cleaning</span><b>→</b><span>Maintenance + vision models</span><b>→</b><span>Knowledge retrieval</span></div><div class="flow"><span>Structured agents</span><b>→</b><span>Digital twin</span><b>→</b><span>Supervisor</span><b>→</b><span>PDF + audit</span></div><p>MLflow records experiments, metrics, artifacts and the selected model version.</p>'),
('Prevent leakage before training','<ul><li>Remove duplicate machine timestamps; coerce types and invalid values.</li><li>Forward-fill within each machine; fit remaining imputation on training only.</li><li>Join production records; create rolling mean and variation features.</li><li>Chronological 60 / 20 / 20 split with six-hour purges.</li><li>Retain plausible high readings: they may be the fault signal.</li></ul>'),
('Measured model results',f'<table><tr><th>Model</th><th>Task</th><th>Precision</th><th>Recall</th><th>F1</th><th>AUC</th></tr>{rows}</table><p>Selected maintenance model: <strong>{selected["name"]}</strong>, registered version {selected["version"]}. Selection uses validation F1.</p><p class="note">Held-out synthetic test set. Maintenance and vision are different prediction tasks; their metrics are not a direct contest.</p>'),
('Show why the models responded','<ul><li>Global permutation importance measures the Random Forest’s reliance on features.</li><li>Local sensitivity replaces each feature with its training median.</li><li>Grad-CAM highlights regions contributing to the CNN’s predicted class.</li></ul><p class="note">These explain model behavior. They do not establish physical causes or calibrated confidence.</p>'),
('Retrieve before explaining','<div class="flow"><span>Symptoms + signals</span><b>→</b><span>TF-IDF section retrieval</span><b>→</b><span>Cited evidence</span><b>→</b><span>Optional LLM</span></div><p>Demonstrate a manual-specific question, then inspect the retrieved section. Select Local Qwen to generate an explanation without paid API credits.</p><p class="note">Offline summaries are explicitly extractive. Verified local case: without retrieval the model invented 5% and 1.4; with retrieval it returned the manual values 70% and 0.55. One case does not prove general accuracy.</p>'),
('Four agents exchange evidence','<ul><li><strong>Maintenance:</strong> model probability, signals and note extraction.</li><li><strong>Vision:</strong> defect score and quality-review flag.</li><li><strong>Knowledge:</strong> retrieved paragraphs and source names.</li><li><strong>Planner:</strong> scenario comparison and pending recommendation.</li></ul><p class="note">Deterministic modular orchestration with visible structured messages; no claim of autonomous reasoning agents.</p>'),
('Simulate the operational choices','<div class="flow"><span>Continue</span><span>Reduce load to 70%</span><span>Stop for 2h maintenance</span></div><ul><li>Compare good units, downtime, failure risk and expected cost.</li><li>Convert the six-hour prediction to an eight-hour horizon using an assumed constant hazard.</li><li>Use observed production rejects for yield; image probability is not a batch defect rate.</li></ul><p class="note">Hypothetical economics and action effects. This is a simplified digital twin.</p>'),
('Human authority is part of the product','<p class="lead">Approve. Reject. Modify.</p><ul><li>Save the decision and evidence snapshot in SQLite.</li><li>Require a reason when modifying the action.</li><li>Download the report with an explicit supervisor status.</li></ul><p>Confirmed outcomes can train a candidate. Chronological evaluation and minimum-quality gates protect the active model; promotion is never automatic.</p>'),
('Live demonstration + next steps','<ol><li>Choose M-01 and observation 610; analyze the defective sample.</li><li>Inspect predictions, explanations, sources and agent messages.</li><li>Compare scenarios; reject or modify the recommendation.</li><li>Download the PDF and show MLflow experiments.</li></ol><p class="note">Before real deployment: real datasets, calibration, external validation, verified SOPs and controlled candidate promotion.</p>')]
html='''<!doctype html><html><head><meta charset="utf-8"><title>AI Factory — Presentation</title><style>
*{box-sizing:border-box}body{margin:0;background:#0c141c;color:#edf3f6;font-family:Arial,sans-serif}section{display:none;min-height:100vh;padding:7vh 8vw 10vh}section.active{display:flex;flex-direction:column;justify-content:center}h1{font-size:clamp(36px,4.5vw,68px);letter-spacing:-2px;margin:0 0 32px;color:#c8f16b}p,li{font-size:clamp(20px,2vw,30px);line-height:1.5}li{margin:16px 0}.lead{font-size:3vw}.note{font-size:19px;color:#a8bbc7}.flow{display:flex;gap:18px;align-items:center;flex-wrap:wrap;margin:25px 0}.flow span{border-bottom:3px solid #c8f16b;padding:18px;font-size:24px}table{border-collapse:collapse;width:100%;font-size:23px}th,td{text-align:left;padding:16px;border-bottom:1px solid #456}th{color:#c8f16b}nav{position:fixed;bottom:20px;right:30px;display:flex;align-items:center;gap:20px}button{background:#c8f16b;border:0;border-radius:4px;padding:12px 20px;font-size:18px;cursor:pointer}@media print{section{display:flex!important;flex-direction:column;justify-content:center;page-break-after:always;height:100vh;padding:32px 52px}h1{font-size:34px;margin:0 0 20px}p,li{font-size:18px;line-height:1.35;margin:10px 0}.lead{font-size:26px}.note{font-size:12px}table{font-size:16px}th,td{padding:8px}.flow{margin:14px 0;gap:12px}.flow span{font-size:18px;padding:10px}nav{display:none}@page{size:landscape;margin:0}}
</style></head><body>'''
for i,(title,body) in enumerate(slides): html+=f'<section class="{"active" if i==0 else ""}"><h1>{title}</h1>{body}</section>'
html+='''<nav><button onclick="show(n-1)">←</button><span id="count"></span><button onclick="show(n+1)">→</button></nav><script>let n=0;const slides=[...document.querySelectorAll('section')];function show(i){n=Math.max(0,Math.min(slides.length-1,i));slides.forEach((s,j)=>s.classList.toggle('active',j===n));document.querySelector('#count').textContent=(n+1)+' / '+slides.length}document.addEventListener('keydown',e=>{if(['ArrowRight',' '].includes(e.key)){e.preventDefault();show(n+1)}if(e.key==='ArrowLeft')show(n-1)});show(0)</script></body></html>'''
(out/'presentation.html').write_text(html,encoding='utf-8')
print('Created report, evidence, Grad-CAM, manual and 12-slide browser presentation.')
