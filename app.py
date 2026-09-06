import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent/'.deps'))
import json
import sqlite3
import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
from PIL import Image
from pypdf import PdfReader
from factory.data import ROOT, FEATURES, prepare
from factory.models import maintenance_predict, vision_predict
from factory.workflow import run_agents, save_decision, retrieve
import factory.llm as llm_client
from importlib import reload
# Refresh this stateless client on rerun without clearing the user's private key/session.
generate_explanation=reload(llm_client).generate_explanation
from factory.reporting import incident_pdf
from factory.learning import save_outcome, load_outcomes, train_candidate

st.set_page_config(page_title='AI Factory | Command Center',page_icon='◈',layout='wide')
st.markdown('''<style>
.stApp {background:#0c141c} h1,h2,h3 {letter-spacing:-.035em} [data-testid="stMetric"] {background:#14222e;padding:18px;border:1px solid #263845;border-radius:8px} .block-container {padding-top:4rem} [data-testid="stSidebar"] {background:#101d28} div.stButton>button[kind="primary"] {background:#c8f16b;color:#101b16;border:0} .eyebrow {color:#b9ea69;letter-spacing:3px;font-size:12px;font-weight:700} </style>''',unsafe_allow_html=True)
st.markdown('<div class="eyebrow">OPERATIONS INTELLIGENCE / FACTORY 2.0</div>',unsafe_allow_html=True)
st.title('Factory Command Center')
st.caption('Bearing production line · Synthetic demonstration · Human-supervised decisions')
if not (ROOT/'artifacts/cnn.pt').exists() or not (ROOT/'artifacts/metrics.csv').exists():
    st.info('Models need training. Run `python train.py` in the project folder, then refresh.'); st.stop()

with st.sidebar:
    st.header('Incident inputs')
    mode=st.radio('Data source',['Built-in demo','Upload CSVs'])
    sensor_upload=prod_upload=None
    if mode=='Upload CSVs':
        sensor_upload=st.file_uploader('Sensor history CSV',type=['csv'])
        prod_upload=st.file_uploader('Production CSV (optional)',type=['csv'])
        st.caption('Sensors: machine_id, timestamp, temperature, vibration, pressure, rpm, load. Production: machine_id, timestamp, units, rejects.')
    image_upload=st.file_uploader('Product image',type=['png','jpg','jpeg'])
    sample=st.selectbox('Demo image',['Defective bearing','Normal bearing'])
    note=st.text_area('Maintenance note','Rising vibration and bearing noise. Inspect lubrication and surface scratches.')
    docs=st.file_uploader('Additional manual / SOP',type=['pdf','txt'],accept_multiple_files=True)
    st.divider(); st.caption('OpenAI receives evidence only when you request generation or comparison with OpenAI selected. Local Qwen keeps inference on this computer.')
    provider=st.selectbox('Explanation provider',['OpenAI API','Local Qwen (no API credits)'],index=1 if (ROOT/'local_models/qwen-0.5b/model.safetensors').exists() else 0)
    api_key=st.text_input('OpenAI API key',type='password')
    llm_model=st.text_input('LLM model','gpt-4o-mini')

def explain_with_selected_provider(snapshot):
    if provider.startswith('Local'):
        from factory.local_llm import generate_local_explanation
        return generate_local_explanation(snapshot)
    return generate_explanation(snapshot,api_key,llm_model)

try:
    if mode=='Upload CSVs' and sensor_upload is None: st.info('Upload a sensor CSV to continue.'); st.stop()
    sensors=pd.read_csv(sensor_upload if sensor_upload else ROOT/'data/sensors.csv')
    prod=pd.read_csv(prod_upload) if prod_upload else (pd.read_csv(ROOT/'data/production.csv') if mode=='Built-in demo' else None)
    data,audit=prepare(sensors,prod)
except Exception as exc:
    st.error('Input could not be read: '+str(exc)); st.stop()

c1,c2=st.columns([1,3])
with c1: machine=st.selectbox('Machine',sorted(data.machine_id.unique()))
history=data[data.machine_id==machine].sort_values('timestamp')
with c2:
    position=st.slider('Observation in machine history',0,len(history)-1,min(610,len(history)-1)) if len(history)>1 else 0
row=history.iloc[position]
st.caption(f'Observation: {row.timestamp} · Rolling features use current and earlier readings only')
extra=[]
for file in docs or []:
    try:
        if file.name.lower().endswith('.pdf'):
            reader=PdfReader(file)
            extra.extend(dict(source=f'{file.name} / page {i+1}',text=p.extract_text() or '') for i,p in enumerate(reader.pages))
        else: extra.append(dict(source=file.name,text=file.getvalue().decode('utf-8')))
    except Exception as exc: st.warning(f'Unable to extract {file.name}: {exc}')
try:
    img=Image.open(image_upload).convert('RGB') if image_upload else Image.open(ROOT/('data/images/test/defect/0001.png' if sample=='Defective bearing' else 'data/images/test/normal/0000.png')).convert('RGB')
except Exception as exc: st.error('Invalid image: '+str(exc)); st.stop()

if st.button('Analyze incident',type='primary',use_container_width=True):
    with st.spinner('Running models, retrieving evidence and comparing scenarios...'):
        maintenance=maintenance_predict(row)
        maintenance['observed_reject_rate']=float(row.reject_rate) if pd.notna(row.reject_rate) else .05
        maintenance['reject_rate_source']='production record' if pd.notna(row.reject_rate) else 'assumed 5% (production data unavailable)'
        vision,cam=vision_predict(img)
        incident=run_agents(maintenance,vision,note,extra)
        incident['explanation']=generate_explanation(incident,api_key='') if not __import__('os').environ.get('OPENAI_API_KEY') else {'mode':'not_requested','available':False,'text':'Click Generate LLM explanation to send evidence to the configured provider.'}
        st.session_state.incident=incident; st.session_state.cam=cam; st.session_state.incident_image=img.copy(); st.session_state.decision=None
        st.session_state.pop('rag_comparison',None)

tabs=st.tabs(['Live overview','Incident & decision','Evidence & agents','Models & data','Controlled learning'])
with tabs[0]:
    cols=st.columns(4)
    for col,label,value in zip(cols,['Temperature','Vibration','Pressure','Load'],[f'{row.temperature:.1f} °C',f'{row.vibration:.2f} mm/s',f'{row.pressure:.2f} bar',f'{row.load:.0%}']): col.metric(label,value)
    shown=history.iloc[max(0,position-71):position+1]
    st.plotly_chart(px.line(shown,x='timestamp',y=['vibration','pressure'],title='Last 72 observations · vibration (mm/s) and pressure (bar)',template='plotly_dark'),use_container_width=True)
    a,b=st.columns([2,1])
    with a:
        st.plotly_chart(px.line(shown,x='timestamp',y='temperature',title='Temperature trend (°C)',template='plotly_dark'),use_container_width=True)
    with b: st.image(img,caption='Selected product image',width=230)
    st.info('Select an observation and click Analyze incident to create a prediction and supervisor review.')

incident=st.session_state.get('incident')
with tabs[1]:
    if incident:
        st.caption(f"Saved observation: {incident['maintenance']['machine_id']} · {incident['maintenance'].get('observation_timestamp','unknown')} · Incident {incident['incident_id'][:8]}. Input changes take effect on the next Analyze click.")
        a,b,c=st.columns(3); a.metric('Failure probability / 6h',f"{incident['maintenance']['probability']:.1%}"); b.metric('Image defect probability',f"{incident['vision']['defect_probability']:.1%}"); c.metric('Suggested action',incident['recommendation']['action'].replace('_',' ').title())
        st.caption('Model probabilities are not calibrated confidence intervals. Synthetic training domain.')
        scenarios=pd.DataFrame(incident['scenarios'])
        st.plotly_chart(px.bar(scenarios,x='action',y='expected_cost',color='action',title='Eight-hour digital twin · illustrative expected cost',template='plotly_dark'),use_container_width=True)
        st.dataframe(scenarios[['action','expected_units','downtime_hours','failure_probability','expected_cost']],hide_index=True,use_container_width=True)
        with st.expander('Simulation assumptions'): st.json(incident['scenarios'])
        st.write(incident['recommendation']['reason'])
        st.subheader('Supervisor decision')
        if not st.session_state.get('decision'):
            with st.form('supervisor'):
                choice=st.radio('Decision',['approve','reject','modify'],horizontal=True)
                action=st.selectbox('Modified action (used only for modify)',['continue','reduce_load','maintenance'])
                reason=st.text_input('Reason / feedback (required for modify)')
                submitted=st.form_submit_button('Record human decision')
            if submitted:
                try: st.session_state.decision=save_decision(incident,choice,reason,action); st.rerun()
                except ValueError as exc: st.error(str(exc))
        else: st.success('Recorded: '+st.session_state.decision['status']); st.json(st.session_state.decision)
        st.download_button('Download incident PDF',incident_pdf(incident,st.session_state.get('decision')),file_name='factory-incident.pdf',mime='application/pdf')
        st.download_button('Download evidence JSON',json.dumps(dict(incident=incident,decision=st.session_state.get('decision')),indent=2),file_name='factory-evidence.json')
    else: st.info('Run an incident analysis first.')
with tabs[2]:
    if incident:
        a,b=st.columns(2)
        with a:
            st.subheader('Sensor explanation')
            f=pd.DataFrame(incident['maintenance']['features']); st.plotly_chart(px.bar(f,x='probability_delta',y='feature',orientation='h',title='Change from replacing one feature with its training median',template='plotly_dark'),use_container_width=True)
            st.caption('Local sensitivity; correlated features and noncausal associations limit interpretation. Global permutation importance is in Models & data.')
        with b: st.image(st.session_state.cam,caption='Grad-CAM · influential regions for predicted image class',width=300)
        st.subheader('Retrieved manual evidence')
        for e in incident['evidence']:
            with st.expander(e['source']+f" · similarity {e['score']:.3f}",expanded=True): st.write(e['text'])
        st.subheader('Grounded explanation')
        if st.button('Generate LLM explanation'):
            with st.spinner('Generating source-grounded explanation...'): incident['explanation']=explain_with_selected_provider(incident)
            st.rerun()
        st.caption('Mode: '+incident['explanation']['mode']); st.write(incident['explanation']['text'])
        if incident['explanation'].get('limitations'): st.caption(incident['explanation']['limitations'])
        if incident['explanation'].get('retrieved_sources'): st.caption('Retrieved sources: '+', '.join(incident['explanation']['retrieved_sources']))
        if not incident['explanation']['available']: st.info(incident['explanation'].get('reason','Hosted generation has not been requested.'))
        comparison_question=st.text_input('Question for the RAG comparison','What reduced-load throughput percentage and risk multiplier does the supplied manual specify? If the manual is absent, say these values are unavailable. Do not guess.')
        if st.button('Compare LLM answer with and without retrieval'):
            import copy
            comparison=copy.deepcopy(incident)
            comparison['question']=comparison_question
            comparison['evidence']=retrieve(comparison_question,extra)
            comparison['messages']=[]
            comparison['scenarios']=[]
            comparison['recommendation']['sources']=[]
            unsupported=copy.deepcopy(comparison)
            unsupported['evidence']=[]
            unsupported['recommendation']['sources']=[]
            unsupported['recommendation']['evidence_available']=False
            unsupported['messages']=[]
            unsupported.pop('explanation',None)
            with st.spinner('Generating two answers for comparison...'):
                first=explain_with_selected_provider(unsupported)
                second=explain_with_selected_provider(comparison) if first['available'] else dict(first,text='Comparison not run because the first generation failed. Resolve the provider error before retrying.')
                st.session_state.rag_comparison={'without_retrieval':first,'with_retrieval':second}
        if st.session_state.get('rag_comparison'):
            left,right=st.columns(2)
            for col,key in [(left,'without_retrieval'),(right,'with_retrieval')]:
                result=st.session_state.rag_comparison[key]
                col.markdown('**'+key.replace('_',' ').title()+'**'); col.caption(result['mode']); col.write(result['text'])
                if not result['available']:
                    col.error(result.get('reason','This result was created before detailed diagnostics were available. Click Compare again to see the current failure reason.'))
                    if result.get('error_code'): col.caption('Diagnostic: '+result['error_code'])
                if result.get('retrieved_sources'): col.caption('Retrieved sources: '+', '.join(result['retrieved_sources']))
            st.caption('Compare source citations and procedure specificity. Only hosted_llm or local_llm indicates actual generation. Fallback/unavailable modes do not constitute a GenAI evaluation.')
            st.warning('The answer without retrieval may invent manual-specific values. Check the retrieved source text before using either answer.')
            st.download_button('Download RAG comparison',json.dumps(st.session_state.rag_comparison,indent=2),file_name='rag-comparison.json')
        st.subheader('Structured agent messages'); st.json(incident['messages'])
    else: st.info('Run an incident analysis first.')
    st.subheader('Test retrieval independently')
    query=st.text_input('Manual question','What should a supervisor do about bearing vibration?')
    if st.button('Retrieve supporting sections'): st.json(retrieve(query,extra))
with tabs[3]:
    st.subheader('Held-out evaluation')
    st.dataframe(pd.read_csv(ROOT/'artifacts/metrics.csv'),hide_index=True,use_container_width=True)
    st.caption('Maintenance models use the same chronological splits; CNN metrics describe a separate image task. Selection uses validation F1 only. Fixed decision threshold 0.5.')
    st.json(json.loads((ROOT/'artifacts/selected_model.json').read_text()))
    st.subheader('Global permutation importance / Random Forest')
    st.dataframe(pd.read_csv(ROOT/'artifacts/feature_importance.csv'),hide_index=True)
    st.subheader('Cleaning audit'); st.json(audit)
    st.subheader('EDA / numeric summary'); st.dataframe(data[FEATURES].describe(),use_container_width=True)
    st.subheader('Maintenance error analysis'); st.dataframe(pd.read_csv(ROOT/'artifacts/maintenance_errors.csv').head(40),hide_index=True)
    st.caption('False negatives miss future synthetic failures; false positives trigger unnecessary review. Borderline wear trajectories and sensor noise are expected sources of overlap.')
    st.code('python tools/mlflow_ui.py',language='powershell')
    st.caption('Opens the local MLflow tracking UI on port 5000. Human feedback is saved for review; it does not become an automatic training label.')
with tabs[4]:
    st.subheader('Confirm an observed outcome')
    st.write('A supervisor decision is not a failure label. Record what actually happened over the full six-hour prediction horizon, and identify any intervention that changed the outcome.')
    if incident and st.session_state.get('decision'):
        with st.form('confirmed_outcome'):
            failed=st.selectbox('Did the machine fail within six hours?',['No','Yes'])
            observed_at=st.text_input('Outcome verified at (UTC)',pd.Timestamp.now(tz='UTC').isoformat())
            intervention=st.selectbox('Intervention during those six hours',['none','reduced_load','maintenance','unknown'])
            outcome_reason=st.text_input('Observation source / evidence')
            confirmed=st.checkbox('I verified the complete six-hour outcome using the stated evidence')
            record=st.form_submit_button('Save confirmed outcome')
        if record:
            if not confirmed: st.error('Confirm that you verified the outcome first.')
            else:
                try:
                    saved=save_outcome(incident['incident_id'],failed=='Yes',observed_at,outcome_reason,intervention)
                    st.success('Outcome saved separately from the supervisor decision.'); st.json(saved)
                except ValueError as exc: st.error(str(exc))
    else: st.info('Analyze an incident and record a supervisor decision before adding its outcome.')
    feedback=load_outcomes()
    st.metric('Confirmed outcomes in live audit',len(feedback))
    if not feedback.empty: st.dataframe(feedback[['machine_id','timestamp','failure_next_6h','intervention','reason']],hide_index=True)
    st.subheader('Train a candidate for review')
    st.caption('Requires 60 unique new outcomes after the original dataset, with both classes in each chronological split. Intervention-affected outcomes are excluded. Training never replaces the active model.')
    if st.button('Train and evaluate feedback candidate'):
        try:
            with st.spinner('Training, evaluating and logging a candidate in MLflow...'): result=train_candidate()
            st.success('Candidate registered for review. Active model unchanged.'); st.json(result)
        except ValueError as exc: st.warning(str(exc))
        except Exception as exc: st.error('Candidate training could not complete: '+type(exc).__name__)
    candidate_path=ROOT/'artifacts/latest_candidate.json'
    if candidate_path.exists():
        st.subheader('Latest candidate evidence')
        st.json(json.loads(candidate_path.read_text()))
        st.caption('Check provenance: synthetic rehearsal evidence does not represent real supervisor-confirmed factory outcomes.')
