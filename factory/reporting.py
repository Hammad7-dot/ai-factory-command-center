"""Portable incident report with evidence and explicit supervisor status."""
from io import BytesIO
from html import escape
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

def incident_pdf(incident, decision=None):
    stream=BytesIO(); styles=getSampleStyleSheet()
    doc=SimpleDocTemplate(stream,pagesize=(595,842),rightMargin=42,leftMargin=42,topMargin=42,bottomMargin=42)
    story=[]
    def para(text,style='BodyText'):
        story.append(Paragraph(escape(str(text)),styles[style])); story.append(Spacer(1,8))
    para('AI FACTORY / INCIDENT REPORT','Title')
    para('Synthetic bearing line | Educational decision support')
    para('Incident '+incident['incident_id'])
    para('Created '+incident['timestamp'])
    para('Supervisor status: '+(decision['status'].upper() if decision else 'PENDING — NOT AUTHORIZED'),'Heading2')
    m=incident['maintenance']; v=incident['vision']
    para('Sensor observation time: '+m.get('observation_timestamp','not recorded'))
    para(f"Machine {m['machine_id']} | Failure probability: {m['probability']:.1%} in {m['horizon']} | Model: {m['model']} v{m['model_version']}")
    para(f"Image classification: {v['label']} | Confidence: {v['confidence']:.1%} | Domain: synthetic bearing surfaces")
    para('Prediction evidence','Heading2')
    para(m['explanation_method'])
    for f in m['features'][:5]:
        value=f"{f['value']:.3f}" if f['value'] is not None else 'missing (imputed)'
        para(f"{f['feature']}: value {value}, reference {f['reference']:.3f}, probability change {f['probability_delta']:+.3f}")
    para('Maintenance note: '+incident['note_analysis']['text'])
    para('Retrieved evidence','Heading2')
    for e in incident['evidence']:
        para(e['source'],'Heading3'); para(e['text'])
    if not incident['evidence']: para('No matching evidence retrieved. Qualified manual review required.')
    para('Digital twin comparison','Heading2')
    rows=[['Action','Good units','Downtime h','Expected cost']]
    rows += [[s['action'],str(s['expected_units']),str(s['downtime_hours']),str(s['expected_cost'])] for s in incident['scenarios']]
    table=Table(rows,colWidths=[150,100,100,155]); table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#152B39')),('TEXTCOLOR',(0,0),(-1,0),colors.white),('GRID',(0,0),(-1,-1),.3,colors.lightgrey),('BOTTOMPADDING',(0,0),(-1,-1),9),('TOPPADDING',(0,0),(-1,-1),9)])); story.append(table)
    a=incident['scenarios'][0]['assumptions']
    para(f"Illustrative assumptions: {a['hours']:g}-hour horizon; {a['rate']:g} units/hour; failure repair cost 2,400; failure downtime 4 hours; lost-unit cost 4. Planned maintenance takes 2 hours and costs 450. Reduced load uses 70% throughput and 0.55 risk multiplier; maintenance uses 0.15 risk multiplier.")
    para(f"The 6-hour model probability converts to the scenario horizon using a hypothetical constant hazard. Reject rate: {a['observed_reject_rate']:.1%} ({a.get('reject_rate_source','assumed')}). Image confidence is not used as the batch reject rate. Monetary amounts are illustrative currency units.")
    para('Recommendation: '+incident['recommendation']['action'],'Heading2')
    para(incident['recommendation']['reason'])
    if 'explanation' in incident:
        para('Explanation mode: '+incident['explanation']['mode']); para(incident['explanation']['text'])
    if decision:
        para('Human decision','Heading2'); para('Decision: '+decision['decision']); para('Final action: '+str(decision['action'])); para('Reason: '+decision['reason']); para('Audit ID: '+decision['decision_id'])
    para('Limitations','Heading2')
    para('Models, images, sensor histories and manuals in this demo are synthetic. Probabilities are uncalibrated; the digital twin uses hypothetical costs and risk multipliers. Grad-CAM and feature sensitivity are explanations of model behavior, not proof of physical cause. No machinery commands are executed.')
    def footer(canvas,doc):
        canvas.setFont('Helvetica',8); canvas.drawString(42,24,'AI Factory Command Center | Synthetic demonstration'); canvas.drawRightString(553,24,str(doc.page))
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    return stream.getvalue()
