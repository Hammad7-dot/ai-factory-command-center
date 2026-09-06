"""Run genuine local generations for incident explanation and a RAG comparison."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import json
import copy
import time
from factory.local_llm import generate_local_explanation
from factory.workflow import retrieve

incident=json.loads((ROOT/'outputs/sample-evidence.json').read_text())
start=time.time()
explanation=generate_local_explanation(incident)
print('Incident:',json.dumps(explanation),flush=True)
assert explanation['available'],explanation['reason'] if 'reason' in explanation else explanation
incident['question']='What reduced-load throughput percentage and risk multiplier does the supplied manual specify? If the manual is absent, say that these values are unavailable. Do not guess.'
incident['evidence']=retrieve('reduced load throughput risk multiplier')
without=copy.deepcopy(incident);without['evidence']=[]
ungrounded=generate_local_explanation(without)
grounded=generate_local_explanation(incident)
evidence=dict(model='Qwen/Qwen2.5-0.5B-Instruct',real_inference=True,question=incident['question'],incident_explanation=explanation,without_retrieval=ungrounded,with_retrieval=grounded,retrieved_sections=incident['evidence'],elapsed_seconds=round(time.time()-start,2),evaluation_note='One demonstration case, not a general LLM accuracy evaluation. Human review required.')
(ROOT/'outputs/local-genai-evidence.json').write_text(json.dumps(evidence,indent=2),encoding='utf-8')
print(json.dumps(evidence,indent=2),flush=True)
