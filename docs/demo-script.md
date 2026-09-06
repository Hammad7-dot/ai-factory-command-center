# 11-minute demonstration script

Before presenting, complete training; inspect `artifacts/metrics.csv`; launch Streamlit and MLflow. Rehearse with the exact machine, observation and image you will show. Select an observation with elevated predicted risk based on the actual trained output rather than promising a specific probability. Have PDF/JSON downloads available as backup. Configure the optional API only if you intend to demonstrate hosted generation.

## 0:00–1:00 — Problem and boundary

“A bearing line produces sensor readings, quality records, images, maintenance notes and manuals. Our supervisor needs a single evidence-backed incident rather than five disconnected screens. This prototype connects those inputs and records the human decision. All factory data here is synthetic; it does not control a machine.”

Show the live overview, selected machine and historical observation. Clarify that the chart is historical demo playback, not a live sensor connection.

## 1:00–2:15 — Five inputs and cleaning

Show the sensor and optional production upload controls, product image, note field and document upload. Explain the exact timestamp join, invalid-range handling, past-only forward fill and rolling features. Show the cleaning audit in Models & data. Say that PDFs need extractable text.

“The note parser is a transparent keyword baseline with local negation handling. ‘No vibration’ is treated differently from ‘rising vibration.’ The image model has only seen generated bearing surfaces.”

## 2:15–3:45 — Analyze the incident

Choose the defective bearing sample, enter the demonstration note, and click Analyze incident. Show the six-hour failure probability and separate individual-image defect probability. Read the displayed values; do not substitute rehearsed numbers.

“The maintenance predictor was selected from logistic regression, Random Forest and ANN using validation F1. The CNN solves a separate visual classification task. An image score is not our factory-wide reject rate.”

## 3:45–5:00 — Explainability

Open Evidence & agents. Show the sensor sensitivity chart and Grad-CAM.

“For each sensor feature, we replace its value with the training median and measure the probability change. This measures local sensitivity, not causation. Grad-CAM shows influential regions for the predicted class, not a verified defect outline.”

Point to one actual prominent feature and visible highlighted region, without claiming that either proves the physical root cause.

## 5:00–6:15 — Retrieval and optional generation

Show retrieved passages, source names and similarity scores. Expand one relevant section. Show independent retrieval with an unusual unrelated word to illustrate no matching evidence.

If API access is configured, click Generate LLM explanation and compare a source citation with the displayed passage. “The hosted model explains our structured incident using retrieved evidence. It is not responsible for approving machinery actions.”

If both local and hosted generation are unavailable, say: “This is the labelled offline extractive fallback. Retrieval works locally; hosted generation has not been demonstrated in this run.” Do not call the fallback an LLM response.

## 6:15–7:15 — Four cooperating modules

Expand the structured messages. Show maintenance and vision outputs, knowledge evidence and planner scenarios.

“These are deterministic agent modules passing structured evidence. We chose a inspectable fixed workflow for the prototype. They are not four independently reasoning LLM agents.”

## 7:15–8:45 — Scenario comparison

Open Incident & decision. Compare continue, reduce load and maintenance. Show assumptions and point out horizon conversion.

“The prediction horizon is six hours; the twin converts it to eight under a hypothetical constant-hazard model. Costs, downtime and risk reductions are assumptions. Actual observed production rejection rate supplies quality loss. The planner picks the lowest illustrative expected cost, while a flagged image requests quality review.”

Explain one tradeoff using current outputs: planned downtime can cost production while reducing hypothetical failure risk. Do not promise that the same action always wins.

## 8:45–9:45 — Human decision and report

Choose Modify, select an action and enter a concrete reason such as “Supervisor requests inspection before releasing the affected lot.” Record the decision. Show the stored result and download PDF/JSON.

“The audit saves the decision and incident evidence. This feedback does not silently become a training label, and recording an approval does not send a command to equipment.”

## 9:45–11:00 — Evaluation and close

Open actual metrics, selection metadata, errors and MLflow. Compare maintenance models only with each other. Describe the six-hour purge and training-only preprocessing. Mention one observed error or limitation.

“The prototype demonstrates a connected, auditable workflow. Next validation work would require genuine factory data, camera-domain testing, probability calibration, verified SOPs and calibrated intervention costs. Our current evidence supports the synthetic demonstration.”

## Likely questions

| Question | Answer |
|---|---|
| Is this truly autonomous? | It autonomously calculates recommendations within a fixed workflow; physical action remains outside the system and requires a supervisor. |
| Is the data real? | No. Sensors, production, notes, images and included manuals are generated educational data. |
| Why such strong image metrics? | The train/test images share a simple synthetic generator. Strong synthetic scores do not establish real-camera performance. Quote actual metrics only. |
| How did you avoid leakage? | Chronological boundaries, six-hour purge for future labels, past-only rolling features and train-fit preprocessing. The same simulated machines still occur across time splits. |
| Does feedback retrain the model? | No. It creates an auditable record for later reviewed use. |
| What if no manual matches? | Retrieval returns no positive match; the explanation states the evidence gap. |
| Is this semantic vector RAG? | It uses TF-IDF lexical vectors and cosine retrieval; hosted generation is optional. It is not embedding-model retrieval. |
| Why not use image confidence as rejection percentage? | A classifier probability for one item does not estimate batch prevalence. Production records provide rejection rate. |
| What happens when the API fails? | An explicitly labelled extractive fallback appears; the UI does not claim successful generation. |
| Are the agents LLM agents? | No. Four deterministic modules exchange structured outputs; the optional LLM explains the assembled evidence. |
| Can it inspect our real factory images? | The UI accepts them, but the trained model is outside its validated domain. A real deployment requires representative labelled images and evaluation. |
| Is the digital twin calibrated? | No. It is a transparent expected-cost calculator with documented hypothetical parameters. |


## Controlled learning demonstration (replace one minute of Q&A if needed)

Open Controlled learning and distinguish an actual future outcome from an approval/rejection. Explain the full six-hour waiting window and exclusion of intervention-affected observations. Show the latest synthetic candidate evidence: validation recall and F1 failed the minimum floor, so the model was withheld. The active model hash is unchanged. Show FactoryMaintenanceCandidate in MLflow. Do not describe simulated supervisor events as human decisions or the failed candidate as an improvement.

If API credits are unavailable, explicitly say hosted GenAI is blocked. Local Qwen is installed and verified on this machine; select it to complete the GenAI demonstration without API credits.

## Verified local GenAI demo

Select Local Qwen in the sidebar; no API key is needed. Generate the incident explanation, then run the supplied manual-specific RAG question. The saved genuine comparison without retrieval invented 5% and 1.4; with retrieval it returned the correct 70% and 0.55. Show Section 4 of bearing_sop.txt. Explain that the model can still hallucinate or overstate urgency; the supervisor remains responsible for accepting or rejecting advice.
