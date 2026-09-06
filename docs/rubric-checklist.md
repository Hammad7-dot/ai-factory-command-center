# Submission evidence and limitations

This checklist maps the implemented project to the proposed hackathon scope. It is not a certification of rubric compliance or completed runtime testing. Compare it with the organizer's exact wording before submission.

| Area | Implementation / evidence | Qualification or gap |
|---|---|---|
| Five data categories | `factory/data.py`; sensors, production, image splits, maintenance notes and manuals | All generated; notes are entered as text in the UI rather than CSV batch ingestion |
| Cleaning and preparation | `prepare`; range validation, deduplication, forward fill, rolling features, production join; `preprocessing.json` | Rolling window is six observations; real irregular sampling needs a time-based policy |
| EDA | Numeric summary, trend charts, `eda_summary.csv` | Descriptive analysis only |
| Baseline ML | Logistic regression and Random Forest in `train.py` | Same synthetic maintenance task |
| Deep learning | Failure ANN and defect CNN with actual training loops | Real inference on synthetic-domain trained weights; no real-factory validation |
| Held-out evaluation | `metrics.csv`, confusion matrix and error CSVs | Actual numbers must come from training output, never invented |
| Leakage controls | Chronological split, six-hour purge, train-fit imputation/scaling | Same simulated fleet and generator across time partitions |
| NLP | `analyze_note`: symptom/urgency keywords and local negation | Rule-based baseline, not a trained language model; negation is limited |
| Vision explanation | CNN Grad-CAM in `factory/models.py` | Influential regions, not segmentation or causal proof |
| Sensor explanation | Local median-replacement sensitivity; global forest permutation importance | Not SHAP; correlated features limit interpretation |
| Knowledge retrieval | Paragraph TF-IDF cosine similarity and source-labelled positive matches | Lexical retrieval; no embedding service, reranker or OCR |
| RAG / GenAI | Optional OpenAI explanation over incident and retrieved evidence | Local Qwen generation verified; hosted route requires credits. Extractive fallback is not GenAI. |
| Four-agent collaboration | Maintenance, vision, knowledge and planner messages in `run_agents` | Deterministic modules; not four LLM-driven autonomous agents or agent-framework integration |
| Digital twin | Three expected-cost actions with explicit assumptions | Hypothetical constant-hazard horizon conversion and intervention multipliers; not physical simulation |
| Multimodal planning | Sensor risk drives costs, production rejection drives yield, image result triggers quality review; note/features/image guide retrieval | No learned joint multimodal fusion model |
| Human oversight | Approve/reject/modify form; modification validation; SQLite snapshot | No equipment control, authentication or role-based access |
| Feedback loop | Confirmed outcomes, gated candidate retraining, prospective evaluation and MLflow registration | No automatic promotion; intervention-confounded observations excluded; rehearsal is synthetic |
| Experiment tracking | MLflow runs, logged metrics/artifacts | Inspect actual completed runs before presenting |
| Model registration | Selected maintenance pyfunc registered as `FactoryMaintenance`; local metadata | UI loads local bundle; registry is not a live serving deployment |
| Reports | Downloadable PDF and structured JSON | Verify downloaded contents during rehearsal |
| Reproducibility | Seeded generators, training script and requirements | Tested direct dependencies recorded in requirements-tested.txt; hardware/library versions can alter results |
| Presentation | `docs/demo-script.md` with timed narration and Q&A | 12-slide HTML/PDF deck in outputs/; no recorded demo |

## Before submission

- Run training and inspect `artifacts/metrics.csv`, selected-model metadata and error examples.
- Run tests and retain their actual result; do not infer an app pass from unit tests.
- Rehearse normal and defective image cases and a low/high risk observation using actual outputs.
- Confirm a modified decision requires a reason, is saved, and appears in report downloads.
- If GenAI is required, demonstrate a successful local or hosted explanation with source checking. Record offline fallback honestly if credentials or network are unavailable.
- Show the experiment viewer, selected registry version and at least three completed experiment runs.
- Include limitations prominently and verify any final slide claims against generated artifacts.

The project is a locally runnable prototype. Real factory integration, deployment hardening, genuine SOP approval, calibration, authentication, online monitoring and automatic production promotion remain outside the implementation.

## Final checks

- Approve, reject and modify rehearsal PDFs passed content verification.
- Controlled-learning candidate registered in MLflow; minimum validation gate failed as intended on the shifted synthetic cohort. Active artifact SHA-256 stayed unchanged.
- Local Qwen weights downloaded and three real generations verified. The saved manual-specific comparison shows retrieval correcting an unsupported numeric hallucination. Hosted OpenAI remains credit-blocked.
