# AI Factory Command Center

Open the app at http://127.0.0.1:8501. Restart with `python -m streamlit run app.py`. Open MLflow with `python tools/mlflow_ui.py`.

## Demonstration

1. Select M-01, observation 610 and Defective bearing, then Analyze incident.
2. Review predictions, explanations and retrieved manual sections.
3. Select Local Qwen (no API credits), then Generate LLM explanation. Local generation is installed and verified; no OpenAI key is needed.
4. Compare the three scenarios, record a supervisor decision and download its PDF.
5. Open Controlled learning to show outcome capture and the withheld candidate.

## Submission materials

- presentation.html and presentation.pdf: 12-slide deck.
- architecture.svg: system diagram.
- data-analysis.png and data-and-model-card.md: EDA and error evidence.
- rehearsal-approve.pdf, rehearsal-reject.pdf and rehearsal-modify.pdf: simulated supervisor examples.
- controlled-learning-evidence.json: candidate evaluation; active model unchanged.
- ai-factory-submission.zip: source, datasets, trained artifacts, MLflow records and deliverables.

Local language-model weights are excluded from the archive because of their size. Use requirements-local-llm.txt and tools/setup_local_llm.py when installing on another computer.

All included manufacturing data, manuals and rehearsal outcomes are synthetic. No physical machinery is controlled. See README.md and docs/demo-script.md for setup, limitations and narration.

Local GenAI evidence: local-genai-evidence.json contains actual generated outputs and a with/without-retrieval comparison. The grounded result matches the manual; the ungrounded answer hallucinated values.
