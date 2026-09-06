# AI Factory Command Center — approved design

A local Streamlit application models a synthetic bearing production line. CSV sensor and production records, product images, maintenance notes and text/PDF manuals feed a single incident workflow. Random Forest and a PyTorch ANN predict failure within the next six sensor samples; a PyTorch CNN detects synthetic surface defects. Models are trained, evaluated and versioned, never replaced with canned predictions.

Maintenance, vision, knowledge and planning agents exchange JSON-compatible evidence. TF-IDF retrieval supplies document citations to a local Qwen model or optional API-backed LLM. Extractive fallback is labelled separately from real local generation. A digital twin compares continue, reduced load and maintenance using disclosed hypothetical costs and risk multipliers. Recommendations remain pending until a supervisor approves, rejects or modifies them. SQLite records decisions. PDF reports include evidence, assumptions and human status.

All included data and manuals are synthetic educational examples. Chronological sensor splits include a six-sample purge. Preprocessing fits training data only. Image splits use independently generated samples. Results are synthetic-domain benchmarks, not industrial validation. Credentials stay out of source and reports.

Deliverables: runnable application, generation/training scripts, datasets, model weights, MLflow experiments and selected version, XAI evidence, architecture, generated report and presentation material. Confirmed outcomes support controlled candidate retraining and prospective evaluation, with no automatic promotion and no live machinery control.
