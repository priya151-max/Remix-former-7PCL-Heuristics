# A Heuristic Informed Clinical Reasoning for Skin Tumor Differential Diagnosis using Multimodal Transformers and Knowledge Graphs

A deployable website that takes a **clinical photo** and a **dermoscopy image** of a skin lesion plus three simple
answers, estimates which of five lesion groups (BCC, melanocytic nevus, melanoma, seborrheic keratosis, other) it
most resembles, reasons over a clinical **knowledge graph**, refines the scores with a **clinical heuristic**, and
explains the result in **plain language** (retrieval-augmented) with a downloadable **PDF report**.

> Research prototype. Not a medical device, not a diagnosis. Scores are uncalibrated estimates from a model trained
> on a small research dataset (Derm7pt).

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt                            # CPU-only torch: see the note at the top of that file
cp .env.example .env                                       # optional: add OPENROUTER_API_KEY
# put the checkpoint at  model/RemixFormerPP_FULL_best_model.pth
python run.py                                              # open http://localhost:8000
```

A CPU analysis takes about 3 s (peak memory about 1.6 GB).

## The website

Left navigation (no Streamlit; plain FastAPI + vanilla JS, Plotly is bundled locally so no CDN is needed):

| Page | What it shows |
|---|---|
| About the system | What the site does, pipeline overview, the five groups |
| New analysis | Two image uploads, three questions (sex, body site, how the lesion feels), optional clinician fields |
| Diagnosis result | Top group and score, all five scores, the seven dermoscopic features, attention overlays, class description |
| AI explanation | Simple-words explanation with citations, the retrieved knowledge sources (RAG), glossary, references |
| Knowledge graph | Interactive graph **with edge weights**, focus on a node, case highlighting, adjacency and proximity matrices |
| Heuristic reasoning | The rule, before/after scores, component values, contradiction check, reliability and priors |
| Intermediate results | Every stage in order: inputs, metadata one-hot, LSM map and crops, dermoscopy attention, cross-modal attention, tokens and weights, node features and embeddings, head outputs, RAG, tensor shapes, timings |
| Model & method | Architecture, parameter counts, dataset statistics, limitations, references |

No file paths, CSV paths, device names or checkpoint information are shown or asked for anywhere.

## Metadata

The checkpoint was trained on **Derm7pt**, so the questions are the Derm7pt metadata fields, in plain words:
sex, body site, elevation (flat / slightly raised / raised bump), plus two optional clinician fields
(diagnostic difficulty, planned management). The **7-point checklist score is not asked**: the AI estimates the
seven checklist features from the images itself. Any field left blank is encoded as an all-zero one-hot block
(identical to scikit-learn's `handle_unknown="ignore"`).

> Known limitation: during training the checklist score, difficulty and management were always present. Leaving
> them blank is a distribution shift whose effect on accuracy has not been measured. Retraining the metadata
> branch on the basic fields only would remove it.

`model/metadata_vocab.json` replaces the old joblib encoder and the CSV path. Rebuild it with
`python scripts/build_metadata_vocab.py` (uses `data/derm7pt/`).

## Explanation: RAG + LLM

1. `src/explain.py::build_case` turns the model outputs into a numbers-only case summary.
2. `src/rag.py` retrieves passages from `src/knowledge/skin_knowledge.json` (tag rules for the predicted group,
   detected features, supplied metadata and safety notes, then BM25 for the remaining slots). No embedding model
   or external service is needed.
3. An LLM (OpenRouter, `OPENROUTER_MODEL`) writes an 8-section explanation in simple words, citing passages as
   `[S1]`, `[S2]`… It receives **no images**. The answer is validated (all sections, no certainty or "you have"
   statements); otherwise, or if there is no key or the request fails, a deterministic writer assembles the same
   sections from the same passages. The UI states which was used.

**The knowledge base is educational text written for this project. Have a dermatologist review and extend it
before public deployment.**

## Configuration (`.env`)

See `.env.example`: `OPENROUTER_API_KEY`, `OPENROUTER_MODEL`, `OPENROUTER_MAX_TOKENS`, `LLM_SHARE_METADATA`,
`HOST`, `PORT`, `DEVICE`, `MAX_UPLOAD_MB`, `JOB_TTL_MINUTES`, `MAX_QUEUE`.

## Deployment

* **One worker process** (`run.py` already does this): the model is held in memory and inference is serialised.
* Analyses run as background jobs polled by the browser, so reverse-proxy timeouts are not an issue.
* Images and results live in memory only and are purged after `JOB_TTL_MINUTES`. Nothing is written to disk.
* Put the app behind HTTPS (Caddy, nginx, a platform load balancer). The app sets security headers and a strict
  Content-Security-Policy (no inline scripts, no third-party requests).
* Docker: place the checkpoint in `model/`, then `docker build -t skin-dx . && docker run -p 8000:8000 --env-file .env skin-dx`.
* Add authentication, rate limiting and a privacy notice appropriate to your jurisdiction before exposing it publicly
  (health data rules such as GDPR/HIPAA may apply to uploaded images).
* **Never commit `.env`.** Rotate any API key that has been shared in an archive.

## Project layout

```
run.py                     start the server
webapp/main.py             FastAPI routes, job queue, security headers
webapp/static/             index.html, css/app.css, js/*.js, vendor/plotly.min.js
src/pipeline.py            AnalysisService: validate -> preprocess -> infer -> RAG -> explain -> result
src/model_architecture.py  RemixFormer++, graph encoder, hybrid heads (unchanged trained architecture)
src/checkpoint_loader.py   strict checkpoint loading and verification
src/prediction.py          inference + attention capture
src/heuristic_reasoning.py graph-integrated clinical heuristic
src/rag.py, src/explain.py, src/knowledge/   retrieval and explanation
src/report.py              PDF report (ReportLab + Matplotlib)
src/results.py             JSON result builder and attention overlays
src/metadata_*.py          form schema and pure-NumPy one-hot encoder
model/                     checkpoint, metadata_vocab.json, dataset_stats.json
data/derm7pt/              Derm7pt metadata used to build the vocabulary (dataset licence applies)
notebook/                  training notebooks
tests/                     pytest suite
```

## API

`GET /api/health`, `GET /api/meta-schema`, `GET /api/model-info`, `POST /api/analyze` (multipart:
`clinical`, `dermoscopy`, `sex`, `location`, `elevation`, `level_of_diagnostic_difficulty`, `management`) → `{job_id}`,
`GET /api/jobs/{id}`, `GET /api/report/{id}.pdf`, `GET /api/results/{id}.json`.

## Tests

```bash
pip install -r requirements-dev.txt
pytest -q          # model-dependent tests are skipped if the checkpoint is missing
```

## References

Argenziano et al., *Arch Dermatol* 1998 (7-point checklist); Kawahara et al., *IEEE JBHI* 2019 (Derm7pt);
Friedman, Rigel & Kopf, *CA Cancer J Clin* 1985 (ABCD rule); Kittler et al., *Lancet Oncol* 2002;
RemixFormer++ (IEEE Trans. Med. Imaging). Please verify citation details before publication.
