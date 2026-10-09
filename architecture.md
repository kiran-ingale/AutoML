# AutoML-KBP — Architecture

## 1. Tech Stack

| Layer | Choice | Notes |
|---|---|---|
| Language | Python 3.10+ | Backend + ML pipeline |
| Backend framework | FastAPI | Async, good for streaming run progress + tool-calling endpoints |
| Frontend | React.js (Vite) | Streamlit is fine for a quick internal demo, but React is assumed below for a real app; swap notes included |
| AutoML engine | AutoGluon (primary) or FLAML | Wraps XGBoost/LightGBM/CatBoost; pick one, don't hand-roll HPO |
| TFM libraries | TabPFN, TabICLv2 (or current successor) | Installed as optional extras; app must run without them (GBDT-only mode) |
| Explainability | SHAP (TreeSHAP) for GBDT; ShapPFN / kernel-based for TFM | Label TFM explanations as approximate in the UI |
| LLM / agent layer | Claude (or GPT) via an API SDK, tool-calling | One "orchestrator" agent with a small fixed toolset — see §4 |
| Data handling | Pandas, NumPy, scikit-learn | Preprocessing, splits, metrics |
| Database | PostgreSQL | Run metadata, audit reports, uploaded-dataset pointers |
| File/object storage | Local disk (dev) → S3/GCS (prod) | Raw datasets, generated artifacts (SHAP plots, reports) |
| Background jobs | FastAPI BackgroundTasks (MVP) → Celery/RQ (later) | Training can take minutes; don't block the request thread |
| Containerization | Docker + docker-compose | One compose file: api, worker (optional), postgres, frontend |
| Version control | Git & GitHub | Standard |

## 2. High-Level App Flow

```
[Frontend]
  Upload CSV + task description
        |
        v
[API] POST /runs  -> creates Run record, stores file, kicks off pipeline job
        |
        v
[Pipeline Worker]
  Stage 1: Profiling            -> profile.json
  Stage 2: Requirement parsing  -> LLM call -> requirements.json
  Stage 3: Router decision      -> route.json (path + rationale)
  Stage 4a/4b: Train (GBDT | TFM)
  Stage 5: Explainability       -> explanations/*
  Stage 6: Failure/fairness scan-> risk_report.json
  Stage 7: Audit report build   -> report.md / report.pdf
        |
        v
[API] GET /runs/{id}  -> status + all artifacts
[API] WS/SSE /runs/{id}/events -> live stage progress
[Chat] POST /runs/{id}/chat -> agent uses tools to answer / trigger re-run
```

The **LLM agent layer** runs alongside stages 2–7 as a tool-calling loop, not
as a separate service: the same tool functions the pipeline calls internally
are exposed to the agent, so "ask the system to re-run with a different
priority" and "the pipeline runs itself" go through the same code path.

## 3. Folder / File Structure

```
automl-x/
├── backend/
│   ├── app/
│   │   ├── main.py                 # FastAPI app entrypoint
│   │   ├── api/
│   │   │   ├── runs.py             # POST /runs, GET /runs/{id}
│   │   │   ├── chat.py             # POST /runs/{id}/chat
│   │   │   └── events.py           # SSE/WS run progress
│   │   ├── core/
│   │   │   ├── config.py           # env/settings
│   │   │   └── db.py               # SQLAlchemy session/engine
│   │   ├── models/                 # SQLAlchemy models (Run, Artifact, Message)
│   │   ├── schemas/                # Pydantic request/response models
│   │   ├── pipeline/
│   │   │   ├── profiling.py        # Stage 2: dataset profiling
│   │   │   ├── requirements.py     # Stage 3: NL -> structured requirements (LLM)
│   │   │   ├── router.py           # Stage 4: rule-based GBDT-vs-TFM decision
│   │   │   ├── preprocessing.py    # Stage 5: cleaning w/ before-after diff
│   │   │   ├── gbdt_path.py        # Stage 6a: AutoGluon/FLAML training
│   │   │   ├── tfm_path.py         # Stage 6b: TabPFN/TabICL training
│   │   │   ├── explain_gbdt.py     # TreeSHAP
│   │   │   ├── explain_tfm.py      # ShapPFN / kernel explainer
│   │   │   ├── risk_scan.py        # Stage 7: failure + fairness analysis
│   │   │   ├── report.py           # Stage 8: audit report builder
│   │   │   └── orchestrator.py     # Runs stages in order, emits progress events
│   │   ├── agent/
│   │   │   ├── tools.py            # Tool definitions exposed to the LLM agent
│   │   │   ├── prompts.py          # System prompts
│   │   │   └── agent.py            # Tool-calling loop wrapper
│   │   └── utils/
│   ├── tests/
│   ├── pyproject.toml / requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── src/
│   │   ├── pages/
│   │   │   ├── UploadPage.tsx
│   │   │   ├── RunDashboardPage.tsx    # profiling, router decision, progress
│   │   │   ├── ReportPage.tsx          # metrics, explanations, risk flags
│   │   │   └── ChatPanel.tsx
│   │   ├── components/
│   │   ├── api/                        # typed API client
│   │   └── App.tsx
│   ├── package.json
│   └── Dockerfile
├── docs/
│   ├── prd.md
│   ├── architecture.md
│   ├── rules.md
│   └── phases.md
├── docker-compose.yml
└── README.md
```

Streamlit alternative (if skipping the React frontend for a faster demo):
replace `frontend/` with a single `streamlit_app.py` that calls the same
backend API — keep the pipeline/agent code in `backend/` unchanged either
way, so the UI choice never touches pipeline logic.

## 4. Agent Tool Layer (contract)

Expose a small, fixed set of tools to the LLM agent — the agent should never
generate ad-hoc code to touch data or models:

- `get_dataset_profile(run_id)` → profiling summary
- `get_router_decision(run_id)` → path + rationale
- `get_run_report(run_id)` → metrics/explanations/risk summary
- `rerun_with_requirements(run_id, requirements)` → kicks off a new run with
  modified priorities/constraints
- `explain_term(term)` → static glossary lookup for ML jargon (no model call
  needed for this one — keep it deterministic)

Keep this list short and each tool single-purpose; grow it deliberately.

## 5. Data Model (core tables)

- **Run**: id, dataset_pointer, raw_requirements_text, parsed_requirements
  (JSON), profile (JSON), route (JSON), status, created_at.
- **Artifact**: id, run_id, type (model/plot/report), storage_path.
- **Message**: id, run_id, role, content, created_at (chat history per run).

## 6. Deployment (MVP)

- `docker-compose up`: postgres + backend + frontend, single-node.
- Env vars for LLM API key, DB URL, storage path/bucket.
- GPU is optional: if unavailable, TFM path is disabled at startup and the
  router always resolves to GBDT — this must be a visible flag in the UI,
  not a silent limitation.
