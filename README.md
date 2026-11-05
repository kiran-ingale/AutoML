# Adaptive AutoML: An Intelligent Router for TFM and GBDT Routing

Adaptive AutoML: An Intelligent Router for TFM and GBDT Routing is a platform for tabular machine learning that routes tasks between traditional gradient-boosted decision tree (GBDT) pipelines and tabular foundation models (TFMs), based on dataset characteristics and task requirements.

## Overview

This project is structured to provide:

- A FastAPI backend for API-driven ML workflows
- A frontend interface for dataset upload and experiment management
- A PostgreSQL-backed data layer for metadata and persistence
- AutoML model training with frameworks such as AutoGluon, XGBoost, LightGBM, CatBoost, and FLAML
- Explainability through SHAP for GBDT and MATLAB-based explainability for tabular foundation models
- Optional LLM-assisted workflows for analytics and recommendations

## Project Goals

- Simplify AutoML workflow setup for tabular and knowledge-base datasets
- Support experimentation and reproducibility
- Enable model benchmarking and comparison
- Expose APIs for integration with external systems and frontends
- Provide a clean and modular extension point for future ML features

## Tech Stack

### Backend
- Python 3.10+
- FastAPI
- Pydantic
- SQLAlchemy / SQLModel
- Alembic
- PostgreSQL
- Uvicorn

### Data and ML
- pandas
- NumPy
- scikit-learn
- SciPy
- AutoGluon
- FLAML
- XGBoost
- LightGBM
- CatBoost
- TabPFN / TabICLv2-related tooling (where applicable)
- SHAP (GBDT explainability)
- MATLAB-based explainability (TFM explainability)

### Frontend
- React
- Vite
- TypeScript

### Dev and Testing
- pytest
- pytest-asyncio
- httpx
- ruff
- black
- mypy

## Current Repository Structure

```text
.
├── backend/
│   ├── app/
│   │   ├── api/
│   │   ├── core/
│   │   ├── models/
│   │   ├── pipeline/
│   │   ├── schemas/
│   │   └── services/
│   ├── alembic/
│   └── tests/
├── modules/
├── .env.example
├── alembic.ini
├── requirements.txt
├── README.md
├── architecture.md
├── phases.md
├── prd.md
└── rules.md
```

The frontend is planned but has not been scaffolded yet.

## Environment Setup

### Prerequisites

- Python 3.10+
- PostgreSQL 15+
- Git
- Node.js and npm when starting frontend work
- Docker and Docker Compose for containerized development/deployment

### Python dependencies

Install the required Python packages:

```bash
.\myenv\Scripts\python.exe -m pip install -r requirements.txt
```

### Configuration

Create a local `.env` file based on `.env.example`. Configure:

- `DATABASE_URL`
- `STORAGE_DIR` (defaults to `storage`)
- `MAX_UPLOAD_BYTES` (defaults to 50 MiB)

Keep `.env` and its secrets out of version control.

### TFM readiness configuration

The router distinguishes installed packages from deployment readiness. For a
TabPFN route, provide `TABPFN_TOKEN` to the backend process through its secret
store and set `TABPFN_LICENSE_ACCEPTED=true` only after accepting the exact
checkpoint license in the account used by that token. The readiness check
records only whether the token is present; it never returns or logs its value.
For TabICL, review the pinned checkpoint terms and set `TABICL_READY=true`.
The backend downloads the classifier or regressor checkpoint from revision
`4dcd344ece2c00be9e831fdd35bed57b5ad83e19` on first use, passes that file to
TabICL, and verifies the recorded classifier SHA-256. The regressor is fetched
from the same pinned revision; an independent expected hash is not recorded,
so its checksum is not claimed as verified. The readiness flag attests to
terms review; it does not test network access in advance. Leave either flag
unset/false when the corresponding access is not ready.

### Run the backend

Apply database migrations, then start FastAPI from the project root:

```powershell
.\myenv\Scripts\python.exe -m alembic upgrade head
.\myenv\Scripts\python.exe -m uvicorn backend.app.main:app --reload
```

Open `http://127.0.0.1:8000/docs` to try the API. `POST /runs` accepts a `.csv` file and a task `description`, then returns the run ID and dataset profile. `/health` checks that the API is responding; `/ready` also checks PostgreSQL connectivity.

### Train an AutoML model

Start training for an uploaded run with `POST /runs/{run_id}/train`:

```json
{
  "target_column": "target",
  "task_type": "classification",
  "time_limit_seconds": 60,
  "test_size": 0.2,
  "random_state": 42,
  "priority": "speed",
  "hardware_constraint": "auto"
}
```

The API returns `202 Accepted` with the queued run and its persisted routing decision. `priority` accepts `accuracy`, `speed`, or `explainability`; `hardware_constraint` accepts `auto` or `cpu_only`; and an optional `path_override` can be `gbdt` or `tfm`. Hard dataset, hardware, and model-readiness gates are applied before route selection. Accuracy defaults to AutoGluon unless applicable validated evidence supports a TFM; speed requires scenario-matched measured end-to-end latency evidence, otherwise it also defaults to AutoGluon. An explicit TFM override selects among eligible TFMs by scale policy, without claiming predictive superiority. Package presence alone does not establish checkpoint/license readiness. A TFM can reduce model-search time, but total runtime depends on dataset size, hardware, and checkpoint availability. `time_limit_seconds` is an AutoGluon search budget; TabPFN and TabICL do not expose the same hard fit-time limit. Poll `GET /runs/{run_id}` for status, route rationale/exclusions, preprocessing summary, leaderboard, selected model, and failure details. Retrieve persisted model and leaderboard artifact references with `GET /runs/{run_id}/artifacts`. Training runs as an in-process FastAPI background task, so it is suitable for local development but is not a durable distributed job queue.

## Typical Workflow

1. Set up the environment and dependencies.
2. Configure backend and database settings.
3. Start the backend API service.
4. Install and start the frontend application.
5. Upload a dataset or connect a source.
6. Run automated profiling and feature analysis.
7. Train and compare AutoML models.
8. Review explainability and metrics.
9. Export or deploy the best-performing model.

## Development Notes

This project is intended to evolve through phased implementation as described in the project planning documents. Development should follow the conventions in the rule and architecture documents, with emphasis on clean separation between ML orchestration, backend APIs, database persistence, and frontend presentation.

## Documentation

The project includes core design and planning documents:

- `prd.md` — product requirements
- `architecture.md` — architecture overview
- `rules.md` — coding and build rules
- `phases.md` — phased implementation roadmap
- `IMPLEMENTATION_PLAN.md` — end-to-end implementation plan

## License

This project does not currently declare a license. Add a license file if you plan to distribute or publish the repository.

## Status

Backend foundation, PostgreSQL persistence, CSV upload/profiling, reproducible train/validation preprocessing, rule-based AutoGluon/TabPFN/TabICL routing, and path-specific training orchestration are implemented. Router and training behavior is covered by unit and mocked integration tests. A real TFM smoke run still requires accessible model weights and any required license acceptance. Explainability, the frontend, durable distributed job execution, end-to-end testing, and deployment remain future work.

## Kaggle GPU router benchmark

The reproducible Kaggle notebook and run instructions are in [`benchmark/README.md`](benchmark/README.md). It runs the router and model paths directly inside Kaggle, records GPU/software metadata and durable raw outputs, and keeps AutoGluon as a CPU-only baseline. The benchmark has **not yet been run**; there are no measured Kaggle results. Dataset licenses and exact TFM checkpoint terms must be verified before the first run. `time_limit_seconds` is not a hard fit timeout for TabPFN or TabICL.
