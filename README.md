# AutoML-KBP

AutoML-KBP is a platform for automated machine learning workflows focused on knowledge-base and tabular data processing. It is designed to support dataset ingestion, automated model selection, training orchestration, explainability, and deployment-ready ML pipelines.

## Overview

This project is structured to provide:

- A FastAPI backend for API-driven ML workflows
- A frontend interface for dataset upload and experiment management
- A PostgreSQL-backed data layer for metadata and persistence
- AutoML model training with frameworks such as AutoGluon, XGBoost, LightGBM, CatBoost, and FLAML
- Explainability through SHAP and related tooling
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
- SHAP

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
  "random_state": 42
}
```

The API returns `202 Accepted` with the queued run. Poll `GET /runs/{run_id}` for its status, preprocessing summary, leaderboard, selected model, and any failure details. Retrieve persisted model and leaderboard artifact references with `GET /runs/{run_id}/artifacts`. The current AutoGluon runner uses CPU only. Training runs as an in-process FastAPI background task, so it is suitable for local development but is not a durable distributed job queue.

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

Backend foundation, PostgreSQL persistence, CSV upload/profiling, reproducible train/validation preprocessing, and CPU-only AutoGluon training orchestration are implemented. Explainability, the frontend, durable distributed job execution, end-to-end testing, and deployment remain future work.
