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

## Repository Structure

```text
.
├── backend/
│   ├── app/
│   ├── alembic/
│   ├── requirements.txt
│   └── ...
├── frontend/
│   ├── package.json
│   └── src/
├── docs/
│   ├── prd.md
│   ├── architecture.md
│   ├── rules.md
│   └── phases.md
├── .env.example
├── docker-compose.yml
├── README.md
└── memory.md
```

## Environment Setup

### Prerequisites

- Python 3.10+
- PostgreSQL 15+
- Docker and Docker Compose
- Git
- Node.js and npm for the frontend

### Python dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

If needed, install additional platform-specific packages such as build tools and native ML dependencies as described in the project documentation.

### Frontend dependencies

```bash
cd frontend
npm install
```

### Configuration

Create a `.env` file based on the required project environment variables, including:

- `DATABASE_URL`
- `ANTHROPIC_API_KEY` or `OPENAI_API_KEY`
- storage configuration
- model runtime settings

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

- `docs/prd.md` — product requirements
- `docs/architecture.md` — architecture overview
- `docs/rules.md` — coding and build rules
- `docs/phases.md` — phased implementation roadmap

## License

This project does not currently declare a license. Add a license file if you plan to distribute or publish the repository.

## Status

This repository is in the initialization and implementation planning stage. Core dependencies and project scaffolding are being prepared before active feature development.
