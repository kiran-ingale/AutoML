# AutoML-KBP Implementation Plan

This document defines the overall implementation roadmap from initial setup through debugging and deployment for the AutoML-KBP project.

## 1. Project Objective

Build a full-stack AutoML platform that allows users to:

- Upload tabular or knowledge-base style datasets
- Profile the dataset and detect schema issues
- Run automated model selection and training
- Evaluate model performance and explainability
- View results through a web interface
- Deploy a working ML application in a stable environment

## 2. Guiding Principles

- Keep backend, frontend, and ML pipeline responsibilities separated
- Prefer modular services and clear interfaces over tightly coupled logic
- Validate each phase with working tests or smoke checks before moving on
- Design for deployability from the beginning, not as an afterthought
- Treat ML model pipelines as versioned and reproducible workflows

## 3. Implementation Phases

### Phase 0: Environment Setup and Baseline Configuration

Goal: Ensure the workspace is ready for development.

Tasks:

1. Create and activate a Python virtual environment
2. Install base dependencies from `requirements.txt`
3. Verify the Python toolchain version and compatibility
4. Set up PostgreSQL locally or in a container
5. Set up Node.js and install frontend dependencies
6. Create `.env` configuration for database, secrets, and storage
7. Confirm project paths, package structure, and working directory conventions

Deliverables:

- Working local dev environment
- Database accessible from backend
- Frontend dependencies installed
- Base environment variables configured

Success checks:

- `pip install -r requirements.txt` succeeds
- `npm install` succeeds in `frontend/`
- backend can connect to Postgres
- app starts in development mode without runtime crashes

---

### Phase 1: Core Architecture and Project Skeleton

Goal: Establish the application foundation.

Tasks:

1. Create backend package structure:
   - `app/`
   - `api/`
   - `core/`
   - `models/`
   - `schemas/`
   - `services/`
   - `db/`
   - `config/`
2. Create frontend app structure:
   - `src/pages/`
   - `src/components/`
   - `src/services/`
   - `src/hooks/`
3. Define base configuration and app startup
4. Add health-check endpoints and baseline app routing
5. Set up logging, error handling, and project settings
6. Create initial Docker or Compose setup for local orchestration

Deliverables:

- Bootable backend app
- Bootable frontend app
- Basic project structure ready for feature work

Success checks:

- Backend root endpoint responds correctly
- Frontend runs without build errors
- Logs show clean startup flow

---

### Phase 2: Database Layer and Persistence

Goal: Provide reliable storage for datasets, jobs, models, and metadata.

Tasks:

1. Define database models for:
   - users or project owners
   - datasets
   - jobs/workflows
   - experiments
   - models
   - metrics
   - explainability reports
2. Configure SQLAlchemy/SQLModel session management
3. Create Alembic migration structure
4. Add database initialization and seed routes if needed
5. Add schemas for API request/response contracts
6. Implement repository/service layer for persistence

Deliverables:

- PostgreSQL-backed persistence layer
- Migration scripts for schema evolution
- Clean data access contracts

Success checks:

- Database migrations run successfully
- CRUD operations work for core entities
- No schema mismatches or broken joins

---

### Phase 3: Dataset Upload and Validation Pipeline

Goal: Support ingestion of datasets from user uploads and external sources.

Tasks:

1. Add file upload endpoint for dataset ingestion
2. Store dataset metadata and file references
3. Validate dataset format, size, delimiters, and schema
4. Detect and classify columns:
   - numeric
   - categorical
   - date/time
   - missing-value fields
5. Implement basic profiling:
   - row/column counts
   - missing values
   - distribution summary
   - data type inference
6. Generate warnings for inconsistent or poor-quality data

Deliverables:

- Dataset upload API
- Dataset profiling results
- Validation and warning system

Success checks:

- Valid CSVs are accepted
- Invalid files are rejected with useful errors
- Dataset profile is displayed correctly to the user

---

### Phase 4: Feature Engineering and Preprocessing Layer

Goal: Prepare data for model training in a consistent and reliable way.

Tasks:

1. Define preprocessing pipeline components
2. Encode categorical features
3. Handle missing values and nulls
4. Standardize numeric features where required
5. Support train/validation split strategy
6. Add reproducible preprocessing configuration to each experiment

Deliverables:

- Reusable preprocessing module
- Experiment-safe transformation pipeline
- Dataset-to-model-ready abstraction

Success checks:

- Preprocessing works for mixed data types
- Transformations are reproducible
- Outputs meet expected training input contracts

---

### Phase 5: AutoML Orchestration Engine

Goal: Run automated model selection and training.

Tasks:

1. Configure AutoML engine selection strategy
   - AutoGluon as primary engine
   - FLAML/XGBoost/LightGBM/CatBoost as alternate or comparison backends
2. Implement experiment creation and job queueing
3. Add model search configuration options:
   - time budget
   - target column
   - metric selection
   - task type (classification/regression)
4. Track training jobs and execution states
5. Save model artifacts and metadata
6. Add result summarization for leaderboard output

Deliverables:

- AutoML job runner
- Experiment tracking system
- Model artifact storage
- Leaderboard / benchmark results

Success checks:

- Training jobs start and complete without crash
- Best-performing models are selected correctly
- Training metadata remains traceable

---

### Phase 6: Model Evaluation and Explainability

Goal: Make model results meaningful and auditable.

Tasks:

1. Compute evaluation metrics:
   - accuracy
   - precision/recall/F1
   - ROC-AUC
   - RMSE / MAE for regression
2. Implement cross-validation or holdout evaluation logic
3. Generate SHAP-based explainability reports
4. Surface feature importance data to the frontend
5. Add reporting of underperforming or biased models when relevant

Deliverables:

- Evaluation API
- Explainability results
- Human-readable model insights

Success checks:

- Metrics are computed correctly
- SHAP output is generated and readable
- Final reports align with model training output

---

### Phase 7: Frontend Experience and User Workflow

Goal: Allow users to interact with the ML platform end to end.

Tasks:

1. Build dashboard screens for:
   - overview
   - dataset upload
   - experiment management
   - model results
   - explainability views
2. Add forms for configuration selection and launch actions
3. Integrate backend API calls with typed client services
4. Show loading states, errors, and validation feedback
5. Add charting for metrics and performance summary

Deliverables:

- Fully navigable frontend experience
- User facing job and dataset flow
- Visual model result pages

Success checks:

- Users can upload a dataset and view a profiling result
- Users can start training and monitor progress
- Result pages render correctly and load fresh data

---

### Phase 8: Debugging, Test Coverage, and Quality Gates

Goal: Catch issues before deployment.

Tasks:

1. Add unit tests for core logic and services
2. Add integration tests around API endpoints and DB usage
3. Add smoke tests for major user workflows
4. Validate model training and profiling pipeline with sample data
5. Perform error logging and debugging for:
   - missing dependencies
   - invalid dataset formats
   - runtime failures in model execution
   - API integration errors
6. Review logs and fix edge cases discovered during testing

Deliverables:

- Test suite covering backend and core workflows
- Debugging logs and known issue tracking
- Stable experimental pipeline

Success checks:

- Key tests pass consistently
- Critical user flows are covered by smoke tests
- No unresolved bootstrap or runtime blockers remain

---

### Phase 9: Deployment Preparation and Release

Goal: Prepare the app for deployment in a real environment.

Tasks:

1. Create production-ready configuration files
2. Add Dockerfile and containerization configuration
3. Configure environment variables for deployment setup
4. Set up a CI/CD pipeline for build and deployment automation
5. Define deployment target options:
   - local container deployment
   - Azure App Service
   - Azure Container Apps
   - self-hosted VM or server
6. Add health checks and monitoring hooks
7. Prepare rollback and restart procedures

Deliverables:

- Production build configuration
- Deployment pipeline or deployment scripts
- Health monitoring and operational runbook

Success checks:

- Application builds successfully in production mode
- Containerized or deployed service stays healthy
- Frontend and backend communicate correctly in target environment

---

## 4. Cross-Cutting Tasks

These tasks span multiple phases:

- Documentation updates
- Error handling standards
- Logging strategy and observability
- Security review for secrets, auth, and API input validation
- Dependency version management
- Project cleanup and architecture consistency reviews

## 5. Recommended Milestones

### Milestone 1: Foundation Ready

- project scaffold complete
- env is working
- DB initialized
- health-check endpoints live

### Milestone 2: Dataset Flow Ready

- upload works
- profiling works
- validation logic is stable

### Milestone 3: AutoML Pipeline Ready

- experiment creation works
- model training can run end to end
- results are stored and retrievable

### Milestone 4: User Experience Ready

- frontend can drive the workflow
- charts and result views are visible
- core workflow is usable by end users

### Milestone 5: Deployment Ready

- app builds cleanly
- tests pass
- deployment path is validated
- monitoring and rollback plan exists

## 6. Debugging Strategy

Use a structured debugging process throughout implementation:

1. Reproduce the issue reliably
2. Capture logs, request payloads, and stack traces
3. Narrow to the failing layer: frontend, API, DB, or ML pipeline
4. Validate assumptions with a targeted local test or isolated script
5. Fix the root cause without broad unneeded refactoring
6. Re-run focused validation checks before proceeding

Common issue areas:

- dataset parsing errors
- dependency mismatch between Python packages
- bad model input schema
- frontend API integration mismatches
- database migration drift
- long-running ML jobs timing out

## 7. Deployment Strategy

Recommended path:

1. Run the app locally in development mode
2. Validate backend + frontend communication
3. Containerize the app for repeatable deployment
4. Deploy to a cloud or staging environment
5. Run smoke tests after deployment
6. Monitor application health and logs
7. Promote to production only after validation passes

Suggested deployment targets:

- Docker Compose for local and test environments
- Azure Container Apps or Azure App Service for managed deployment
- PostgreSQL managed service in Azure or another supported hosting environment

## 8. Final Delivery Checklist

Before marking the project complete, verify:

- project builds successfully
- backend and frontend start correctly
- database connections work
- dataset upload works end to end
- AutoML model training works
- metrics and explanations are generated
- frontend shows results correctly
- tests cover critical flows
- deployment pipeline or runtime setup exists
- operational logs and health checks are in place

## 9. Recommended Next Actions

1. Build the environment and confirm the toolchain is stable
2. Implement the backend skeleton and database foundation
3. Complete dataset upload and profiling
4. Add the AutoML training workflow
5. Validate with focused tests and debug cycles
6. Build the frontend workflow around the proven backend
7. Prepare deployment and monitoring

This plan is intentionally staged to reduce risk and provide clear delivery checkpoints from initial setup through production deployment.
