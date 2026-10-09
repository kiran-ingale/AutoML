# Module 06: AutoML Pipeline and Training Orchestration

## Purpose
Create the training engine responsible for model selection, orchestration, and job execution.

## Goals
- Launch supervised ML tasks
- Support multiple AutoML backends or comparison strategies
- Persist training status and model artifacts
- Manage experiment execution lifecycle

## Step-by-step

### Step 1: Define experiment and job model
- [x] Reuse the run record as the local training-job record.
- [x] Persist task type, target, metric, split options, engine, preset, and attempt ID.
- [x] Persist current status and failure stage/message.
- [ ] Add separate experiment records and append-only status history.

### Step 2: Build the execution runner
- [x] Add a local FastAPI background-task runner with duplicate-start protection.
- [x] Persist queued, running, succeeded, and failed states.
- [x] Allow a failed run to be retried as a new attempt.
- [ ] Add durable queueing and cancellation support (for example, Celery/Redis).

### Step 3: Integrate ML framework selection
- [x] Integrate AutoGluon Tabular as the initial engine.
- [x] Infer AutoGluon binary versus multiclass problem type from the training target.
- [x] Force CPU execution and use a held-out validation split.
- [ ] Add alternate engines and cross-engine comparisons.

### Step 4: Manage configuration options
- [x] Validate target, task type, metric, time limit, test size, and random state.
- [x] Save the validated request and engine settings with the run.

### Step 5: Persist artifacts
- [x] Save the trained model and leaderboard under an attempt-specific storage directory.
- [x] Persist model/leaderboard artifact references, selected model, and leaderboard.
- [x] Retain the uploaded dataset and preprocessing summary for traceability.

### Step 6: Monitor and handle runtime failures
- [x] Log training failures and persist their stage and safe error message.
- [x] Expose status, preprocessing details, leaderboard, artifacts, and failures through run endpoints.
- [x] Permit retry after failure while retaining attempt-specific output paths.
- [ ] Add cancellation, progress events, and durable status history.

## Deliverables
- [x] Local training orchestration service and AutoGluon execution adapter.
- [x] Training API, persisted status/configuration/results, and artifact references.

## Exit Criteria
Users can launch training for an uploaded run and retrieve its status, results, artifact references, or failure details. Verified with mocked lifecycle/API tests and a real CPU-only AutoGluon classification smoke run. The local FastAPI background task is not a durable production queue.
