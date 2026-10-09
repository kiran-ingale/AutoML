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
- Add experiment tracking records
- Capture configuration such as task type, target column, and model options
- Record status history for each task

### Step 2: Build the execution runner
- Create a service that launches model training jobs
- Add queueing logic or single-run orchestration for local use
- Handle cancellation or failure states gracefully

### Step 3: Integrate ML framework selection
- Use AutoGluon as the primary engine
- Add support for alternate strategies such as FLAML, XGBoost, LightGBM, or CatBoost
- Keep configuration flexible for future algorithm comparisons

### Step 4: Manage configuration options
- Accept time constraints, metric preferences, and target settings
- Validate inputs before starting model execution
- Save execution config with experiment metadata

### Step 5: Persist artifacts
- Save trained model objects or references
- Store model metadata, leaderboard output, and result files
- Retain the source dataset and preprocessing configuration for traceability

### Step 6: Monitor and handle runtime failures
- Capture logs during training
- Report partial results or failed attempts clearly
- Re-run or debug failed jobs without losing context

## Deliverables
- Experiment/job orchestration pipeline
- AutoML execution service
- Model artifact and status persistence

## Exit Criteria
Users can create an experiment, launch model training, and retrieve stored results or failure details reliably.
