# Module 05: Feature Engineering and Data Preprocessing

## Purpose
Prepare uploaded data for AutoML input with consistent transformations and data handling.

## Progress
- [x] Validate target column, task type, target values, and dataset split constraints
- [x] Split data reproducibly and stratify classification data
- [x] Fit numeric imputation/scaling and categorical imputation/one-hot encoding on training data only
- [x] Remove training-set constant features and report missing-target removals
- [x] Persist the preprocessing configuration/summary with its run
- [x] Test pure pipeline behavior and stored-run loading/persistence

## Goals
- Normalize mixed-type datasets
- Handle missing values and encoding
- Create pipeline-safe transformation logic
- Keep processing reproducible for each experiment

## Step-by-step

### Step 1: Define preprocessing strategy
- [x] Handle numeric features numerically; string/date-like fields are treated as categorical until a date parser is explicitly requested
- [x] Impute numeric values by the training-set median and categorical values with a missing-value category
- [x] Use a reproducible 80/20 split by default and stratify classification targets

### Step 2: Implement baseline feature transforms
- [x] Convert features using scikit-learn pipelines
- [x] Standardize numeric features using training-only statistics
- [x] One-hot encode categories with unknown validation categories safely ignored

### Step 3: Add scalable preprocessing workflow
- [x] Build reusable per-run preprocessing service
- [x] Fit a separate transformer per invocation/run
- [x] Persist split and transformation summary on the run

### Step 4: Integrate preprocessing with training inputs
- [x] Return training/validation features and targets with a fitted transformer
- [x] Keep train and validation feature widths aligned; use sparse one-hot output when appropriate
- [x] Validate output and task inputs before model training

### Step 5: Add schema validation for training data
- [x] Reject missing/constant targets, non-numeric or non-finite regression targets, and absent target columns
- [x] Remove constant features based on training data; reject datasets with no usable features
- [x] Report unsafe classification splits instead of silently falling back to unstratified splits

### Step 6: Validate with sample data
- [x] Run synthetic mixed-type classification and regression fixtures
- [x] Verify preprocessing fits only on training rows and handles unseen categories
- [x] Validate run-specific data loading and persisted summary behavior

## Deliverables
- Reusable preprocessing pipeline and per-run service
- Fitted transform and train/validation dataset output
- Persisted, typed preprocessing summary for auditability

## Exit Criteria
Uploaded data can be transformed into reproducible training/validation inputs with recorded preprocessing metadata and actionable validation failures.
