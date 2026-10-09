# Module 04: Dataset Upload, Validation, and Profiling

## Purpose
Allow the platform to accept user data and prepare it for machine learning workflows.

## Progress
- [x] CSV upload API accepts a dataset and task description
- [x] Enforce file extension and configurable upload-size checks
- [x] Validate UTF-8 CSV headers and tabular contents
- [x] Profile rows, column types, missing values, cardinality, duplicates, target guess, and class imbalance
- [x] Store CSV under a generated run-specific path and persist the profile on the `Run`
- [x] Return a typed profile response and verify errors with focused tests

## Goals
- Accept uploaded datasets
- Validate their structure and integrity
- Generate dataset profiles and warnings
- Store dataset metadata for downstream use

## Step-by-step

### Step 1: Build upload API
- [x] Create `POST /runs` for CSV and task-description upload
- [x] Save files to a managed local storage location using a generated run ID
- [x] Record the stored dataset pointer and creation time on the run

### Step 2: Validate dataset format
- [x] Accept CSV files
- [x] Validate extension, UTF-8 encoding, nonblank/unique headers, and nonempty tabular data
- [x] Enforce a configurable maximum upload size and return actionable errors

### Step 3: Profile dataset contents
- [x] Count rows and columns
- [x] Detect per-column missing counts and percentages
- [x] Infer column types and cardinality
- [x] Provide an explicitly labeled last-column target guess and class imbalance ratio

### Step 4: Implement data quality checks
- [x] Flag missing values and duplicate rows
- [x] Return non-critical findings as actionable warnings

### Step 5: Persist dataset state
- [x] Save the dataset pointer, profile, task description, and run status
- [x] Keep uploaded data under a run-specific generated path

### Step 6: Expose profile results through the API
- [x] Return a typed structured profile from the upload API
- [x] Include warnings and target-guess details for frontend confirmation

## Deliverables
- `POST /runs` upload endpoint
- CSV validation and profiling service
- Run records and local CSV storage

## Exit Criteria
A CSV can be uploaded, validated, profiled, and persisted with useful summary details before training begins.
