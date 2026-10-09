# Module 03: Database Layer and Persistence Setup

## Purpose
Create a reliable persistence layer for datasets, jobs, experiments, and model metadata.

## Goals
- Configure database connectivity
- Define main project entities
- Create migration foundation
- Implement data access patterns

## Step-by-step

### Step 1: Define core database models
- Create models for datasets, experiments, jobs, models, and metrics
- Add relationships needed between records
- Keep schema readable and consistent with project usage

### Step 2: Configure database sessions
- Set up SQLAlchemy or SQLModel session management
- Add dependency injection for transaction handling
- Ensure clean connection lifecycle management

### Step 3: Create migration structure
- Initialize Alembic or equivalent migration tooling
- Create the first migration set for core tables
- Validate migration execution on a fresh database

### Step 4: Add repository or service access layer
- Build abstraction for CRUD operations
- Centralize repetitive database logic
- Use services to isolate strategy from controllers

### Step 5: Add API schemas
- Define request and response models
- Handle id, metadata, timestamps, and status fields consistently
- Align schemas with persistence models

### Step 6: Validate persistence flow
- Test create/read/update/delete operations for core entities
- Check database constraints and integrity rules
- Fix any schema mismatch or migration issue before moving on

## Deliverables
- Database models and migration scripts
- Session management and repository layer
- API contract mapping for persisted entities

## Exit Criteria
Core project data can be stored and retrieved reliably through the backend services.
