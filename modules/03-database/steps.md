# Module 03: Database Layer and Persistence Setup

## Purpose
Create a reliable persistence layer for datasets, jobs, experiments, and model metadata.

## Progress
- [x] Core `Run`, `Artifact`, and `Message` models and relationships
- [x] SQLAlchemy session wiring and repository CRUD operations
- [x] Initial Alembic migration applied to the local `automl` database
- [x] Pydantic read schemas and SQLite unit tests
- [x] PostgreSQL create/read/delete smoke test and cascade verification

## Goals
- Configure database connectivity
- Define main project entities
- Create migration foundation
- Implement data access patterns

## Step-by-step

### Step 1: Define core database models
- [x] Create the architecture-defined `Run`, `Artifact`, and `Message` models
- [x] Add relationships and run-scoped cascading cleanup
- [x] Keep JSON fields for parsed requirements, profile, and route data

### Step 2: Configure database sessions
- [x] Set up SQLAlchemy session management
- [x] Expose a FastAPI-compatible session dependency
- [x] Ensure sessions are closed after use

### Step 3: Create migration structure
- [x] Initialize Alembic migration tooling
- [x] Create the first migration for the core tables
- [x] Apply and validate the migration against the local database

### Step 4: Add repository or service access layer
- [x] Add repository operations for runs, artifacts, and messages
- [x] Centralize create, read/list, and delete operations

### Step 5: Add API schemas
- [x] Define Pydantic read schemas
- [x] Represent IDs, JSON metadata, timestamps, and status consistently
- [x] Validate ORM records through the API schemas

### Step 6: Validate persistence flow
- [x] Test repository persistence and run-scoped queries
- [x] Validate status constraints, foreign keys, and cascading deletes
- [x] Verify PostgreSQL CRUD round-trip and migration state

## Deliverables
- Database models and migration scripts
- Session management and repository layer
- API contract mapping for persisted entities

## Exit Criteria
Core run data, artifacts, and messages can be stored, retrieved, and deleted reliably through the backend persistence layer.
