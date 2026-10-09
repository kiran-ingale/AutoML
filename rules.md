# AutoML-KBP — Build Rules & Boundaries

Purpose: guardrails for anyone (human or AI coding assistant) working on this
codebase, so generated code stays consistent, safe, and doesn't quietly grow
scope. Read alongside `architecture.md`.

## 1. Libraries — use these

- **AutoML engine**: AutoGluon (preferred) or FLAML. Do NOT hand-roll a
  hyperparameter search loop — that's what the AutoML engine is for.
- **GBDT models**: XGBoost, LightGBM, CatBoost — only via the AutoML engine's
  wrapper, not called directly unless debugging.
- **TFMs**: TabPFN, TabICLv2 (or their current maintained successors). Check
  each library's row/feature-count limits before calling — the router must
  enforce these, not discover them via a crash.
- **Explainability**: `shap` for TreeSHAP. For TFMs, use the library's own
  recommended explainer (e.g. ShapPFN) if available; otherwise fall back to
  a generic kernel/permutation explainer from `shap` and label the output as
  approximate in both the UI and the audit report.
- **Data handling**: pandas, numpy, scikit-learn (splits, metrics, standard
  preprocessing transformers). Don't reimplement train/test splitting,
  imputation, or standard encoders by hand.
- **Backend**: FastAPI + Pydantic for all request/response schemas — no bare
  dicts crossing the API boundary.
- **DB access**: SQLAlchemy (or SQLModel). No raw SQL string interpolation.
- **LLM calls**: one thin wrapper module per provider; tool-calling schemas
  defined once in `agent/tools.py` and imported everywhere — don't redefine
  tool schemas inline in prompts.

## 2. Libraries / patterns — avoid

- No custom deep-learning training loops for the GBDT/TFM paths — this
  project routes to *existing* AutoML/TFM libraries, it doesn't train
  models from scratch.
- No global mutable state for run data — everything about a run lives in
  the DB/Artifact store, not in in-process variables, so runs are resumable
  and inspectable.
- No silent `except: pass` — every caught exception must either be handled
  with a specific fallback (see §3) or re-raised with context.
- No calling TFM libraries without first checking the router's own size/
  feature limits — don't rely on the library to error out gracefully.
- No hardcoded API keys / secrets in code — env vars only, loaded via
  `core/config.py`.
- No frontend calling ML libraries directly (no client-side "quick model" —
  all modelling happens server-side; keep that boundary strict for auditability).
- Don't let the LLM agent execute arbitrary/generated code against the
  dataset — it may only call the fixed tool functions in `agent/tools.py`.

## 3. Error Handling & Fallback Behavior

These are product requirements, not just code style — the PRD's "graceful
degradation" acceptance bar depends on them:

- **LLM requirement-extraction fails or returns unparseable output** → fall
  back to a structured form the user fills manually; never block the run on
  the LLM call.
- **TFM libraries unavailable / no GPU** → router must detect this at
  startup and always resolve to the GBDT path, with a visible "TFM path
  disabled: <reason>" message — never a stack trace.
- **Dataset exceeds TFM's row/feature limits** → router excludes the TFM
  path automatically and logs the specific limit that was hit in the
  rationale (don't just say "not chosen").
- **Any pipeline stage fails** → mark the Run as `failed` with the stage
  name and a user-readable message; partial artifacts from completed stages
  stay available. Never leave a Run stuck in `running` on a crash.
- **Chat/agent tool call fails** → return an apologetic in-chat message with
  the underlying error class, not a raw traceback; log the traceback
  server-side.

## 4. Explainability & Reporting Integrity

- Every number in the audit report must be traceable to an artifact
  generated during that run — never fabricate or approximate a metric
  without labeling it as approximate.
- TFM explanations are experimental — always show a short disclaimer next
  to them in both UI and report ("approximate; TFM explainability methods
  are an active research area").
- The router's rationale text must reference the actual profiling numbers
  used in the decision (e.g. "12,000 rows exceeds TabPFN's practical limit
  of ~10,000") — no generic boilerplate rationale.

## 5. Scope Boundaries for AI Coding Assistants

When asking an AI assistant to generate code for this repo:

- Stay inside the folder structure defined in `architecture.md` — don't
  invent new top-level modules without updating that doc first.
- One pipeline stage = one file in `backend/app/pipeline/`. Don't merge
  stages into a single mega-function; the orchestrator depends on each
  stage being independently callable and independently testable.
- Don't add new LLM-agent tools without adding them to the tool contract in
  `architecture.md` §4 first — the tool list should stay short and
  deliberate, not grow ad hoc per feature request.
- Anything listed under PRD §3 "Non-goals for v1" (time-series/multimodal,
  federated deployment, plugin marketplace, multi-tenant auth) is out of
  scope until `phases.md` explicitly schedules it — don't let an assistant
  "helpfully" start building these early.
- Tests: every new pipeline stage needs at least one unit test using a small
  synthetic dataset fixture (not the full AutoGluon/TabPFN run) so the test
  suite stays fast.

## 6. Data & Privacy

- Treat every uploaded dataset as potentially sensitive by default — no
  logging of raw data rows, only shapes/stats/hashes.
- Audit reports and artifacts are scoped per Run; no cross-run data mixing.
