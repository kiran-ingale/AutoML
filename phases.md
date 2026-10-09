# AutoML-KBP — Build Phases

Each phase should end in something runnable/demoable. Don't start a phase's
"nice to have" items before its core items are working end-to-end.

## Phase 0 — Project Skeleton
**Goal:** empty but wired-together app.
- Set up repo structure per `architecture.md` (backend/, frontend/, docs/).
- FastAPI app boots with a health-check endpoint.
- Postgres running via docker-compose; SQLAlchemy models for Run/Artifact/
  Message created and migrated.
- Minimal frontend that can hit the health-check endpoint.
- Docker Compose brings up the whole stack with one command.

## Phase 1 — Dataset Upload & Profiling (Stage 1–2)
**Goal:** upload a CSV, get a profile back — no modelling yet.
- `POST /runs` accepts a CSV + free-text description, stores both, creates a
  Run row.
- `profiling.py`: rows, columns, dtypes, missing %, cardinality, class
  imbalance, target-column guess.
- Frontend: upload form + profile summary view with an editable "target
  column" field.
- Unit tests on profiling using small synthetic CSVs (clean, missing-heavy,
  imbalanced).

## Phase 2 — Requirement Parsing + Router (Stage 3–4)
**Goal:** given profile + free text, produce a routed decision with a
rationale — still no training.
- `requirements.py`: LLM call to turn free text into structured requirements
  (priority: accuracy/speed/explainability; hardware flag), with the
  structured-form fallback from `rules.md` §3 implemented from day one.
- `router.py`: rule-based decision using profile + requirements + TFM
  library availability/limits. Output includes a rationale string that cites
  actual numbers (per `rules.md` §4).
- Frontend: show profile → show router decision + rationale → accept/
  override control (override just flips the path for this run).
- Tests: router unit tests covering each decision boundary (large dataset,
  low explainability priority, no GPU, feature-count over TFM limit, etc.).

## Phase 3 — GBDT Path (Stage 5a/6a, one path only)
**Goal:** first full run producing a trained model + metrics, GBDT only.
- `preprocessing.py`: cleaning with before/after diff, veto hook (veto UI
  can come later; the hook exists now).
- `gbdt_path.py`: AutoGluon/FLAML training on the preprocessed data,
  train/test split, core metrics.
- `orchestrator.py`: chains stages 1–6a with progress events; Run status
  transitions (queued → running → succeeded/failed) implemented per
  `rules.md` §3.
- Frontend: run dashboard showing live stage progress; results view showing
  metrics.

## Phase 4 — Explainability + Audit Report (GBDT path)
**Goal:** the GBDT path is now "complete" per the PRD acceptance bar.
- `explain_gbdt.py`: TreeSHAP, top-feature summary.
- `risk_scan.py`: basic failure/fairness checks (weak segments, obvious
  leakage signals, imbalance effects) — start with the highest-value checks,
  not an exhaustive list.
- `report.py`: builds the Markdown/HTML audit report from all stage
  artifacts; PDF export can follow.
- Frontend: Report page showing metrics, SHAP summary, risk flags, and a
  download/export button.

## Phase 5 — TFM Path (Stage 5b/6b + explainability)
**Goal:** the second modelling path, mirroring Phase 3–4 but for TFMs.
- `tfm_path.py`: TabPFN/TabICLv2 selection and inference per the router's
  scale-aware choice; explicit handling for "TFM unavailable" per
  `rules.md` §3.
- `explain_tfm.py`: ShapPFN or kernel-based fallback, with the approximate-
  explanation disclaimer wired into both UI and report.
- Router's TFM branch now actually executable end-to-end; re-test the
  Phase 2 router boundary tests against real runs.

## Phase 6 — Conversational Agent Layer
**Goal:** chat panel that can answer questions about a completed run and
trigger re-runs.
- `agent/tools.py`: implement the fixed tool contract from
  `architecture.md` §4.
- `agent/agent.py`: tool-calling loop; `POST /runs/{id}/chat` endpoint.
- Frontend: `ChatPanel.tsx` wired to the chat endpoint, per-run message
  history persisted via the Message table.
- Test the "explain a term" tool as a fully deterministic path (no LLM
  variance) since it's meant to be a static glossary lookup.

## Phase 7 — Polish & Hardening
**Goal:** meet the PRD's full acceptance bar, not just the happy path.
- Full pass on `rules.md` §3 fallback behaviors — deliberately trigger each
  failure mode (kill LLM key, force a too-large dataset, disable GPU) and
  confirm graceful handling end-to-end.
- Report export polish (PDF), UI copy pass so a non-expert can follow it
  without ML jargon (or with the glossary tool covering the jargon that
  remains).
- Basic auth/session handling if this moves beyond a single-user demo.
- Load a few real-world-shaped datasets (not just synthetic fixtures) through
  the full pipeline as a final sanity pass.

## Explicitly deferred (do not start early)
Matches PRD §3 non-goals — revisit only after Phase 7:
- Time-series / multimodal support.
- Federated / on-prem deployment.
- Active-learning router refinement.
- Plugin architecture for third-party models/explainers.
- Multi-tenant auth, billing, scaling work.
