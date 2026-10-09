# AutoML-KBP — Product Requirements Document (PRD)

## 1. Summary

AutoML-KBP is an adaptive AutoML platform that accepts a tabular dataset plus
user-stated requirements/constraints, profiles the data, and intelligently
routes the modelling task to either:

- a **traditional GBDT-based AutoML pipeline** (XGBoost / LightGBM / CatBoost
  via an AutoML engine), or
- a **Tabular Foundation Model (TFM) pipeline** (TabPFN / TabICLv2 / hybrid),

then applies model-specific explainability (TreeSHAP or ShapPFN/kernel-ICL
style methods), runs failure/fairness analysis, and produces a **trust &
audit report** alongside predictions. The whole workflow is wrapped in a
conversational (LLM + tool calling) interface so non-expert users can query
or steer the pipeline in plain language.

This PRD defines *what* to build. See `architecture.md` for *how* it's
structured, `rules.md` for coding conventions/boundaries, and `phases.md`
for build order.

## 2. Problem Statement

- Existing AutoML tools (AutoGluon, H2O AutoML, TPOT) lock users into a
  single modelling paradigm (almost always GBDTs) regardless of whether
  that's the best fit for the dataset.
- Newer Tabular Foundation Models (TabPFN, TabICL) can beat GBDTs on small/
  medium datasets with zero training, but nothing decides *automatically*
  when to prefer one over the other.
- Both paradigms are largely black boxes to non-expert users — there's no
  unified, model-aware explainability or audit trail.
- Non-technical users can't easily interrogate *why* a pipeline made the
  choices it made.

## 3. Goals (v1 / MVP scope)

1. Accept a tabular dataset (CSV) + a short natural-language description of
   the task and priorities (e.g. "predict churn, prioritize accuracy").
2. Automatically profile the dataset (rows, columns, dtypes, missing values,
   class balance, likely target column).
3. Route to GBDT-AutoML path or TFM path using a transparent, rule-based
   decision (a full learned router is future scope — see below).
4. Run the chosen pipeline and produce a trained model + predictions on a
   held-out split.
5. Generate model-specific explanations (TreeSHAP for GBDT, a TFM-appropriate
   method for TabPFN/TabICL).
6. Generate a human-readable **trust & audit report**: what was chosen, why,
   dataset characteristics, preprocessing performed, metrics, limitations,
   rejected alternative.
7. Provide a conversational interface (chat panel backed by an LLM with tool
   calling) so the user can ask about the run, request re-runs with
   different priorities, or ask "why" questions about the report.

### Non-goals for v1 (explicitly out of scope)

- Time-series and multimodal (text/image + tabular) tasks.
- Federated / on-prem deployment.
- Active learning / human-in-the-loop router refinement.
- A plugin marketplace for third-party models/explainers.
- Production-grade multi-tenant auth, billing, or scaling — v1 is a single-
  user / demo-grade app.

These map to `Future Scope` in the original proposal and are tracked as
post-MVP.

## 4. Target Users

| Persona | Description | Needs |
|---|---|---|
| **Non-expert analyst** (business/ops, healthcare, finance) | Has a dataset and a question, no ML background | Plain-language input, no config screens, trustworthy plain-English report |
| **Student / researcher** | Wants to compare GBDT vs TFM approaches on a dataset | Visibility into *why* the router picked a path, ability to override |
| **ML engineer (power user)** | Wants a fast baseline / sanity check before building custom pipelines | Full control over constraints, access to raw metrics/artifacts, exportable report |

Primary persona for MVP: the **non-expert analyst** — the UI and report must
be understandable without ML jargon; power-user controls are secondary
(available but not required to get a result).

## 5. Core User Flow (MVP)

1. User uploads a CSV and types a short description of the task and
   priorities (accuracy vs. speed vs. explainability; hardware available).
2. System profiles the dataset and shows a summary (rows/cols, dtypes,
   missing %, detected target, class balance) for user confirmation —
   user can correct the detected target column.
3. System shows the router's decision (GBDT path or TFM path) with a
   one-paragraph rationale *before* running, and lets the user accept or
   override.
4. System runs the selected pipeline (preprocessing → training → eval →
   explainability → fairness/failure scan) with a visible progress trail.
5. System presents: metrics, top feature explanations, flagged risks
   (leakage/imbalance/weak segments), and the full audit report.
6. User can ask the chat panel follow-up questions ("why not the TFM path?",
   "what does this feature importance mean?", "re-run prioritizing speed").

## 6. Feature List (MVP, mapped to proposal's key features)

- **F1 — Natural-language task intake**: free-text task description →
  structured requirements object (target hint, priority: accuracy/speed/
  explainability, hardware flag) via LLM extraction, with a fallback
  structured form if the LLM output can't be parsed.
- **F2 — Automatic dataset profiling**: rows, columns, dtypes, missing-value
  %, cardinality, class imbalance ratio, target detection.
- **F3 — Intelligent router**: rule-based decision using dataset size,
  feature count, missing %, hardware availability, and stated priority.
  Router decision + rationale is always shown, never silent.
- **F4 — Scale-aware TFM selection**: if TFM path chosen, pick TabPFN vs
  TabICLv2 based on row/feature-count limits of each.
- **F5 — Explainable preprocessing**: standard cleaning (impute, encode,
  outlier flags) with a before/after summary table; user can veto a step.
- **F6 — Model-specific explainability**: TreeSHAP for GBDT models;
  TFM-appropriate explanation method for TFM models.
- **F7 — Failure & fairness scan**: underperforming segments, possible
  leakage signals, class-imbalance effects.
- **F8 — Trust & audit report**: consolidated Markdown/HTML/PDF report of
  the full run.
- **F9 — Conversational agent layer**: chat interface with tool calling over
  the pipeline (query results, re-run with new constraints, explain a term).

## 7. Success Criteria / Acceptance Bar

- A user can go from CSV upload to a finished report in one sitting without
  touching code or config files.
- The router's decision is always accompanied by a rationale a non-expert
  can understand.
- Every prediction is traceable to a specific preprocessing + model +
  explanation combination recorded in the audit report.
- The system degrades gracefully: if the LLM call fails, structured forms
  still let the user complete a run; if TFM libraries are unavailable
  (no GPU / install issue), the router falls back to the GBDT path with a
  clear message instead of crashing.

## 8. Key Risks / Open Questions

- TFM explainability methods (ShapPFN etc.) are still immature — MVP may
  need to ship a simpler kernel-based explainer for TFMs and clearly label
  it as an approximation.
- Router in v1 is rule-based, not learned — document this clearly in the
  report so users don't mistake it for a trained model.
- Dataset size/privacy: MVP assumes the user is allowed to upload the data
  to wherever the app/LLM API runs; no on-prem story yet.
