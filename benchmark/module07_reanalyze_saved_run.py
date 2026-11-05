from __future__ import annotations

import json
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import module07_runner as runner

SOURCE_ZIP = Path(r"D:\module07_artifacts (1).zip")
OUT_ROOT = Path(__file__).with_name("module07_corrected_reanalysis")
ARTIFACTS = OUT_ROOT / "module07_artifacts"

if not SOURCE_ZIP.exists():
    raise FileNotFoundError(SOURCE_ZIP)
if OUT_ROOT.exists():
    raise FileExistsError(f"Refusing to overwrite existing reanalysis: {OUT_ROOT}")
ARTIFACTS.mkdir(parents=True)

with zipfile.ZipFile(SOURCE_ZIP) as archive:
    for entry in archive.infolist():
        if entry.is_dir():
            continue
        relative = Path(entry.filename)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Unsafe archive path: {entry.filename}")
        destination = OUT_ROOT / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        with archive.open(entry) as src, destination.open("wb") as dst:
            shutil.copyfileobj(src, dst)


def read_jsonl(path: Path) -> pd.DataFrame:
    return pd.read_json(path, lines=True)


def write_jsonl(df: pd.DataFrame, path: Path) -> None:
    df.to_json(path, orient="records", lines=True, force_ascii=False)


raw = read_jsonl(ARTIFACTS / "raw_attempts.jsonl")
routes = read_jsonl(ARTIFACTS / "router_decisions.jsonl")
labels = read_jsonl(ARTIFACTS / "stage1_development_labels.jsonl")

# Restore the GBDT class implied by the original paired comparison and frozen
# tie rule. Leave genuinely unavailable paired outcomes unlabeled.
reason_to_family = {
    "AutoGluon_validation_RMSE_better": "gbdt",
    "within_tie_tolerance_GBDT_preferred": "gbdt",
    "AutoGluon_validation_macro_F1_better": "gbdt",
    "selected_TFM_validation_RMSE_better": "tfm",
    "selected_TFM_validation_macro_F1_better": "tfm",
}
for i, row in labels.iterrows():
    reason = row.get("label_reason")
    if pd.isna(row.get("family_winner")) and reason in reason_to_family:
        labels.at[i, "family_winner"] = reason_to_family[reason]

# Refit the inspectable Stage 1 tree from saved, corrected development labels;
# this does not fit any AutoGluon or TFM candidate models.
corrected_routes: list[dict[str, Any]] = []
corrected_eval: list[dict[str, Any]] = []
raw_records = raw.to_dict(orient="records")
for route in routes.to_dict(orient="records"):
    task = route["task"]
    seed = int(route["seed"])
    heldout = route["dataset"]
    heldout_labels = labels[
        (labels.outer_heldout_dataset == heldout)
        & (labels.seed == seed)
        & (labels.task == task)
    ].to_dict(orient="records")

    eligibility = route["eligible_engines"]
    features = {
        **route["profile"],
        "tabpfn_eligible": bool(eligibility.get("tabpfn_cuda", {}).get("eligible")),
        "tabicl_eligible": bool(eligibility.get("tabicl_cuda", {}).get("eligible")),
    }
    selected_family, selector_info, _ = runner._select_family(heldout_labels, features)
    selector_info.setdefault("predicted_family", selected_family)

    tfm_engine = route["development_tfm_only_engine"]
    if selected_family == "tfm" and tfm_engine == "autogluon_cpu":
        selected_engine = "autogluon_cpu"
        selected_family = "gbdt"
        selection_reason = "no_eligible_TFM_for_stage2"
    else:
        selected_engine = tfm_engine if selected_family == "tfm" else "autogluon_cpu"
        selection_reason = selector_info.get(
            "selection_reason",
            selector_info.get("reason", "selector_prediction_eligible"),
        )

    label_rows_for_route = [
        {
            "dataset": item["development_dataset"],
            "family_winner": item.get("family_winner"),
            "tfm_engine_for_label": item.get("tfm_engine_for_label"),
            "label_reason": item.get("label_reason"),
        }
        for item in heldout_labels
    ]
    corrected_route = {
        **route,
        "selected_family": selected_family,
        "selected_engine": selected_engine,
        "selector_predicted_family_before_eligibility_guard": selector_info["predicted_family"],
        "selection_reason": selection_reason,
        "selector_method": selector_info["method"],
        "selector_confidence": selector_info.get("confidence"),
        "selector_confidence_calibrated": selector_info.get(
            "confidence_is_calibrated_probability", False
        ),
        "selector_support_status": selector_info.get("support_status"),
        "selector_metadata": selector_info,
        "development_task_count": selector_info["development_task_count"],
        "development_family_label_counts": selector_info["winner_label_counts"],
        "stage1_development_family_labels": label_rows_for_route,
    }
    corrected_routes.append(corrected_route)

    candidate_rows = [
        row for row in raw_records
        if row.get("dataset") == heldout
        and int(row.get("seed", -1)) == seed
        and row.get("phase") == "outer_test_evaluation"
    ]
    outer_records = {row["engine"]: row for row in candidate_rows}
    observed_winner, observed_ties, outer_scores = runner._winner(
        task, eligibility, candidate_rows
    )
    selected_row = outer_records.get(selected_engine, {})
    selected_success = selected_row.get("status") == "success"
    metric = runner._primary(task)
    selected_metric = (
        selected_row.get("metrics", {}).get(metric) if selected_success else None
    )
    if selected_metric is None or not outer_scores:
        regret = None
    else:
        best_score = min(outer_scores.values()) if task == "regression" else max(outer_scores.values())
        if task == "regression":
            regret = max(0.0, (selected_metric - best_score) / max(abs(best_score), 1e-12))
        else:
            regret = max(0.0, (best_score - selected_metric) / max(abs(best_score), 0.01))
    matches = bool(selected_engine in observed_ties) if selected_success and observed_ties else None
    tfm_row = outer_records.get(tfm_engine, {})
    ag_row = outer_records.get("autogluon_cpu", {})
    corrected_eval.append({
        "dataset": heldout,
        "task": task,
        "seed": seed,
        "selected_family": selected_family,
        "primary_metric": metric,
        "selected_engine": selected_engine,
        "selected_status": selected_row.get("status", "not_run"),
        "selected_primary_score": selected_metric,
        "selected_route_normalized_regret_if_success": regret,
        "autogluon_status": ag_row.get("status", "not_eligible"),
        "autogluon_primary_score": ag_row.get("metrics", {}).get(metric),
        "development_tfm_only_engine": tfm_engine,
        "development_tfm_only_status": tfm_row.get("status", "not_eligible"),
        "development_tfm_only_primary_score": tfm_row.get("metrics", {}).get(metric),
        "development_tfm_winner_counts": route.get("development_tfm_winner_counts", {}),
        "outer_observed_best_engine_descriptive_only": observed_winner,
        "outer_observed_ties": observed_ties,
        "selected_matches_outer_test_best_or_tie_descriptive_only": matches,
        "outer_successful_scores": outer_scores,
        "selector_metadata": selector_info,
    })

corrected_routes_df = pd.DataFrame(corrected_routes)
corrected_eval_df = pd.DataFrame(corrected_eval)
write_jsonl(labels, ARTIFACTS / "stage1_development_labels.jsonl")
labels.to_csv(ARTIFACTS / "stage1_development_labels.csv", index=False)
write_jsonl(corrected_routes_df, ARTIFACTS / "router_decisions.jsonl")
corrected_routes_df.to_csv(ARTIFACTS / "router_decisions.csv", index=False)
write_jsonl(corrected_eval_df, ARTIFACTS / "outer_evaluation.jsonl")
corrected_eval_df.to_csv(ARTIFACTS / "outer_evaluation.csv", index=False)

# Recreate seed-aggregated task summaries and grouped bootstrap comparisons.
group_eval = corrected_eval_df.groupby(["dataset", "task"], as_index=False).agg(
    selected_completion_rate=("selected_status", lambda x: float(np.mean(np.asarray(x) == "success"))),
    mean_selected_normalized_regret=("selected_route_normalized_regret_if_success", "mean"),
    selected_best_or_tie_rate=("selected_matches_outer_test_best_or_tie_descriptive_only", "mean"),
    selected_family=("selected_family", "first"),
    selected_engine=("selected_engine", "first"),
)
group_eval.to_csv(ARTIFACTS / "selector_summary_by_dataset.csv", index=False)

corrected_routes_df.groupby(["task", "selected_engine"], dropna=False).agg(
    heldout_dataset_seed_count=("dataset", "size"),
    independent_dataset_group_count=("dataset", "nunique"),
).reset_index().to_csv(ARTIFACTS / "route_counts.csv", index=False)
corrected_routes_df.groupby(["task", "selected_family"], dropna=False).agg(
    heldout_dataset_seed_count=("dataset", "size"),
    independent_dataset_group_count=("dataset", "nunique"),
).reset_index().to_csv(ARTIFACTS / "route_family_counts.csv", index=False)

metric_rows = []
task_rng = np.random.default_rng(2026)
for task, task_groups in group_eval.groupby("task"):
    for measure in [
        "selected_completion_rate",
        "mean_selected_normalized_regret",
        "selected_best_or_tie_rate",
    ]:
        values = task_groups[measure].dropna().to_numpy(dtype=float)
        if not len(values):
            continue
        draws = np.asarray([
            task_rng.choice(values, size=len(values), replace=True).mean()
            for _ in range(5000)
        ])
        metric_rows.append({
            "task": task,
            "measure": measure,
            "n_independent_dataset_groups": len(values),
            "mean": float(values.mean()),
            "bootstrap_95_ci_low": float(np.quantile(draws, 0.025)),
            "bootstrap_95_ci_high": float(np.quantile(draws, 0.975)),
            "unit": "dataset group after seed aggregation",
            "interpretation": (
                "descriptive comparison to the outer-test oracle"
                if measure == "selected_best_or_tie_rate"
                else "exploratory held-out dataset estimate"
            ),
        })
pd.DataFrame(metric_rows).to_csv(ARTIFACTS / "task_level_metrics_summary.csv", index=False)

comparison_rows = []
comparison_rng = np.random.default_rng(81)
for task, task_eval in corrected_eval_df.groupby("task"):
    metric = runner._primary(task)
    for baseline, score_col, status_col in [
        ("AutoGluon-only", "autogluon_primary_score", "autogluon_status"),
        ("development-selected-TFM-only", "development_tfm_only_primary_score", "development_tfm_only_status"),
    ]:
        paired = []
        for _dataset, rows in task_eval.groupby("dataset"):
            valid = rows[
                rows["selected_primary_score"].notna()
                & (rows[status_col] == "success")
                & rows[score_col].notna()
            ]
            if valid.empty:
                continue
            selected = float(valid["selected_primary_score"].mean())
            base = float(valid[score_col].mean())
            paired.append(
                (base - selected) / max(abs(base), 1e-12)
                if task == "regression"
                else selected - base
            )
        if not paired:
            continue
        values = np.asarray(paired, dtype=float)
        draws = np.asarray([
            comparison_rng.choice(values, size=len(values), replace=True).mean()
            for _ in range(5000)
        ])
        comparison_rows.append({
            "task": task,
            "primary_metric": metric,
            "selector_vs": baseline,
            "n_independent_dataset_groups": len(values),
            "mean_selector_improvement": float(values.mean()),
            "bootstrap_95_ci_low": float(np.quantile(draws, 0.025)),
            "bootstrap_95_ci_high": float(np.quantile(draws, 0.975)),
            "comparison_scale": "relative RMSE improvement for regression; macro-F1 points for classification",
            "sign_convention": "positive favors selector; dataset-level pairs after seed aggregation",
            "interpretation": "corrected post-hoc grouped leave-one-dataset-out estimate; 45-second AutoGluon budget",
        })
pd.DataFrame(comparison_rows).to_csv(ARTIFACTS / "grouped_comparisons.csv", index=False)

protocol_path = ARTIFACTS / "protocol.json"
protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
protocol["postprocessing_note"] = (
    "Stage 1 GBDT labels were restored from recorded paired-outcome reasons where the original code left them null; "
    "selector/routes and selected outer scores were recomputed from saved records without refitting candidate models."
)
protocol_path.write_text(json.dumps(protocol, indent=2), encoding="utf-8")

status_path = ARTIFACTS / "run_status.json"
status = json.loads(status_path.read_text(encoding="utf-8"))
status["route_engine_counts"] = corrected_routes_df["selected_engine"].value_counts().to_dict()
status["route_family_counts"] = corrected_routes_df["selected_family"].value_counts().to_dict()
status["selected_route_success_count"] = int((corrected_eval_df["selected_status"] == "success").sum())
status["selected_route_completion_rate"] = float(np.mean(corrected_eval_df["selected_status"] == "success"))
status["postprocessed_at_utc"] = datetime.now(timezone.utc).isoformat()
status["postprocessing_note"] = "Routes and outer metrics corrected from saved candidate fits; no model refits performed."
status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")

note = {
    "source_archive": str(SOURCE_ZIP),
    "analysis": "Corrected post-hoc Stage 1 labels and recomputed routes/outer evaluations from saved model-fit records.",
    "model_fits_repeated": False,
    "correction": "Rows labeled AutoGluon_validation_*_better or within_tie_tolerance_GBDT_preferred are family_winner=gbdt.",
    "fit_budget_seconds": protocol.get("auto_gluon_fit_budget_seconds"),
    "limitations": [
        "45-second AutoGluon budget differs from the original 300-second protocol.",
        "TabICL regression checkpoint SHA-256 remains unverified in source provenance.",
        "Results cover the 20 benchmark task groups and tested software/checkpoint setup only.",
    ],
}
(ARTIFACTS / "corrected_reanalysis_note.json").write_text(json.dumps(note, indent=2), encoding="utf-8")

zip_path = OUT_ROOT / "module07_artifacts_corrected_reanalysis.zip"
with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
    for path in ARTIFACTS.rglob("*"):
        if path.is_file():
            archive.write(path, path.relative_to(OUT_ROOT))

print("Corrected route counts:")
print(pd.read_csv(ARTIFACTS / "route_family_counts.csv").to_string(index=False))
print("\nCorrected grouped comparisons:")
print(pd.read_csv(ARTIFACTS / "grouped_comparisons.csv").to_string(index=False))
print(f"\nCorrected artifacts: {OUT_ROOT}")
print(f"ZIP: {zip_path}")
