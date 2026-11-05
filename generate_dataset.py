"""Generate a reproducible synthetic training set for AutoML router experiments.

The labels are policy-generated synthetic targets, not measured model outcomes.
Run: python generate_dataset.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


SEED = 20261027
N_SCENARIOS = 3_000
OUT_PATH = Path(__file__).resolve().parent / "automl_router_dataset.csv"
REPORT_PATH = Path(__file__).resolve().parent / "dataset_validation_report.txt"
TARGET = "recommended_model_family"
FEATURES = [
    "n_rows",
    "n_features",
    "n_numeric_features",
    "n_categorical_features",
    "missing_value_ratio",
    "class_imbalance_ratio",
    "task_type",
    "n_classes",
    "high_cardinality_ratio",
    "hardware_type",
    "available_ram_gb",
    "latency_requirement_seconds",
    "training_budget_minutes",
    "priority_accuracy",
    "priority_speed",
    "priority_interpretability",
    "priority_resource_efficiency",
    "requires_probability_output",
    "requires_feature_importance",
    "supports_task",
    "memory_feasibility",
    "latency_feasibility",
    "training_budget_feasibility",
]


def _runtime_estimates(row: dict[str, object]) -> dict[str, dict[str, float]]:
    """Return modeled resource/runtime estimates; these are synthetic proxies."""
    rows = float(row["n_rows"])
    features = float(row["n_features"])
    categorical = float(row["n_categorical_features"])
    missing = float(row["missing_value_ratio"])
    gpu = row["hardware_type"] == "GPU-available"
    accel = 0.45 if gpu else 1.0
    shape = rows * (0.65 + features / 180.0)

    # Approximate peak RAM in GiB and training seconds. No measurement is implied.
    gbdt_ram = 0.35 + rows * features * 8 / (1024**3) * (2.5 + categorical / max(features, 1))
    tfm_ram = 0.75 + rows * min(features, 2_000) * 8 / (1024**3) * (5.0 if gpu else 8.0)
    gbdt_train = 4.0 + shape / 1_700 + categorical * rows / 180_000
    tfm_train = (3.0 + shape / 6_000 + missing * 2.0) * accel

    # p95 prediction latency estimate for one batch of 100 rows, seconds.
    gbdt_latency = (0.008 + features * 0.00005 + rows * 0.00000005) * (0.72 if gpu else 1.0)
    tfm_latency = (0.025 + features * 0.00011 + rows * 0.00000011) * (0.48 if gpu else 1.0)
    return {
        "GBDT": {"ram": gbdt_ram, "train_seconds": gbdt_train, "latency": gbdt_latency},
        "TABULAR_FOUNDATION_MODEL": {
            "ram": tfm_ram,
            "train_seconds": tfm_train,
            "latency": tfm_latency,
        },
    }


def _candidates(row: dict[str, object]) -> tuple[list[str], dict[str, dict[str, float]]]:
    estimates = _runtime_estimates(row)
    supported = ["GBDT", "TABULAR_FOUNDATION_MODEL"]
    feasible: list[str] = []
    for family in supported:
        estimate = estimates[family]
        if family == "TABULAR_FOUNDATION_MODEL":
            row_limit = 100_000 if row["hardware_type"] == "GPU-available" else 1_000
            if int(row["n_rows"]) > row_limit or int(row["n_features"]) > 2_000:
                continue
            if row["task_type"] == "classification" and int(row["n_classes"]) > 160:
                continue
        if estimate["ram"] > float(row["available_ram_gb"]):
            continue
        if estimate["latency"] > float(row["latency_requirement_seconds"]):
            continue
        if estimate["train_seconds"] > float(row["training_budget_minutes"]) * 60:
            continue
        feasible.append(family)
    return feasible, estimates


def _utility(row: dict[str, object], family: str, estimates: dict[str, dict[str, float]]) -> float:
    """Deterministic weighted utility, with priorities on a common 1–5 scale."""
    rows = float(row["n_rows"])
    features = float(row["n_features"])
    categorical_ratio = float(row["n_categorical_features"]) / features
    imbalance = float(row["class_imbalance_ratio"])
    estimate = estimates[family]
    other = "TABULAR_FOUNDATION_MODEL" if family == "GBDT" else "GBDT"

    # Smooth heuristic priors, deliberately not asserted as actual performance.
    if family == "GBDT":
        accuracy_score = 0.69 + 0.09 / (1 + np.exp(-(np.log10(max(rows, 100)) - 3.6) * 1.2))
        accuracy_score += 0.035 * categorical_ratio + 0.02 * (1 - imbalance)
        speed_score = 1 / (1 + estimate["train_seconds"] / 35)
        interpretability_score = 0.80
        resource_score = 1 / (1 + estimate["ram"] / 12 + estimate["train_seconds"] / 150)
    else:
        accuracy_score = 0.70 + 0.14 * np.exp(-((np.log10(max(rows, 100)) - 3.45) ** 2) / 1.6)
        accuracy_score += 0.012 * (1 - categorical_ratio) + 0.012 * (1 - imbalance)
        speed_score = 1 / (1 + estimate["train_seconds"] / 35)
        interpretability_score = 0.67
        resource_score = 1 / (1 + estimate["ram"] / 12 + estimate["train_seconds"] / 150)

    if row["requires_feature_importance"] and family == "TABULAR_FOUNDATION_MODEL":
        interpretability_score *= 0.52
    if row["requires_probability_output"] and row["task_type"] == "classification":
        # Both families can provide probabilities; this term rewards confidence
        # output as a requirement without claiming equal calibration quality.
        accuracy_score += 0.005

    weights = np.array(
        [row["priority_accuracy"], row["priority_speed"],
         row["priority_interpretability"], row["priority_resource_efficiency"]],
        dtype=float,
    )
    scores = np.array([accuracy_score, speed_score, interpretability_score, resource_score])
    weighted = float(np.dot(weights, scores) / weights.sum())

    # Strong feasibility preference and deterministic GBDT tie break are handled
    # outside this function. Mild task/size suitability adjusts close decisions.
    if row["task_type"] == "regression" and family == "TABULAR_FOUNDATION_MODEL" and rows > 5_000:
        weighted -= 0.015
    if rows > 25_000 and family == "GBDT":
        weighted += 0.015
    del other
    return weighted


def _recommend(row: dict[str, object]) -> tuple[str, bool, bool, bool]:
    candidates, estimates = _candidates(row)
    # These are scenario-level availability flags: at least one candidate
    # family satisfies each constraint. They are computed without referencing
    # the eventual recommendation and therefore do not leak the target.
    ram_feasible = any(
        e["ram"] <= float(row["available_ram_gb"]) for e in estimates.values()
    )
    latency_feasible = any(
        e["latency"] <= float(row["latency_requirement_seconds"])
        for e in estimates.values()
    )
    budget_feasible = any(
        e["train_seconds"] <= float(row["training_budget_minutes"]) * 60
        for e in estimates.values()
    )
    if not candidates:
        return "NO_FEASIBLE_MODEL", ram_feasible, latency_feasible, budget_feasible

    scored = [( _utility(row, family, estimates), family) for family in candidates]
    # Lexical tie-break makes exact ties reproducible and inspectable.
    recommendation = sorted(scored, key=lambda pair: (-pair[0], pair[1]))[0][1]
    return recommendation, ram_feasible, latency_feasible, budget_feasible


def _make_row(rng: np.random.Generator) -> dict[str, object]:
    task = str(rng.choice(["classification", "regression"], p=[0.72, 0.28]))
    n_features = int(np.clip(round(np.exp(rng.uniform(np.log(3), np.log(500)))), 3, 500))
    n_categorical = int(rng.binomial(n_features, rng.uniform(0.05, 0.65)))
    n_numeric = n_features - n_categorical
    n_rows = int(np.clip(round(np.exp(rng.uniform(np.log(100), np.log(200_000)))), 100, 200_000))
    hardware = str(rng.choice(["CPU-only", "GPU-available"], p=[0.66, 0.34]))
    ram = float(round(rng.uniform(2, 64 if hardware == "CPU-only" else 80), 2))
    n_classes = int(rng.choice([2, 3, 4, 5, 8, 10, 20, 50, 100, 160, 200])) if task == "classification" else 0
    n_high_card = int(rng.integers(0, n_categorical + 1)) if n_categorical else 0
    priorities = rng.integers(1, 6, size=4)
    return {
        "n_rows": n_rows,
        "n_features": n_features,
        "n_numeric_features": n_numeric,
        "n_categorical_features": n_categorical,
        "missing_value_ratio": round(float(rng.beta(1.4, 12) * 0.65), 4),
        "class_imbalance_ratio": round(float(rng.uniform(0.015, 1.0)), 4) if task == "classification" else 1.0,
        "task_type": task,
        "n_classes": n_classes,
        "high_cardinality_ratio": round(n_high_card / n_categorical, 4) if n_categorical else 0.0,
        "hardware_type": hardware,
        "available_ram_gb": ram,
        # p95 prediction latency for a fixed batch of 100 prediction rows.
        "latency_requirement_seconds": round(float(np.exp(rng.uniform(np.log(0.01), np.log(5.0)))), 4),
        "training_budget_minutes": round(float(np.exp(rng.uniform(np.log(0.5), np.log(240)))), 3),
        "priority_accuracy": int(priorities[0]),
        "priority_speed": int(priorities[1]),
        "priority_interpretability": int(priorities[2]),
        "priority_resource_efficiency": int(priorities[3]),
        "requires_probability_output": bool(task == "classification" and rng.random() < 0.42),
        "requires_feature_importance": bool(rng.random() < 0.28),
        # Both modeled families support the two supervised task types. This is
        # scenario-level compatibility, not a feature of the chosen label.
        "supports_task": True,
    }


def generate_dataset() -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    rows: list[dict[str, object]] = []
    seen: set[tuple[object, ...]] = set()
    while len(rows) < N_SCENARIOS:
        row = _make_row(rng)
        key = tuple(row[column] for column in FEATURES[:20])
        if key in seen:
            continue
        seen.add(key)
        label, memory_feasible, latency_feasible, budget_feasible = _recommend(row)
        row["memory_feasibility"] = bool(memory_feasible)
        row["latency_feasibility"] = bool(latency_feasible)
        row["training_budget_feasibility"] = bool(budget_feasible)
        row[TARGET] = label
        rows.append(row)
    frame = pd.DataFrame(rows, columns=FEATURES + [TARGET])
    frame.to_csv(OUT_PATH, index=False)
    write_validation_report(frame)
    return frame


def write_validation_report(frame: pd.DataFrame) -> None:
    """Write structural checks and descriptive QA; no benchmark claims."""
    lines = [
        "Synthetic AutoML Router Dataset — Validation Report",
        f"Generator seed: {SEED}",
        "",
        "1. Dataset dimensions",
        f"Rows (excluding header): {len(frame)}",
        f"Columns (including target): {len(frame.columns)}",
        f"Target: {TARGET}",
        f"Input columns: {len(FEATURES)}",
        "",
        "2. Missing values",
        frame.isna().sum().to_string(),
        "",
        "3. Duplicate scenarios",
        f"Duplicate complete rows: {int(frame.duplicated().sum())}",
        f"Duplicate input scenarios: {int(frame[FEATURES].duplicated().sum())}",
        "",
        "4. Categorical unique-value counts",
    ]
    categorical = ["task_type", "hardware_type", TARGET]
    for column in categorical:
        values = frame[column].value_counts(dropna=False)
        lines.append(f"{column}: {frame[column].nunique(dropna=False)} unique")
        lines.append(values.to_string())
    lines += ["", "5. Feature consistency checks"]
    check_results = {
        "n_numeric_features + n_categorical_features == n_features": bool(
            (frame.n_numeric_features + frame.n_categorical_features == frame.n_features).all()
        ),
        "n_features positive": bool((frame.n_features > 0).all()),
        "n_rows >= 100": bool((frame.n_rows >= 100).all()),
        "missing_value_ratio in [0, 1]": bool(frame.missing_value_ratio.between(0, 1).all()),
        "high_cardinality_ratio in [0, 1]": bool(frame.high_cardinality_ratio.between(0, 1).all()),
        "classification has n_classes >= 2": bool(
            (frame.loc[frame.task_type == "classification", "n_classes"] >= 2).all()
        ),
        "regression n_classes == 0": bool(
            (frame.loc[frame.task_type == "regression", "n_classes"] == 0).all()
        ),
        "regression class_imbalance_ratio convention == 1.0": bool(
            (frame.loc[frame.task_type == "regression", "class_imbalance_ratio"] == 1.0).all()
        ),
        "feature counts consistent with high-cardinality ratio": bool(
            frame.loc[frame.n_categorical_features == 0, "high_cardinality_ratio"].eq(0).all()
        ),
        "priorities all in [1, 5]": bool(
            frame[["priority_accuracy", "priority_speed", "priority_interpretability", "priority_resource_efficiency"]]
            .apply(lambda col: col.between(1, 5).all()).all()
        ),
        "supports_task true for supported supervised tasks": bool(frame.supports_task.all()),
    }
    lines.extend(f"{name}: {'PASS' if passed else 'FAIL'}" for name, passed in check_results.items())
    lines += ["", "6. Target-class distribution", frame[TARGET].value_counts().to_string(), ""]
    if frame[TARGET].value_counts(normalize=True).min() < 0.05:
        lines.append("WARNING: at least one target class is below 5% of rows.")
    else:
        lines.append("No target class is below 5% of rows.")

    numerical = [
        "n_rows", "n_features", "n_numeric_features", "n_categorical_features",
        "missing_value_ratio", "class_imbalance_ratio", "n_classes",
        "high_cardinality_ratio", "available_ram_gb", "latency_requirement_seconds",
        "training_budget_minutes", "priority_accuracy", "priority_speed",
        "priority_interpretability", "priority_resource_efficiency",
    ]
    lines += ["", "7. Numerical feature distributions (pandas describe)", frame[numerical].describe().round(4).to_string()]
    lines += ["", "8. Target cross-tabulations"]
    for column in [
        "task_type", "hardware_type", "priority_accuracy", "priority_speed",
        "priority_interpretability", "priority_resource_efficiency",
    ]:
        lines += [f"\n{TARGET} by {column} (row percentages):", pd.crosstab(
            frame[column], frame[TARGET], normalize="index"
        ).round(3).to_string()]

    # Shortcut audit: labels are not deterministic copies of any single input.
    # Report the best single categorical predictor's majority-class accuracy.
    single_feature_scores: dict[str, float] = {}
    for column in ["task_type", "hardware_type", "supports_task", "memory_feasibility", "latency_feasibility"]:
        grouped = pd.crosstab(frame[column], frame[TARGET])
        single_feature_scores[column] = float(grouped.max(axis=1).sum() / len(frame))
    x_probe = pd.get_dummies(frame[FEATURES], columns=["task_type", "hardware_type"], dtype=float)
    x_train, x_test, y_train, y_test = train_test_split(
        x_probe, frame[TARGET], test_size=0.25, random_state=SEED, stratify=frame[TARGET]
    )
    shallow_probe = DecisionTreeClassifier(max_depth=3, min_samples_leaf=25, random_state=SEED)
    shallow_probe.fit(x_train, y_train)
    probe_accuracy = float(accuracy_score(y_test, shallow_probe.predict(x_test)))
    majority_baseline = float(frame[TARGET].value_counts(normalize=True).iloc[0])
    lines += [
        "",
        "9. Leakage and shortcut audit",
        f"Target excluded from model input feature list: {TARGET not in FEATURES}",
        "No scenario ID, hidden utility score, rule ID, or generation score is exported.",
        "Feasibility flags are scenario-level (whether any family meets the named constraint), computed before target selection.",
        "supports_task means at least one modeled family supports the supervised task; it is true for both task types in this candidate set.",
        "Exact input duplicates: " + str(int(frame[FEATURES].duplicated().sum())),
        "Best single-feature categorical shortcut accuracies (descriptive):",
        pd.Series(single_feature_scores).sort_values(ascending=False).round(4).to_string(),
        f"Depth-3 decision-tree holdout accuracy (synthetic-label learnability probe only): {probe_accuracy:.4f}",
        f"Majority-class baseline accuracy: {majority_baseline:.4f}",
        "This random-row probe is not a real benchmark or an unbiased estimate of generalization to new datasets.",
        "Flag: the synthetic rule intentionally creates learnable structure; investigate a probe above 0.98 for direct shortcuts or overly deterministic boundaries.",
        "Rows are independently sampled; no shared scenario template or template identifier is used.",
        "Do a grouped split if future revisions introduce templates or repeated base scenarios.",
        "",
        "10. Label policy, conflict resolution, and limitations",
        "Each candidate family is first filtered by synthetic task/row limits, estimated memory, p95 latency, and training-budget feasibility.",
        "If none remain, the label is NO_FEASIBLE_MODEL. Otherwise a deterministic weighted utility ranks feasible GBDT and tabular foundation model candidates.",
        "The four priority weights use the same 1–5 scale and are normalized by their sum. Accuracy, speed, interpretability, and resource proxies contribute to utility; deterministic tie-breaking favors GBDT.",
        "Feature-importance requirements reduce the TFM interpretability proxy. Probability output is only requested for classification; both families are modeled as capable of probabilities there.",
        "Runtime and memory are hand-designed heuristic estimates, not measurements. Row/feature caps are synthetic application policies, not universal vendor limits.",
        "Labels encode this declared synthetic policy only. They are not ground truth and do not establish that any family is better, faster, or more resource efficient in practice.",
        "For real router training, replace or validate labels using measured, paired benchmark runs on identical splits; retain failures and uncertainty, and use development tasks to tune policy before evaluating held-out tasks.",
        "Latency definition: modeled p95 prediction latency for one batch of 100 prediction rows, in seconds; this is not end-to-end training time.",
        "Regression convention: class_imbalance_ratio=1.0 (not applicable/balanced sentinel); n_classes=0.",
        "",
        "Warnings / interpretation",
        "All estimates and labels are synthetic. The dataset is suitable only for initial pipeline/software experimentation.",
    ]
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    result = generate_dataset()
    print(f"Wrote {len(result):,} scenarios to {OUT_PATH}")
    print(f"Wrote validation report to {REPORT_PATH}")
    print(result[TARGET].value_counts().to_string())
