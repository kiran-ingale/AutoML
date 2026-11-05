"""Train and evaluate CPU-friendly classifiers on the synthetic router dataset.

Run from the project root with the project environment:
    python train_router.py

This evaluates reproduction of synthetic policy labels only. It is not a
benchmark of real model quality or evidence that a routing policy is optimal.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from threadpoolctl import threadpool_limits


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "automl_router_dataset.csv"
OUTPUT_DIR = ROOT / "storage" / "router_model"
SEED = 42
TARGET = "recommended_model_family"


def _make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    categorical = [column for column in X.columns if X[column].dtype == "object"]
    numeric = [column for column in X.columns if column not in categorical]
    numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    categorical_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        [("numeric", numeric_pipe, numeric), ("categorical", categorical_pipe, categorical)],
        remainder="drop",
        verbose_feature_names_out=True,
    )


def _make_pipeline(X: pd.DataFrame, classifier: Any) -> Pipeline:
    return Pipeline([("preprocess", _make_preprocessor(X)), ("classifier", classifier)])


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    data = pd.read_csv(DATA_PATH)
    if TARGET not in data:
        raise ValueError(f"Missing required target column {TARGET!r}.")
    if data.isna().any().any():
        raise ValueError("Dataset contains missing values; inspect before training.")
    if data.drop(columns=TARGET).duplicated().any():
        raise ValueError("Duplicate input scenarios found; refusing evaluation.")

    X = data.drop(columns=TARGET)
    y = data[TARGET]
    classes = sorted(y.unique().tolist())
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=SEED,
        stratify=y,
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    scorers = {
        "accuracy": "accuracy",
        "balanced_accuracy": "balanced_accuracy",
        "macro_f1": "f1_macro",
        "weighted_f1": "f1_weighted",
    }
    candidates = {
        "RandomForestClassifier": RandomForestClassifier(
            n_estimators=400,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced_subsample",
            # Single-process execution is predictable and avoids worker-pool
            # restrictions in CPU-only environments.
            n_jobs=1,
            random_state=SEED,
        ),
        "DecisionTreeClassifier": DecisionTreeClassifier(
            min_samples_leaf=8,
            class_weight="balanced",
            random_state=SEED,
        ),
        "HistGradientBoostingClassifier": HistGradientBoostingClassifier(
            max_iter=180,
            learning_rate=0.08,
            max_leaf_nodes=15,
            l2_regularization=1.0,
            random_state=SEED,
        ),
        "DummyClassifier_most_frequent": DummyClassifier(strategy="most_frequent"),
    }

    rows: list[dict[str, Any]] = []
    fitted: dict[str, Pipeline] = {}
    for name, classifier in candidates.items():
        pipeline = _make_pipeline(X_train, classifier)
        start = time.perf_counter()
        with threadpool_limits(limits=1):
            cv_result = cross_validate(
                pipeline,
                X_train,
                y_train,
                cv=cv,
                scoring=scorers,
                n_jobs=1,
                return_train_score=False,
            )
        cv_seconds = time.perf_counter() - start
        start = time.perf_counter()
        with threadpool_limits(limits=1):
            pipeline.fit(X_train, y_train)
        fit_seconds = time.perf_counter() - start
        start = time.perf_counter()
        with threadpool_limits(limits=1):
            predictions = pipeline.predict(X_test)
        predict_seconds = time.perf_counter() - start
        fitted[name] = pipeline
        record: dict[str, Any] = {
            "model": name,
            "cv_seconds": cv_seconds,
            "fit_seconds": fit_seconds,
            "predict_seconds": predict_seconds,
            "holdout_accuracy": accuracy_score(y_test, predictions),
            "holdout_balanced_accuracy": balanced_accuracy_score(y_test, predictions),
            "holdout_macro_f1": f1_score(y_test, predictions, average="macro"),
            "holdout_weighted_f1": f1_score(y_test, predictions, average="weighted"),
        }
        for score_name in scorers:
            values = cv_result[f"test_{score_name}"]
            record[f"cv_{score_name}_mean"] = float(np.mean(values))
            record[f"cv_{score_name}_std"] = float(np.std(values, ddof=1))
        rows.append(record)

    results = pd.DataFrame(rows).sort_values(
        ["cv_macro_f1_mean", "cv_balanced_accuracy_mean", "cv_accuracy_mean"],
        ascending=False,
    )
    winner_name = str(results.iloc[0]["model"])
    winner = fitted[winner_name]
    winner_predictions = winner.predict(X_test)
    report_dict = classification_report(
        y_test, winner_predictions, labels=classes, output_dict=True, zero_division=0
    )
    matrix = confusion_matrix(y_test, winner_predictions, labels=classes)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)
    pd.DataFrame(
        matrix,
        index=[f"actual_{label}" for label in classes],
        columns=[f"predicted_{label}" for label in classes],
    ).to_csv(OUTPUT_DIR / "confusion_matrix.csv")
    (OUTPUT_DIR / "classification_report.json").write_text(
        json.dumps(report_dict, indent=2), encoding="utf-8"
    )
    joblib.dump(winner, OUTPUT_DIR / "router_model.joblib", compress=3)

    importance_path = OUTPUT_DIR / "feature_importance.csv"
    classifier = winner.named_steps["classifier"]
    if hasattr(classifier, "feature_importances_"):
        feature_names = winner.named_steps["preprocess"].get_feature_names_out()
        importance = pd.DataFrame(
            {"transformed_feature": feature_names, "importance": classifier.feature_importances_}
        ).sort_values("importance", ascending=False)
        importance.to_csv(importance_path, index=False)

    metadata = {
        "dataset": str(DATA_PATH.name),
        "dataset_sha256": _sha256(DATA_PATH),
        "rows": len(data),
        "input_features": X.shape[1],
        "target": TARGET,
        "target_distribution": y.value_counts().to_dict(),
        "random_seed": SEED,
        "split": "stratified 75% train / 25% holdout; 5-fold shuffled stratified CV on training partition",
        "selection_rule": "highest mean 5-fold training-partition macro-F1, then balanced accuracy, then accuracy",
        "selected_model": winner_name,
        "python_library_versions": {"python": __import__("sys").version, "scikit_learn": sklearn.__version__, "pandas": pd.__version__, "numpy": np.__version__},
        "label_interpretation": "Predicts synthetic policy-generated routing labels only; not measured model superiority.",
        "features_used": X.columns.tolist(),
        "excluded_target": TARGET,
    }
    (OUTPUT_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    lines = [
        "Synthetic AutoML Router — Model Training and Evaluation",
        "",
        "WARNING: This model predicts synthetic policy labels. Metrics measure label reproduction only; they are not evidence of real-world routing quality or model-family superiority.",
        "",
        f"Dataset rows: {len(data)}",
        f"Input features: {X.shape[1]}",
        f"Target distribution: {y.value_counts().to_dict()}",
        f"Split: {metadata['split']}",
        f"Random seed: {SEED}",
        f"Selected model: {winner_name}",
        "Selection uses training-only 5-fold CV; final holdout is not used to select the classifier.",
        "",
        "Model comparison (holdout metrics and training-only CV):",
        results.to_string(index=False, float_format=lambda value: f"{value:.4f}"),
        "",
        "Selected model class-level holdout report:",
        pd.DataFrame(report_dict).transpose().round(4).to_string(),
        "",
        "Confusion matrix (rows=actual, columns=predicted):",
        pd.DataFrame(matrix, index=classes, columns=classes).to_string(),
        "",
        "Limitations:",
        "- Random scenario-level splitting tests interpolation among synthetic scenarios, not generalization to unseen datasets or benchmark tasks.",
        "- Labels encode hand-designed rules and estimated resource proxies; no actual AutoGluon, TabPFN, or TabICL fits were run here.",
        "- The model is a prototype. Real use requires replacing/validating labels with paired benchmark measurements and grouped task-level splits.",
        "",
        f"Artifacts saved to: {OUTPUT_DIR}",
    ]
    (OUTPUT_DIR / "evaluation_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(results.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"Selected: {winner_name}")
    print(f"Saved model and reports under {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
