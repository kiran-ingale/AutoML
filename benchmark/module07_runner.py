"""Leakage-safe, dataset-grouped two-stage router evaluation for Module 07.

Stage 1 predicts GBDT versus TFM family; Stage 2 conditionally selects an
eligible TFM. Development labels and choices exclude the current task group,
and outer test outcomes are used only for final scoring.
"""

from __future__ import annotations

import hashlib
import importlib.metadata as metadata
import io
import json
import math
import os
import platform
import shutil
import sys
import time
import traceback
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


MANIFEST_PATH = Path(__file__).with_name("module07_datasets.json")
OUTER_TEST_SIZE = 0.20
INNER_VALIDATION_SIZE = 0.25
CLASSIFICATION_PRIMARY = "macro_f1"
REGRESSION_PRIMARY = "rmse"
CLASSIFICATION_TIE_TOLERANCE = 0.005
REGRESSION_RELATIVE_TIE_TOLERANCE = 0.01
ENGINE_TIE_ORDER = ["autogluon_cpu", "tabicl_cuda", "tabpfn_cuda"]
ENGINES = ["autogluon_cpu", "tabpfn_cuda", "tabicl_cuda"]
TABICL_REPO = "jingang/TabICL"
TABICL_REVISION = "4dcd344ece2c00be9e831fdd35bed57b5ad83e19"
TABICL_FILES = {
    "classification": "tabicl-classifier-v2-20260212.ckpt",
    "regression": "tabicl-regressor-v2-20260212.ckpt",
}
TABPFN_FILES = {
    "classification": "tabpfn-v3-classifier-v3_default.ckpt",
    "regression": "tabpfn-v3-regressor-v3_default.ckpt",
}
TABICL_CLASSIFICATION_SHA256 = (
    "bdc7dbd5e4ff21f8f0456fcf90c6b7cdf72dbea960f2d05b19bec19f9b3d4ed0"
)
MAX_GPU_ROWS = 100_000
MAX_FEATURES = 2_000
TABPFN_MAX_CLASSES = 160


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    if pd.isna(value):
        return None
    return str(value)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, default=_json_default, allow_nan=False),
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _versions() -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for package in [
        "torch",
        "numpy",
        "pandas",
        "pyarrow",
        "scikit-learn",
        "autogluon.tabular",
        "tabpfn",
        "tabicl",
        "ucimlrepo",
        "huggingface-hub",
    ]:
        try:
            result[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            result[package] = None
    return result


def _load_manifest() -> list[dict[str, Any]]:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if len(manifest) != 20:
        raise ValueError(
            f"Module 07 requires exactly 20 frozen task groups; found {len(manifest)}."
        )
    names = [entry["name"] for entry in manifest]
    ids = [entry["uci_id"] for entry in manifest]
    if len(set(names)) != len(names) or len(set(ids)) != len(ids):
        raise ValueError("Dataset manifest has duplicate task names or UCI IDs.")
    for entry in manifest:
        if not entry.get("license_reviewed") or entry.get("license") != "CC BY 4.0":
            raise ValueError(
                f"{entry['name']} has no verified, permitted dataset license record."
            )
        if not entry.get("target") or not entry.get("citation"):
            raise ValueError(f"{entry['name']} is missing its target or citation.")
    return manifest


def _load_dataset(spec: dict[str, Any]) -> tuple[pd.DataFrame, str, dict[str, Any]]:
    from ucimlrepo import fetch_ucirepo

    loader_method = "ucimlrepo"
    retrieval_url = spec["source_url"]
    try:
        dataset = fetch_ucirepo(id=int(spec["uci_id"]))
        features = dataset.data.features.copy()
        targets = dataset.data.targets.copy()
        metadata_dict = getattr(dataset, "metadata", {}) or {}
    except Exception as exc:
        # UCI currently serves the Seeds dataset page and archive, but the
        # ucimlrepo API may mark its dataset endpoint unavailable for import.
        # In that specific case, fall back to the official UCI static archive.
        if spec["name"] != "seeds" or "not available for import" not in str(exc):
            raise
        archive_url = "https://archive.ics.uci.edu/static/public/236/seeds.zip"
        request = urllib.request.Request(
            archive_url, headers={"User-Agent": "AutoML-KBP-Module07/1.0"}
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            archive_bytes = response.read()
        with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
            member = next(
                (name for name in archive.namelist() if Path(name).name == "seeds_dataset.txt"),
                None,
            )
            if member is None:
                raise ValueError("Official UCI Seeds archive has no seeds_dataset.txt file.")
            raw = archive.read(member)
        names = [
            "area", "perimeter", "compactness", "kernel_length",
            "kernel_width", "asymmetry_coefficient", "kernel_groove_length", "Class",
        ]
        seed_frame = pd.read_csv(
            io.BytesIO(raw), sep=r"\s+", header=None, names=names, engine="python"
        )
        if seed_frame.shape != (210, 8):
            raise ValueError(
                f"Unexpected Seeds archive shape {seed_frame.shape}; expected (210, 8)."
            )
        features = seed_frame.drop(columns=["Class"])
        targets = seed_frame[["Class"]]
        metadata_dict = {}
        loader_method = "official_uci_static_zip_fallback"
        retrieval_url = archive_url
    requested_target = spec["target"]
    target = requested_target
    if target not in targets.columns:
        aliases = spec.get("target_aliases", [])
        exact_matches = [
            candidate
            for candidate in [requested_target, *aliases]
            if candidate in targets.columns
        ]
        if len(exact_matches) == 1:
            target = exact_matches[0]
        else:
            normalize = lambda value: "".join(
                character.casefold() for character in str(value) if character.isalnum()
            )
            normalized_matches = [
                candidate
                for candidate in targets.columns
                if normalize(candidate)
                in {normalize(requested_target), *(normalize(a) for a in aliases)}
            ]
            if len(normalized_matches) == 1:
                target = normalized_matches[0]
    if target not in targets.columns:
        raise ValueError(
            f"Target mismatch for {spec['name']}: expected {requested_target!r}; "
            f"aliases={spec.get('target_aliases', [])!r}; "
            f"UCI returned {list(targets.columns)!r}."
        )
    # Only the declared response is included; other response columns never
    # become predictors (for example Energy Efficiency Y2 when predicting Y1).
    frame = features.copy()
    frame[target] = targets[target].to_numpy()
    frame = frame.dropna(subset=[target]).reset_index(drop=True)
    if frame[target].nunique(dropna=True) < 2:
        raise ValueError("Target has fewer than two observed values/classes.")
    observed_license = metadata_dict.get("license") if isinstance(metadata_dict, dict) else None
    provenance = {
        "uci_id": int(spec["uci_id"]),
        "source_url": spec["source_url"],
        "retrieval_url": retrieval_url,
        "loader_method": loader_method,
        "citation": spec["citation"],
        "declared_license": spec["license"],
        "license_reviewed": bool(spec["license_reviewed"]),
        "loader_license_field": observed_license,
        "target": target,
        "requested_target": requested_target,
        "target_columns_returned": list(targets.columns),
        "target_class_counts": (
            frame[target].value_counts(dropna=False).to_dict()
            if spec["task"] == "classification"
            else None
        ),
        "rows_after_missing_target_removal": int(len(frame)),
        "columns_including_target": int(frame.shape[1]),
        "raw_frame_sha256": hashlib.sha256(
            pd.util.hash_pandas_object(frame, index=True).values.tobytes()
        ).hexdigest(),
        "retrieved_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    return frame, target, provenance


def _metrics(
    task: str, y_true: pd.Series, prediction: np.ndarray
) -> dict[str, float]:
    if task == "regression":
        return {
            "rmse": float(mean_squared_error(y_true, prediction) ** 0.5),
            "mae": float(mean_absolute_error(y_true, prediction)),
            "r2": float(r2_score(y_true, prediction)),
        }
    return {
        "macro_f1": float(f1_score(y_true, prediction, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y_true, prediction)),
    }


def _safe_split(
    frame: pd.DataFrame,
    target: str,
    task: str,
    test_size: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    stratify = None
    strategy = "unstratified"
    if task == "classification":
        counts = frame[target].value_counts(dropna=True)
        class_count = int(counts.size)
        n_test = int(math.ceil(len(frame) * test_size))
        n_train = len(frame) - n_test
        if (
            class_count > 1
            and int(counts.min()) >= 2
            and n_test >= class_count
            and n_train >= class_count
        ):
            stratify = frame[target]
            strategy = "stratified"
        else:
            strategy = "unstratified_due_to_rare_or_small_class"
    train, test = train_test_split(
        frame,
        test_size=test_size,
        random_state=seed,
        stratify=stratify,
    )
    return train.reset_index(drop=True), test.reset_index(drop=True), strategy


def _profile(
    train_frame: pd.DataFrame, target: str, task: str
) -> dict[str, Any]:
    """Compute route-time features from the outer training partition only."""
    X = train_frame.drop(columns=[target]).copy()
    y = train_frame[target]
    numeric = [column for column in X if pd.api.types.is_numeric_dtype(X[column])]
    categorical = [column for column in X if column not in numeric]
    missing_ratio = float(X.replace("?", np.nan).isna().to_numpy().mean())
    categorical_high_cardinality = (
        sum(X[column].nunique(dropna=True) > 20 for column in categorical)
        / len(categorical)
        if categorical
        else 0.0
    )
    if task == "classification":
        class_counts = y.value_counts(dropna=True)
        class_count = int(class_counts.size)
        imbalance_ratio = float(class_counts.min() / class_counts.max())
    else:
        class_count = 0
        imbalance_ratio = 1.0
    return {
        "n_rows": int(len(X)),
        "n_features": int(X.shape[1]),
        "n_numeric_features": int(len(numeric)),
        "n_categorical_features": int(len(categorical)),
        "missing_value_ratio": missing_ratio,
        "high_cardinality_ratio": float(categorical_high_cardinality),
        "task_type": task,
        "n_classes": class_count,
        "class_balance_ratio": imbalance_ratio,
    }


def _eligibility(
    task: str,
    train_rows: int,
    feature_count: int,
    class_count: int,
    gpu_available: bool,
    tabpfn_ready: bool,
    tabicl_ready: bool,
) -> dict[str, dict[str, Any]]:
    candidates: dict[str, dict[str, Any]] = {
        "autogluon_cpu": {"eligible": True, "reason": "always included as baseline"}
    }
    if not gpu_available:
        candidates["tabpfn_cuda"] = {"eligible": False, "reason": "CUDA unavailable"}
        candidates["tabicl_cuda"] = {"eligible": False, "reason": "CUDA unavailable"}
        return candidates
    if not tabpfn_ready:
        candidates["tabpfn_cuda"] = {
            "eligible": False,
            "reason": "TabPFN package/token/license readiness unavailable",
        }
    elif train_rows > MAX_GPU_ROWS:
        candidates["tabpfn_cuda"] = {
            "eligible": False,
            "reason": f"training rows exceed project GPU cap {MAX_GPU_ROWS}",
        }
    elif task == "classification" and class_count > TABPFN_MAX_CLASSES:
        candidates["tabpfn_cuda"] = {
            "eligible": False,
            "reason": f"class count exceeds project cap {TABPFN_MAX_CLASSES}",
        }
    else:
        candidates["tabpfn_cuda"] = {
            "eligible": True,
            "reason": "package/token ready and within project task-size policy",
        }
    if not tabicl_ready:
        candidates["tabicl_cuda"] = {
            "eligible": False,
            "reason": "TabICL package/checkpoint readiness unavailable",
        }
    elif train_rows < 300:
        candidates["tabicl_cuda"] = {
            "eligible": False,
            "reason": "below the project 300-row minimum evaluation policy",
        }
    elif train_rows > MAX_GPU_ROWS or feature_count > MAX_FEATURES:
        candidates["tabicl_cuda"] = {
            "eligible": False,
            "reason": "exceeds project GPU row/feature policy",
        }
    else:
        candidates["tabicl_cuda"] = {
            "eligible": True,
            "reason": "package/checkpoint ready and within project task-size policy",
        }
    return candidates


def _make_tfm(engine: str, task: str, seed: int, checkpoint: str | None) -> Any:
    if engine == "tabpfn_cuda":
        from tabpfn import TabPFNClassifier, TabPFNRegressor

        model_type = TabPFNClassifier if task == "classification" else TabPFNRegressor
        return model_type(
            model_path=TABPFN_FILES[task], device="cuda", random_state=seed
        )
    from tabicl import TabICLClassifier, TabICLRegressor

    model_type = TabICLClassifier if task == "classification" else TabICLRegressor
    return model_type(model_path=checkpoint, device="cuda", random_state=seed, n_jobs=1)


def _fit_score(
    engine: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_score: pd.DataFrame,
    y_score: pd.Series,
    task: str,
    target: str,
    seed: int,
    output_path: Path,
    checkpoint: str | None,
    ag_seconds: int,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "engine": engine,
        "status": "failure",
        "start_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    started = time.perf_counter()
    cuda_peak_memory_bytes = None
    try:
        if engine == "autogluon_cpu":
            from autogluon.tabular import TabularPredictor

            train_data = X_train.copy()
            train_data[target] = y_train.to_numpy()
            metric = "root_mean_squared_error" if task == "regression" else "f1_macro"
            predictor = TabularPredictor(
                label=target,
                problem_type="regression" if task == "regression" else None,
                eval_metric=metric,
                path=str(output_path),
                verbosity=0,
            )
            fit_started = time.perf_counter()
            predictor.fit(
                train_data=train_data,
                time_limit=ag_seconds,
                presets="good_quality",
                num_gpus=0,
                dynamic_stacking=False,
                fit_weighted_ensemble=False,
            )
            fit_seconds = time.perf_counter() - fit_started
            predict_started = time.perf_counter()
            prediction = np.asarray(predictor.predict(X_score))
            predict_seconds = time.perf_counter() - predict_started
            record["best_model"] = predictor.model_best
        else:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            model_started = time.perf_counter()
            model = _make_tfm(engine, task, seed, checkpoint)
            initialization_seconds = time.perf_counter() - model_started
            record["model_class"] = type(model).__name__
            if engine == "tabpfn_cuda":
                resolved_model_path = getattr(model, "model_path", None)
                record["model_version"] = "TabPFN-3"
                record["requested_checkpoint"] = TABPFN_FILES[task]
                record["resolved_model_path"] = (
                    str(resolved_model_path) if resolved_model_path is not None else None
                )
            fit_started = time.perf_counter()
            model.fit(X_train, y_train)
            fit_seconds = time.perf_counter() - fit_started
            predict_started = time.perf_counter()
            prediction = np.asarray(model.predict(X_score))
            predict_seconds = time.perf_counter() - predict_started
            if torch.cuda.is_available():
                cuda_peak_memory_bytes = int(torch.cuda.max_memory_allocated())
            record["initialization_seconds"] = initialization_seconds
        record.update(
            {
                "status": "success",
                "metrics": _metrics(task, y_score, prediction),
                "fit_seconds": float(fit_seconds),
                "predict_seconds": float(predict_seconds),
                "fit_predict_seconds": float(fit_seconds + predict_seconds),
                "end_to_end_seconds": float(time.perf_counter() - started),
                "cuda_peak_memory_allocated_bytes": cuda_peak_memory_bytes,
            }
        )
    except Exception as exc:  # retain every failure as an outcome
        message = str(exc)
        token = os.environ.get("TABPFN_TOKEN")
        if token:
            message = message.replace(token, "[REDACTED]")
        trace = traceback.format_exc()[-3000:]
        if token:
            trace = trace.replace(token, "[REDACTED]")
        record.update(
            {
                "status": "failure",
                "failure_type": type(exc).__name__,
                "failure_message": message[:1000],
                "traceback": trace,
                "end_to_end_seconds": float(time.perf_counter() - started),
            }
        )
    finally:
        if engine == "autogluon_cpu":
            # Each call uses a unique AutoGluon path; model files are not
            # required to score or reproduce these recorded predictions.
            shutil.rmtree(output_path, ignore_errors=True)
    return record


def _primary(task: str) -> str:
    return REGRESSION_PRIMARY if task == "regression" else CLASSIFICATION_PRIMARY


def _winner(
    task: str,
    eligible: dict[str, dict[str, Any]],
    candidate_records: list[dict[str, Any]],
) -> tuple[str | None, list[str], dict[str, float]]:
    successful = {
        row["engine"]: row["metrics"][_primary(task)]
        for row in candidate_records
        if row.get("status") == "success"
    }
    successful = {
        engine: value
        for engine, value in successful.items()
        if eligible.get(engine, {}).get("eligible")
        and value is not None
        and math.isfinite(float(value))
    }
    if not successful:
        return None, [], {}
    minimize = task == "regression"
    best = min(successful.values()) if minimize else max(successful.values())
    tied = []
    for engine, value in successful.items():
        if minimize:
            within_tolerance = abs(value - best) <= max(
                abs(best) * REGRESSION_RELATIVE_TIE_TOLERANCE, 1e-12
            )
        else:
            within_tolerance = abs(value - best) <= CLASSIFICATION_TIE_TOLERANCE
        if within_tolerance:
            tied.append(engine)
    winner = min(tied, key=ENGINE_TIE_ORDER.index)
    return winner, tied, successful


def _profile_to_features(
    profile: dict[str, Any], eligible: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    # TabPFN/TabICL readiness/eligibility are included as pre-fit metadata.
    return {
        **profile,
        "tabpfn_eligible": bool(eligible.get("tabpfn_cuda", {}).get("eligible")),
        "tabicl_eligible": bool(eligible.get("tabicl_cuda", {}).get("eligible")),
    }


def _stage1_development_rows(
    development_rows: list[dict[str, Any]], task: str
) -> list[dict[str, Any]]:
    """Label family outcomes using a TFM choice learned without that task."""
    task_rows = [row for row in development_rows if row.get("task") == task]
    stage1_rows: list[dict[str, Any]] = []
    for row in task_rows:
        peers = [candidate for candidate in task_rows if candidate["dataset"] != row["dataset"]]
        features = row["features"]
        eligibility = {
            "autogluon_cpu": {"eligible": True},
            "tabpfn_cuda": {"eligible": bool(features.get("tabpfn_eligible"))},
            "tabicl_cuda": {"eligible": bool(features.get("tabicl_eligible"))},
        }
        tfm_engine, _ = _development_tfm_choice(peers, task, eligibility)
        scores = row.get("development_scores", {})
        gbdt_score = scores.get("autogluon_cpu")
        tfm_score = scores.get(tfm_engine)
        family_winner = None
        family_label_reason = "paired_family_outcome_unavailable"
        if (
            tfm_engine != "autogluon_cpu"
            and gbdt_score is not None
            and tfm_score is not None
            and math.isfinite(float(gbdt_score))
            and math.isfinite(float(tfm_score))
        ):
            if task == "regression":
                tolerance = max(
                    abs(float(gbdt_score)) * REGRESSION_RELATIVE_TIE_TOLERANCE,
                    1e-12,
                )
                if float(tfm_score) < float(gbdt_score) - tolerance:
                    family_winner = "tfm"
                    family_label_reason = "selected_TFM_validation_RMSE_better"
                elif float(gbdt_score) < float(tfm_score) - tolerance:
                    family_winner = "gbdt"
                    family_label_reason = "AutoGluon_validation_RMSE_better"
                else:
                    family_winner = "gbdt"
                    family_label_reason = "within_tie_tolerance_GBDT_preferred"
            else:
                difference = float(tfm_score) - float(gbdt_score)
                if difference > CLASSIFICATION_TIE_TOLERANCE:
                    family_winner = "tfm"
                    family_label_reason = "selected_TFM_validation_macro_F1_better"
                elif difference < -CLASSIFICATION_TIE_TOLERANCE:
                    family_winner = "gbdt"
                    family_label_reason = "AutoGluon_validation_macro_F1_better"
                else:
                    family_winner = "gbdt"
                    family_label_reason = "within_tie_tolerance_GBDT_preferred"
        elif tfm_engine == "autogluon_cpu":
            family_label_reason = "no_eligible_TFM_for_paired_family_label"
        stage1_rows.append(
            {
                **row,
                "family_winner": family_winner,
                "family_label_reason": family_label_reason,
                "stage1_tfm_engine_for_label": tfm_engine,
            }
        )
    return stage1_rows


def _select_family(
    development_rows: list[dict[str, Any]], test_features: dict[str, Any]
) -> tuple[str, dict[str, Any], DecisionTreeClassifier | None]:
    usable = [row for row in development_rows if row.get("family_winner")]
    labels = sorted({row["family_winner"] for row in usable})
    if not usable:
        return "gbdt", {
            "method": "fallback",
            "reason": "no development tasks produced paired family labels",
            "development_task_count": 0,
            "winner_label_counts": {},
        }, None
    if len(labels) == 1:
        return "gbdt", {
            "method": "conservative-fallback",
            "reason": "development tasks contain only one family label; selector has not learned a choice boundary",
            "development_task_count": len(usable),
            "winner_label_counts": pd.Series(
                [row["family_winner"] for row in usable]
            ).value_counts().to_dict(),
            "confidence": None,
            "support_status": "insufficient_label_diversity",
        }, None
    X = pd.DataFrame([row["features"] for row in usable])
    y = np.asarray([row["family_winner"] for row in usable])
    categorical = [column for column in X if X[column].dtype == "object"]
    X_encoded = pd.get_dummies(X, columns=categorical, dummy_na=True, dtype=float)
    test_encoded = pd.get_dummies(
        pd.DataFrame([test_features]), columns=categorical, dummy_na=True, dtype=float
    ).reindex(columns=X_encoded.columns, fill_value=0.0)
    selector = DecisionTreeClassifier(
        max_depth=2,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=17,
    )
    selector.fit(X_encoded, y)
    raw_choice = str(selector.predict(test_encoded)[0])
    probabilities = selector.predict_proba(test_encoded)[0]
    confidence = float(np.max(probabilities))
    unseen_categories = []
    for column in categorical:
        if test_features.get(column) not in set(X[column].dropna().astype(str)):
            unseen_categories.append(column)
    outside_training_range = []
    for column in X.columns:
        if column in categorical:
            continue
        value = test_features.get(column)
        if value is None:
            continue
        minimum, maximum = float(X[column].min()), float(X[column].max())
        if float(value) < minimum or float(value) > maximum:
            outside_training_range.append(column)
    support_ok = not unseen_categories and not outside_training_range
    confidence_threshold = 0.55
    if not support_ok or confidence < confidence_threshold:
        chosen = "gbdt"
        route_reason = (
            "profile outside selector development support"
            if not support_ok
            else "selector leaf confidence below conservative threshold"
        )
    else:
        chosen = raw_choice
        route_reason = "selector prediction met support and confidence checks"
    metadata_row = {
        "method": "stage1_family_decision_tree",
        "model": "DecisionTreeClassifier(max_depth=2,min_samples_leaf=2,class_weight='balanced')",
        "development_task_count": len(usable),
        "winner_label_counts": pd.Series(y).value_counts().to_dict(),
        "features": list(X_encoded.columns),
        "classes_seen": labels,
        "tree_depth": int(selector.get_depth()),
        "leaf_count": int(selector.get_n_leaves()),
        "predicted_family": raw_choice,
        "confidence": confidence,
        "confidence_is_calibrated_probability": False,
        "confidence_threshold": confidence_threshold,
        "support_status": "supported" if support_ok else "outside_training_support",
        "unseen_categorical_features": unseen_categories,
        "outside_training_range_features": outside_training_range,
        "selection_reason": route_reason,
    }
    return chosen, metadata_row, selector


def _development_tfm_choice(
    development_rows: list[dict[str, Any]],
    task: str,
    heldout_eligibility: dict[str, dict[str, Any]],
) -> tuple[str, dict[str, int]]:
    """Choose a TFM from validation outcomes on other development datasets."""
    heldout_tfms = [
        engine
        for engine in ["tabpfn_cuda", "tabicl_cuda"]
        if heldout_eligibility.get(engine, {}).get("eligible")
    ]
    if not heldout_tfms:
        return "autogluon_cpu", {}
    wins = {engine: 0 for engine in ["tabpfn_cuda", "tabicl_cuda"]}
    for row in development_rows:
        if row.get("task") != task:
            continue
        scores = {
            engine: score
            for engine, score in row.get("development_scores", {}).items()
            if engine in wins and math.isfinite(float(score))
        }
        if not scores:
            continue
        best = min(scores.values()) if task == "regression" else max(scores.values())
        tied = []
        for engine, score in scores.items():
            if task == "regression":
                close = abs(score - best) <= max(
                    abs(best) * REGRESSION_RELATIVE_TIE_TOLERANCE, 1e-12
                )
            else:
                close = abs(score - best) <= CLASSIFICATION_TIE_TOLERANCE
            if close:
                tied.append(engine)
        if tied:
            wins[min(tied, key=ENGINE_TIE_ORDER.index)] += 1
    selected = min(
        heldout_tfms,
        key=lambda engine: (-wins.get(engine, 0), ENGINE_TIE_ORDER.index(engine)),
    )
    return selected, wins


def _load_tabicl_checkpoints(
    active_tasks: set[str], output_dir: Path, versions: dict[str, str | None]
) -> tuple[dict[str, str], dict[str, Any]]:
    if not versions.get("tabicl"):
        return {}, {"status": "package_missing"}
    from huggingface_hub import hf_hub_download

    paths: dict[str, str] = {}
    provenance: dict[str, Any] = {
        "repository": TABICL_REPO,
        "revision": TABICL_REVISION,
        "license": "BSD-3-Clause",
        "license_url": "https://huggingface.co/jingang/TabICL",
        "package_version": versions.get("tabicl"),
        "checkpoints": {},
    }
    for task in sorted(active_tasks):
        filename = TABICL_FILES[task]
        resolved = Path(
            hf_hub_download(
                repo_id=TABICL_REPO,
                filename=filename,
                revision=TABICL_REVISION,
            )
        )
        checksum = _sha256(resolved)
        expected = (
            TABICL_CLASSIFICATION_SHA256 if task == "classification" else None
        )
        if expected and checksum.lower() != expected.lower():
            raise ValueError(
                f"TabICL {task} checkpoint checksum mismatch: {checksum}"
            )
        paths[task] = str(resolved)
        provenance["checkpoints"][task] = {
            "filename": filename,
            "revision": TABICL_REVISION,
            "sha256": checksum,
            "expected_sha256": expected,
            "sha256_verified": bool(expected),
        }
    provenance["status"] = "ready"
    return paths, provenance


def run_module07(
    output_dir: str | Path,
    *,
    run_mode: str = "pilot",
    auto_gluon_seconds: int | None = None,
    seeds: list[int] | None = None,
) -> dict[str, Any]:
    """Run grouped leave-one-dataset-out development and outer evaluation."""
    if run_mode not in {"pilot", "full"}:
        raise ValueError("run_mode must be 'pilot' or 'full'.")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest = _load_manifest()
    versions = _versions()
    gpu_available = bool(torch.cuda.is_available())
    if not gpu_available:
        raise RuntimeError("CUDA is required by this Kaggle GPU benchmark.")
    actual_seeds = seeds if seeds is not None else ([17] if run_mode == "pilot" else [17, 42, 2026])
    ag_seconds = auto_gluon_seconds or (120 if run_mode == "pilot" else 900)

    tabpfn_status = "package_missing"
    tabpfn_ready = False
    if versions.get("tabpfn"):
        try:
            from kaggle_secrets import UserSecretsClient

            token = UserSecretsClient().get_secret("TABPFN_TOKEN")
            if token:
                os.environ["TABPFN_TOKEN"] = token
                tabpfn_status = "secret_configured_value_not_logged"
                tabpfn_ready = True
            else:
                tabpfn_status = "secret_empty"
        except Exception as exc:
            tabpfn_status = f"secret_unavailable_{type(exc).__name__}"
    tabicl_paths, tabicl_provenance = _load_tabicl_checkpoints(
        {row["task"] for row in manifest}, output, versions
    )
    tabicl_ready = bool(tabicl_paths)

    protocol = {
        "protocol_version": "module07-v1.0",
        "claim_under_test": "Two-stage predictive-quality routing: Stage 1 selects GBDT or TFM family; Stage 2 selects an eligible TFM when Stage 1 chooses TFM.",
        "stage_1_objective": "Select GBDT versus TFM family from pre-fit task profile using development-only paired validation outcomes; within-tolerance ties prefer GBDT.",
        "stage_2_objective": "Select the TFM with the most development-task validation wins, conditional on TFM family selection and held-out eligibility.",
        "stage_1_primary_claim_limit": "Predictive-quality family routing for the benchmark task mix; this experiment does not measure user requirement understanding, explainability quality, or universal speed superiority.",
        "primary_metrics": {
            "classification": CLASSIFICATION_PRIMARY,
            "regression": REGRESSION_PRIMARY,
        },
        "secondary_metrics": {"classification": ["accuracy"], "regression": ["mae", "r2"]},
        "tie_tolerances": {
            "classification_macro_f1_absolute": CLASSIFICATION_TIE_TOLERANCE,
            "regression_rmse_relative": REGRESSION_RELATIVE_TIE_TOLERANCE,
            "tie_order": ENGINE_TIE_ORDER,
        },
        "outer_evaluation": "leave-one-dataset-out; all seeds from a dataset remain in the same outer task group",
        "inner_development": "one shared seeded split within each outer training partition; winner labels from validation only",
        "outer_test_size": OUTER_TEST_SIZE,
        "inner_validation_size_of_outer_train": INNER_VALIDATION_SIZE,
        "model_candidates": ENGINES,
        "tabpfn_checkpoint_policy": {
            "package_version": versions.get("tabpfn"),
            "model_version": "TabPFN-3",
            "files": TABPFN_FILES,
            "model_license": "Prior Labs TabPFN-3 non-commercial license; account acceptance is required before fit",
            "license_url": "https://github.com/PriorLabs/TabPFN",
        },
        "runtime_is_secondary_outcome": True,
        "auto_gluon_fit_budget_seconds": ag_seconds,
        "seeds": actual_seeds,
        "run_mode": run_mode,
        "stage_1_selector": "DecisionTreeClassifier(max_depth=2,min_samples_leaf=2,class_weight='balanced'); predicts family labels gbdt/tfm.",
        "stage_2_selector": "Development-only TFM winner count within task type, excluding held-out dataset; scale/eligibility gates applied before selection.",
        "selector_uncertainty_policy": "Stage 1 falls back to GBDT if family labels lack diversity, profile is outside development support, or decision-tree leaf share is below 0.55; confidence is heuristic, not calibrated probability.",
        "resource_policy": {"max_gpu_rows": MAX_GPU_ROWS, "max_features": MAX_FEATURES},
        "dataset_groups": len(manifest),
        "dataset_manifest_sha256": _sha256(MANIFEST_PATH),
        "dataset_license_policy": "CC BY 4.0 entries independently reviewed; UCI metadata, targets, and citations retained in provenance.",
        "versions": versions,
        "gpu_available": gpu_available,
        "tabpfn_readiness": tabpfn_status,
        "tabicl_checkpoint_provenance": tabicl_provenance,
        "known_limit": "One inner validation split per dataset and seed makes this an initial exploratory meta-evaluation; seed count is not task count.",
    }
    _write_json(output / "protocol.json", protocol)
    # Preserve source bytes so the archived manifest hash matches protocol.json,
    # including line endings on Windows-generated source bundles.
    (output / "dataset_manifest.json").write_bytes(MANIFEST_PATH.read_bytes())
    _write_json(
        output / "environment.json",
        {
            "python": sys.version,
            "platform": platform.platform(),
            "versions": versions,
            "gpu": {
                "cuda_available": gpu_available,
                "torch_cuda_version": torch.version.cuda,
                "devices": [
                    {
                        "index": i,
                        "name": torch.cuda.get_device_properties(i).name,
                        "total_memory_bytes": torch.cuda.get_device_properties(i).total_memory,
                    }
                    for i in range(torch.cuda.device_count())
                ],
            },
            "tabpfn_status": tabpfn_status,
            "tabicl_checkpoint_provenance": tabicl_provenance,
        },
    )

    loaded: dict[str, tuple[pd.DataFrame, str, dict[str, Any]]] = {}
    load_attempts: list[dict[str, Any]] = []
    for load_index, spec in enumerate(manifest, start=1):
        print(f"[load {load_index}/{len(manifest)}] {spec['name']}", flush=True)
        try:
            frame, target, provenance = _load_dataset(spec)
            loaded[spec["name"]] = (frame, target, provenance)
            print(f"[load {load_index}/{len(manifest)}] {spec['name']}: success ({len(frame)} rows)", flush=True)
            load_attempts.append(
                {"dataset": spec["name"], "status": "success", **provenance}
            )
        except Exception as exc:
            print(f"[load {load_index}/{len(manifest)}] {spec['name']}: FAILED ({type(exc).__name__}: {exc})", flush=True)
            load_attempts.append(
                {
                    "dataset": spec["name"],
                    "status": "dataset_failure",
                    "failure_type": type(exc).__name__,
                    "failure_message": str(exc)[:1000],
                    "traceback": traceback.format_exc()[-2500:],
                }
            )
    pd.DataFrame(load_attempts).to_json(
        output / "dataset_load_status.jsonl", orient="records", lines=True
    )
    if len(loaded) != len(manifest):
        failed = [row["dataset"] for row in load_attempts if row["status"] != "success"]
        raise RuntimeError(
            f"Only {len(loaded)} of {len(manifest)} task groups loaded; all 20 are required. "
            f"Failed datasets: {failed}. Inspect dataset_load_status.jsonl, fix the source issue, and rerun. "
            "No model fits were started."
        )

    development_rows: list[dict[str, Any]] = []
    raw_records: list[dict[str, Any]] = []
    profile_records: list[dict[str, Any]] = []
    route_records: list[dict[str, Any]] = []
    evaluation_records: list[dict[str, Any]] = []
    stage1_label_records: list[dict[str, Any]] = []

    # Each held-out group gets its own selector trained from development
    # validation outcomes on all other groups. Dataset seeds stay grouped.
    for seed in actual_seeds:
        print(f"[seed {seed}] development validation starting", flush=True)
        per_dataset: dict[str, dict[str, Any]] = {}
        for spec in manifest:
            name = spec["name"]
            if name not in loaded:
                continue
            frame, target, provenance = loaded[name]
            task = spec["task"]
            outer_train, outer_test, outer_split_strategy = _safe_split(
                frame, target, task, OUTER_TEST_SIZE, seed
            )
            inner_train, inner_valid, inner_split_strategy = _safe_split(
                outer_train, target, task, INNER_VALIDATION_SIZE, seed + 10_000
            )
            X_outer_train = outer_train.drop(columns=[target])
            X_outer_test = outer_test.drop(columns=[target])
            y_outer_train = outer_train[target]
            y_outer_test = outer_test[target]
            X_inner_train = inner_train.drop(columns=[target])
            X_inner_valid = inner_valid.drop(columns=[target])
            y_inner_train = inner_train[target]
            y_inner_valid = inner_valid[target]
            profile = _profile(outer_train, target, task)
            eligible = _eligibility(
                task,
                len(outer_train),
                X_outer_train.shape[1],
                int(y_outer_train.nunique()) if task == "classification" else 0,
                gpu_available,
                tabpfn_ready,
                tabicl_ready,
            )
            features = _profile_to_features(profile, eligible)
            profile_records.append(
                {
                    "dataset": name,
                    "task": task,
                    "seed": seed,
                    "profile_source": "outer_training_partition_only",
                    "outer_split_strategy": outer_split_strategy,
                    "inner_split_strategy": inner_split_strategy,
                    **profile,
                    "eligible_engines": [
                        engine for engine, status in eligible.items() if status["eligible"]
                    ],
                }
            )
            inner_records: list[dict[str, Any]] = []
            for engine in ENGINES:
                status = eligible.get(engine, {"eligible": False, "reason": "not listed"})
                base = {
                    "dataset": name,
                    "task": task,
                    "seed": seed,
                    "phase": "development_validation",
                    "target": target,
                    "engine": engine,
                    "eligible": bool(status["eligible"]),
                    "eligibility_reason": status["reason"],
                    "provenance": provenance,
                }
                if not status["eligible"]:
                    result = {"engine": engine, "status": "not_eligible", "reason": status["reason"]}
                else:
                    result = _fit_score(
                        engine,
                        X_inner_train,
                        y_inner_train,
                        X_inner_valid,
                        y_inner_valid,
                        task,
                        target,
                        seed,
                        output / "models" / name / str(seed) / "development" / engine,
                        tabicl_paths.get(task),
                        ag_seconds,
                    )
                row = {**base, **result}
                raw_records.append(row)
                inner_records.append(row)
                print(f"[fit {len(raw_records)}] {name} seed={seed} development {engine}: {row.get('status')}", flush=True)
                pd.DataFrame(raw_records).to_json(
                    output / "raw_attempts.jsonl", orient="records", lines=True
                )
            winner, tied, successful = _winner(task, eligible, inner_records)
            dev_row = {
                "dataset": name,
                "task": task,
                "seed": seed,
                "features": features,
                "winner": winner,
                "tied_engines": tied,
                "primary_metric": _primary(task),
                "development_scores": successful,
            }
            per_dataset[name] = {
                "spec": spec,
                "frame": frame,
                "target": target,
                "provenance": provenance,
                "outer_train": outer_train,
                "outer_test": outer_test,
                "X_outer_train": X_outer_train,
                "X_outer_test": X_outer_test,
                "y_outer_train": y_outer_train,
                "y_outer_test": y_outer_test,
                "profile": profile,
                "outer_split_strategy": outer_split_strategy,
                "inner_split_strategy": inner_split_strategy,
                "features": features,
                "eligible": eligible,
                "development": dev_row,
            }
            development_rows.append(dev_row)

        print(f"[seed {seed}] outer evaluation starting", flush=True)
        for heldout_name, item in per_dataset.items():
            development = [row for row in development_rows if row["seed"] == seed and row["dataset"] != heldout_name]
            stage1_development = _stage1_development_rows(
                development, item["spec"]["task"]
            )
            stage1_label_records.extend(
                {
                    "outer_heldout_dataset": heldout_name,
                    "seed": seed,
                    "task": row["task"],
                    "development_dataset": row["dataset"],
                    "family_winner": row["family_winner"],
                    "tfm_engine_for_label": row["stage1_tfm_engine_for_label"],
                    "label_reason": row["family_label_reason"],
                    "features": row["features"],
                }
                for row in stage1_development
            )
            selected_family, selector_info, selector = _select_family(
                stage1_development, item["features"]
            )
            selector_info.setdefault("predicted_family", selected_family)
            eligible = item["eligible"]
            tfm_development_engine, tfm_development_wins = _development_tfm_choice(
                development, item["spec"]["task"], eligible
            )
            if selected_family == "tfm" and tfm_development_engine == "autogluon_cpu":
                # Stage 2 has no eligible, development-selected TFM. Preserve
                # the family guardrail and route to the GBDT baseline.
                selected_engine = "autogluon_cpu"
                selected_family = "gbdt"
                selection_reason = "no_eligible_TFM_for_stage2"
            else:
                selected_engine = (
                    tfm_development_engine
                    if selected_family == "tfm"
                    else "autogluon_cpu"
                )
                selection_reason = selector_info.get(
                    "selection_reason",
                    selector_info.get("reason", "selector_prediction_eligible"),
                )
            route_records.append(
                {
                    "dataset": heldout_name,
                    "task": item["spec"]["task"],
                    "seed": seed,
                    "selected_family": selected_family,
                    "selected_engine": selected_engine,
                    "selector_predicted_family_before_eligibility_guard": selector_info[
                        "predicted_family"
                    ],
                    "selection_reason": selection_reason,
                    "selector_method": selector_info["method"],
                    "selector_confidence": selector_info.get("confidence"),
                    "selector_confidence_calibrated": selector_info.get(
                        "confidence_is_calibrated_probability", False
                    ),
                    "selector_support_status": selector_info.get("support_status"),
                    "selector_metadata": selector_info,
                    "development_tfm_only_engine": tfm_development_engine,
                    "development_tfm_winner_counts": tfm_development_wins,
                    "development_task_count": selector_info["development_task_count"],
                    "development_family_label_counts": selector_info["winner_label_counts"],
                    "stage1_development_family_labels": [
                        {
                            "dataset": row["dataset"],
                            "family_winner": row["family_winner"],
                            "tfm_engine_for_label": row["stage1_tfm_engine_for_label"],
                            "label_reason": row["family_label_reason"],
                        }
                        for row in stage1_development
                    ],
                    "stage2_development_tfm_winner_counts": tfm_development_wins,
                    "development_datasets_excluded_from_selector": [heldout_name],
                    "route_profile_source": "outer_training_partition_only",
                    "profile": item["profile"],
                    "outer_split_strategy": item["outer_split_strategy"],
                    "inner_split_strategy": item["inner_split_strategy"],
                    "eligible_engines": eligible,
                }
            )
            Xtr = item["X_outer_train"]
            Xte = item["X_outer_test"]
            ytr = item["y_outer_train"]
            yte = item["y_outer_test"]
            task = item["spec"]["task"]
            target = item["target"]
            outer_records: dict[str, dict[str, Any]] = {}
            for engine in ENGINES:
                status = eligible.get(engine, {"eligible": False, "reason": "not listed"})
                base = {
                    "dataset": heldout_name,
                    "task": task,
                    "seed": seed,
                    "phase": "outer_test_evaluation",
                    "target": target,
                    "engine": engine,
                    "eligible": bool(status["eligible"]),
                    "eligibility_reason": status["reason"],
                    "provenance": item["provenance"],
                }
                if status["eligible"]:
                    result = _fit_score(
                        engine,
                        Xtr,
                        ytr,
                        Xte,
                        yte,
                        task,
                        target,
                        seed,
                        output / "models" / heldout_name / str(seed) / "outer" / engine,
                        tabicl_paths.get(task),
                        ag_seconds,
                    )
                else:
                    result = {
                        "engine": engine,
                        "status": "not_eligible",
                        "reason": status["reason"],
                    }
                record = {**base, **result}
                raw_records.append(record)
                outer_records[engine] = record
                print(f"[fit {len(raw_records)}] {heldout_name} seed={seed} outer {engine}: {record.get('status')}", flush=True)
                pd.DataFrame(raw_records).to_json(
                    output / "raw_attempts.jsonl", orient="records", lines=True
                )
            observed_winner, observed_ties, outer_scores = _winner(
                task,
                eligible,
                list(outer_records.values()),
            )
            selected_row = outer_records.get(selected_engine, {})
            selected_success = selected_row.get("status") == "success"
            selected_metric = (
                selected_row.get("metrics", {}).get(_primary(task))
                if selected_success
                else None
            )
            best_score = (
                min(outer_scores.values())
                if task == "regression" and outer_scores
                else max(outer_scores.values()) if outer_scores else None
            )
            if selected_metric is None or best_score is None:
                normalized_regret = None
            elif task == "regression":
                normalized_regret = max(0.0, (selected_metric - best_score) / max(abs(best_score), 1e-12))
            else:
                normalized_regret = max(0.0, (best_score - selected_metric) / max(abs(best_score), 0.01))
            selected_matches_best_or_tie = (
                bool(selected_engine in observed_ties) if selected_success and observed_ties else None
            )
            evaluation_records.append(
                {
                    "dataset": heldout_name,
                    "task": task,
                    "seed": seed,
                    "selected_family": selected_family,
                    "primary_metric": _primary(task),
                    "selected_engine": selected_engine,
                    "selected_status": selected_row.get("status", "not_run"),
                    "selected_primary_score": selected_metric,
                    "selected_route_normalized_regret_if_success": normalized_regret,
                    "autogluon_status": outer_records.get("autogluon_cpu", {}).get("status", "not_eligible"),
                    "autogluon_primary_score": outer_records.get("autogluon_cpu", {}).get("metrics", {}).get(_primary(task)),
                    "development_tfm_only_engine": tfm_development_engine,
                    "development_tfm_only_status": outer_records.get(tfm_development_engine, {}).get("status", "not_eligible"),
                    "development_tfm_only_primary_score": outer_records.get(tfm_development_engine, {}).get("metrics", {}).get(_primary(task)),
                    "development_tfm_winner_counts": tfm_development_wins,
                    "outer_observed_best_engine_descriptive_only": observed_winner,
                    "outer_observed_ties": observed_ties,
                    "selected_matches_outer_test_best_or_tie_descriptive_only": selected_matches_best_or_tie,
                    "outer_successful_scores": outer_scores,
                    "selector_metadata": selector_info,
                }
            )
            pd.DataFrame(route_records).to_json(
                output / "router_decisions.jsonl", orient="records", lines=True
            )
            pd.DataFrame(evaluation_records).to_json(
                output / "outer_evaluation.jsonl", orient="records", lines=True
            )
        pd.DataFrame(development_rows).to_json(
            output / "development_winners.jsonl", orient="records", lines=True
        )
        pd.DataFrame(profile_records).to_json(
            output / "dataset_profiles.jsonl", orient="records", lines=True
        )
        pd.DataFrame(stage1_label_records).to_json(
            output / "stage1_development_labels.jsonl", orient="records", lines=True
        )

    raw_df = pd.DataFrame(raw_records)
    evaluation_df = pd.DataFrame(evaluation_records)
    routes_df = pd.DataFrame(route_records)
    profiles_df = pd.DataFrame(profile_records)
    raw_df.to_csv(output / "raw_attempts.csv", index=False)
    evaluation_df.to_csv(output / "outer_evaluation.csv", index=False)
    routes_df.to_csv(output / "router_decisions.csv", index=False)
    profiles_df.to_csv(output / "dataset_profiles.csv", index=False)
    pd.DataFrame(development_rows).to_csv(output / "development_winners.csv", index=False)
    pd.DataFrame(stage1_label_records).to_csv(
        output / "stage1_development_labels.csv", index=False
    )

    # Summaries first average seeds within each dataset. Bootstrap resampling
    # therefore treats independent dataset/task groups, not seeds, as units.
    summary_rows: list[dict[str, Any]] = []
    for (dataset, task, engine), group in raw_df[
        (raw_df.status == "success") & (raw_df.phase == "outer_test_evaluation")
    ].groupby(["dataset", "task", "engine"]):
        metric = _primary(task)
        values = [row.get(metric) for row in group.metrics if isinstance(row, dict)]
        values = [float(value) for value in values if value is not None and math.isfinite(float(value))]
        if values:
            summary_rows.append(
                {
                    "dataset": dataset,
                    "task": task,
                    "engine": engine,
                    "primary_metric": metric,
                    "n_seeds": len(values),
                    "mean_primary_score": float(np.mean(values)),
                    "sd_across_seeds": float(np.std(values, ddof=1)) if len(values) > 1 else None,
                }
            )
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output / "candidate_summary_by_dataset.csv", index=False)

    group_eval = evaluation_df.groupby(["dataset", "task"], as_index=False).agg(
        selected_completion_rate=("selected_status", lambda values: float(np.mean(np.asarray(values) == "success"))),
        mean_selected_normalized_regret=("selected_route_normalized_regret_if_success", "mean"),
        selected_best_or_tie_rate=("selected_matches_outer_test_best_or_tie_descriptive_only", "mean"),
        selected_family=("selected_family", lambda values: values.iloc[0]),
        selected_engine=("selected_engine", lambda values: values.iloc[0]),
    )
    group_eval.to_csv(output / "selector_summary_by_dataset.csv", index=False)
    task_metric_rows: list[dict[str, Any]] = []
    task_rng = np.random.default_rng(2026)
    for task, task_groups in group_eval.groupby("task"):
        for metric_name in [
            "selected_completion_rate",
            "mean_selected_normalized_regret",
            "selected_best_or_tie_rate",
        ]:
            values = task_groups[metric_name].dropna().to_numpy(dtype=float)
            if len(values) == 0:
                continue
            draws = np.asarray(
                [
                    task_rng.choice(values, size=len(values), replace=True).mean()
                    for _ in range(5000)
                ]
            )
            task_metric_rows.append(
                {
                    "task": task,
                    "measure": metric_name,
                    "n_independent_dataset_groups": len(values),
                    "mean": float(values.mean()),
                    "bootstrap_95_ci_low": float(np.quantile(draws, 0.025)),
                    "bootstrap_95_ci_high": float(np.quantile(draws, 0.975)),
                    "unit": "dataset group after seed aggregation",
                    "interpretation": (
                        "descriptive comparison to the outer-test oracle"
                        if metric_name == "selected_best_or_tie_rate"
                        else "exploratory held-out dataset estimate"
                    ),
                }
            )
    pd.DataFrame(task_metric_rows).to_csv(output / "task_level_metrics_summary.csv", index=False)
    routes_df.groupby(["task", "selected_engine"], dropna=False).agg(
        heldout_dataset_seed_count=("dataset", "size"),
        independent_dataset_group_count=("dataset", "nunique"),
    ).reset_index().to_csv(output / "route_counts.csv", index=False)
    routes_df.groupby(["task", "selected_family"], dropna=False).agg(
        heldout_dataset_seed_count=("dataset", "size"),
        independent_dataset_group_count=("dataset", "nunique"),
    ).reset_index().to_csv(output / "route_family_counts.csv", index=False)

    comparison_rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(81)
    for task, group in group_eval.groupby("task"):
        metric = _primary(task)
        task_evaluations = evaluation_df[evaluation_df.task == task].copy()
        task_evaluations["selected_score"] = task_evaluations.apply(
            lambda row: row["selected_primary_score"], axis=1
        )
        for baseline_name, score_column, status_column in [
            ("AutoGluon-only", "autogluon_primary_score", "autogluon_status"),
            ("development-selected-TFM-only", "development_tfm_only_primary_score", "development_tfm_only_status"),
        ]:
            means = []
            paired = []
            for dataset, rows in task_evaluations.groupby("dataset"):
                valid = rows[
                    rows["selected_score"].notna()
                    & (rows[status_column] == "success")
                    & rows[score_column].notna()
                ]
                if not valid.empty:
                    selected_value = float(valid["selected_score"].mean())
                    baseline_value = float(valid[score_column].mean())
                    difference = (
                        (baseline_value - selected_value)
                        / max(abs(baseline_value), 1e-12)
                        if task == "regression"
                        else selected_value - baseline_value
                    )
                    paired.append((dataset, difference))
                    means.append(difference)
            if not paired:
                continue
            values = np.asarray([value for _, value in paired], dtype=float)
            boot = np.asarray(
                [rng.choice(values, size=len(values), replace=True).mean() for _ in range(5000)]
            )
            comparison_rows.append(
                {
                    "task": task,
                    "primary_metric": metric,
                    "selector_vs": baseline_name,
                    "n_independent_dataset_groups": len(values),
                    "mean_selector_improvement": float(values.mean()),
                    "bootstrap_95_ci_low": float(np.quantile(boot, 0.025)),
                    "bootstrap_95_ci_high": float(np.quantile(boot, 0.975)),
                    "comparison_scale": "relative RMSE improvement for regression; macro-F1 points for classification",
                    "sign_convention": "positive favors selector; dataset-level pairs after seed aggregation",
                    "interpretation": "exploratory grouped leave-one-dataset-out estimate",
                }
            )
    pd.DataFrame(comparison_rows).to_csv(output / "grouped_comparisons.csv", index=False)

    status = {
        "run_completed_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "run_mode": run_mode,
        "dataset_manifest_task_groups": len(manifest),
        "loaded_task_groups": len(loaded),
        "evaluated_task_groups": int(evaluation_df.dataset.nunique()) if not evaluation_df.empty else 0,
        "seeds": actual_seeds,
        "outer_evaluation_count": int(len(evaluation_df)),
        "candidate_attempt_count": int(len(raw_df)),
        "candidate_success_count": int((raw_df.status == "success").sum()),
        "candidate_failure_or_exclusion_count": int((raw_df.status != "success").sum()),
        "selected_route_success_count": int((evaluation_df.selected_status == "success").sum()) if not evaluation_df.empty else 0,
        "selected_route_completion_rate": float(np.mean(evaluation_df.selected_status == "success")) if not evaluation_df.empty else None,
        "route_engine_counts": routes_df.selected_engine.value_counts().to_dict() if not routes_df.empty else {},
        "route_family_counts": routes_df.selected_family.value_counts().to_dict() if not routes_df.empty else {},
        "tabpfn_status": tabpfn_status,
        "tabicl_status": tabicl_provenance.get("status"),
        "note": "Scores and route choices on held-out outer test partitions were not used to fit the corresponding selector.",
    }
    _write_json(output / "run_status.json", status)
    return status
