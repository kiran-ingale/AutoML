import hashlib
import logging
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)

from backend.app.pipeline.gbdt_path import (
    DEFAULT_METRIC,
    SUPPORTED_METRICS,
    AutoGluonProblemType,
    TaskType,
    TrainingError,
    TrainingResult,
)
from backend.app.pipeline.preprocessing import PreparedDataset

logger = logging.getLogger(__name__)

TABICL_REPO = "jingang/TabICL"
TABICL_REVISION = "4dcd344ece2c00be9e831fdd35bed57b5ad83e19"
TABICL_CLASSIFICATION_CHECKPOINT = "tabicl-classifier-v2-20260212.ckpt"
TABICL_REGRESSION_CHECKPOINT = "tabicl-regressor-v2-20260212.ckpt"
TABICL_CLASSIFICATION_SHA256 = (
    "bdc7dbd5e4ff21f8f0456fcf90c6b7cdf72dbea960f2d05b19bec19f9b3d4ed0"
)


def _metric_score(
    *,
    metric: str,
    task_type: TaskType,
    y_true: pd.Series,
    predictions: np.ndarray,
    probabilities: np.ndarray | None,
    classes: np.ndarray | None,
) -> float:
    if task_type == "regression":
        if metric == "root_mean_squared_error":
            return -float(np.sqrt(mean_squared_error(y_true, predictions)))
        if metric == "mean_absolute_error":
            return -float(mean_absolute_error(y_true, predictions))
        return float(r2_score(y_true, predictions))

    if metric == "accuracy":
        return float(accuracy_score(y_true, predictions))
    if metric == "f1":
        average = "binary" if classes is not None and len(classes) == 2 else "macro"
        options: dict[str, Any] = {"average": average}
        if average == "binary" and classes is not None:
            options["pos_label"] = classes[1]
        return float(f1_score(y_true, predictions, **options))
    if probabilities is None or classes is None:
        raise TrainingError(
            f"Metric {metric!r} requires class probabilities from the TFM."
        )
    if metric == "log_loss":
        return -float(log_loss(y_true, probabilities, labels=classes))
    if len(classes) != 2:
        raise TrainingError("ROC AUC is supported only for binary classification.")
    return float(roc_auc_score(y_true, probabilities[:, 1]))


def train_tfm(
    *,
    engine: Literal["tabpfn", "tabicl"],
    prepared: PreparedDataset,
    task_type: TaskType,
    output_directory: Path,
    evaluation_metric: str | None = None,
    random_state: int = 42,
    device: Literal["cpu", "cuda"] = "cpu",
) -> TrainingResult:
    metric = evaluation_metric or DEFAULT_METRIC[task_type]
    if task_type == "classification":
        class_count = prepared.y_train.nunique(dropna=True)
        problem_type: AutoGluonProblemType = (
            "binary" if class_count == 2 else "multiclass"
        )
    else:
        problem_type = "regression"
    if metric not in SUPPORTED_METRICS[problem_type]:
        raise TrainingError(
            f"Metric {metric!r} is not supported for {problem_type} classification."
            if problem_type != "regression"
            else f"Metric {metric!r} is not supported for regression."
        )

    try:
        model = _create_tfm(
            engine=engine,
            task_type=task_type,
            random_state=random_state,
            device=device,
        )
        model.fit(prepared.X_train_raw, prepared.y_train)
        predictions = np.asarray(model.predict(prepared.X_validation_raw))
        classes = (
            np.asarray(model.classes_)
            if task_type == "classification" and hasattr(model, "classes_")
            else None
        )
        probabilities = (
            np.asarray(model.predict_proba(prepared.X_validation_raw))
            if task_type == "classification" and metric in {"roc_auc", "log_loss"}
            else None
        )
        score = _metric_score(
            metric=metric,
            task_type=task_type,
            y_true=prepared.y_validation,
            predictions=predictions,
            probabilities=probabilities,
            classes=classes,
        )

        model_path = output_directory / "model"
        leaderboard_path = output_directory / "leaderboard.csv"
        model_path.mkdir(parents=True, exist_ok=True)
        leaderboard_path.parent.mkdir(parents=True, exist_ok=True)
        if engine == "tabpfn":
            model.save_fit_state(model_path / "predictor.tabpfn_fit")
        else:
            model.save(
                model_path / "predictor.pkl",
                save_model_weights=False,
                save_training_data=True,
            )
        leaderboard = [
            {
                "model": type(model).__name__,
                "score_test": score,
                "eval_metric": metric,
            }
        ]
        pd.DataFrame(leaderboard).to_csv(leaderboard_path, index=False)
        return TrainingResult(
            model_path=model_path,
            leaderboard_path=leaderboard_path,
            leaderboard=leaderboard,
            best_model=type(model).__name__,
            evaluation_metric=metric,
        )
    except TrainingError:
        raise
    except Exception as exc:
        logger.exception("%s training failed", engine)
        raise TrainingError(
            f"{engine} training failed ({type(exc).__name__})."
        ) from exc


def _create_tfm(
    *,
    engine: Literal["tabpfn", "tabicl"],
    task_type: TaskType,
    random_state: int,
    device: Literal["cpu", "cuda"],
) -> Any:
    try:
        if engine == "tabpfn":
            if task_type == "classification":
                from tabpfn import TabPFNClassifier

                return TabPFNClassifier(device=device, random_state=random_state)
            from tabpfn import TabPFNRegressor

            return TabPFNRegressor(device=device, random_state=random_state)

        checkpoint_path = _download_tabicl_checkpoint(task_type)
        if task_type == "classification":
            from tabicl import TabICLClassifier

            return TabICLClassifier(
                model_path=str(checkpoint_path),
                device=device,
                random_state=random_state,
                n_jobs=1,
            )
        from tabicl import TabICLRegressor

        return TabICLRegressor(
            model_path=str(checkpoint_path),
            device=device,
            random_state=random_state,
            n_jobs=1,
        )
    except ImportError as exc:
        raise TrainingError(
            f"The selected TFM engine {engine!r} is not installed."
        ) from exc


def _download_tabicl_checkpoint(task_type: TaskType) -> Path:
    """Resolve TabICL weights from a fixed Hub commit and check known hash."""
    try:
        from huggingface_hub import hf_hub_download

        filename = (
            TABICL_CLASSIFICATION_CHECKPOINT
            if task_type == "classification"
            else TABICL_REGRESSION_CHECKPOINT
        )
        path = Path(
            hf_hub_download(
                repo_id=TABICL_REPO,
                filename=filename,
                revision=TABICL_REVISION,
            )
        )
        if task_type == "classification":
            checksum = hashlib.sha256()
            with path.open("rb") as checkpoint_file:
                for block in iter(lambda: checkpoint_file.read(1024 * 1024), b""):
                    checksum.update(block)
            digest = checksum.hexdigest()
            if digest.lower() != TABICL_CLASSIFICATION_SHA256:
                raise TrainingError(
                    "The pinned TabICL classifier checkpoint failed its SHA-256 check."
                )
        return path
    except ImportError as exc:
        raise TrainingError(
            "TabICL checkpoint support requires huggingface_hub."
        ) from exc
