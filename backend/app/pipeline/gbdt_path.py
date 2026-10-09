import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor

from backend.app.pipeline.preprocessing import PreparedDataset

logger = logging.getLogger(__name__)

TaskType = Literal["classification", "regression"]
AutoGluonProblemType = Literal["binary", "multiclass", "regression"]

SUPPORTED_METRICS: dict[AutoGluonProblemType, set[str]] = {
    "binary": {"accuracy", "f1", "roc_auc", "log_loss"},
    "multiclass": {"accuracy", "log_loss"},
    "regression": {
        "root_mean_squared_error",
        "mean_absolute_error",
        "r2",
    },
}
DEFAULT_METRIC: dict[TaskType, str] = {
    "classification": "accuracy",
    "regression": "root_mean_squared_error",
}


class TrainingError(RuntimeError):
    """Raised when an AutoGluon training run cannot be completed."""


@dataclass(frozen=True)
class TrainingResult:
    model_path: Path
    leaderboard_path: Path
    leaderboard: list[dict[str, Any]]
    best_model: str
    evaluation_metric: str


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if pd.isna(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _training_frame(
    X: pd.DataFrame,
    y: pd.Series,
    target_column: str,
) -> pd.DataFrame:
    frame = X.copy()
    frame[target_column] = y
    return frame


def train_autogluon(
    *,
    prepared: PreparedDataset,
    target_column: str,
    task_type: TaskType,
    output_directory: Path,
    time_limit_seconds: int,
    evaluation_metric: str | None = None,
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

    model_path = output_directory / "model"
    leaderboard_path = output_directory / "leaderboard.csv"
    output_directory.mkdir(parents=True, exist_ok=True)
    train_data = _training_frame(
        prepared.X_train_raw,
        prepared.y_train,
        target_column,
    )
    validation_data = _training_frame(
        prepared.X_validation_raw,
        prepared.y_validation,
        target_column,
    )

    try:
        predictor = TabularPredictor(
            label=target_column,
            problem_type=problem_type,
            eval_metric=metric,
            path=str(model_path),
            verbosity=2,
        )
        predictor.fit(
            train_data=train_data,
            tuning_data=validation_data,
            time_limit=time_limit_seconds,
            presets="good_quality",
            num_gpus=0,
            dynamic_stacking=False,
            use_bag_holdout=True,
            fit_weighted_ensemble=False,
        )
        leaderboard_frame = predictor.leaderboard(
            data=validation_data,
            silent=True,
        )
        if leaderboard_frame.empty:
            raise TrainingError("AutoGluon completed without leaderboard results.")

        leaderboard_path.parent.mkdir(parents=True, exist_ok=True)
        leaderboard_frame.to_csv(leaderboard_path, index=False)
        leaderboard = [
            _json_safe(row) for row in leaderboard_frame.to_dict(orient="records")
        ]
        return TrainingResult(
            model_path=model_path,
            leaderboard_path=leaderboard_path,
            leaderboard=leaderboard,
            best_model=str(leaderboard_frame.iloc[0]["model"]),
            evaluation_metric=metric,
        )
    except TrainingError:
        raise
    except Exception as exc:
        logger.exception("AutoGluon training failed")
        raise TrainingError(
            f"AutoGluon training failed ({type(exc).__name__})."
        ) from exc
