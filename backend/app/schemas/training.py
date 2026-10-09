from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

ClassificationMetric = Literal["accuracy", "f1", "roc_auc", "log_loss"]
RegressionMetric = Literal[
    "root_mean_squared_error",
    "mean_absolute_error",
    "r2",
]
EvaluationMetric = ClassificationMetric | RegressionMetric
CLASSIFICATION_METRICS = {"accuracy", "f1", "roc_auc", "log_loss"}
REGRESSION_METRICS = {
    "root_mean_squared_error",
    "mean_absolute_error",
    "r2",
}


class TrainingRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target_column: str = Field(min_length=1, max_length=256)
    task_type: Literal["classification", "regression"]
    time_limit_seconds: int = Field(default=60, ge=1, le=7200)
    test_size: float = Field(default=0.2, gt=0, lt=0.5)
    random_state: int = Field(default=42, ge=0, le=2**32 - 1)
    evaluation_metric: EvaluationMetric | None = None

    @model_validator(mode="after")
    def validate_metric_for_task(self) -> "TrainingRequest":
        if self.evaluation_metric is None:
            return self
        supported_metrics = (
            CLASSIFICATION_METRICS
            if self.task_type == "classification"
            else REGRESSION_METRICS
        )
        if self.evaluation_metric not in supported_metrics:
            raise ValueError(
                f"Metric {self.evaluation_metric!r} is not supported for "
                f"{self.task_type}."
            )
        return self
