from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class PreprocessingSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target_column: str
    task_type: Literal["classification", "regression"]
    split_method: str
    test_size: float = Field(gt=0, lt=1)
    random_state: int
    original_rows: int = Field(gt=0)
    rows_without_target: int = Field(ge=0)
    training_rows: int = Field(gt=0)
    validation_rows: int = Field(gt=0)
    input_feature_count: int = Field(gt=0)
    output_feature_count: int = Field(gt=0)
    numeric_columns: list[str]
    categorical_columns: list[str]
    dropped_constant_columns: list[str]
    missing_feature_values_before_imputation: int = Field(ge=0)
    warnings: list[str]
