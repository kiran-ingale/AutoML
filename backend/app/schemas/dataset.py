from pydantic import BaseModel, ConfigDict, Field


class DatasetColumnProfile(BaseModel):
    name: str
    dtype: str
    missing_count: int = Field(ge=0)
    missing_percent: float = Field(ge=0, le=100)
    cardinality: int = Field(ge=0)


class DatasetProfile(BaseModel):
    rows: int = Field(gt=0)
    columns_count: int = Field(gt=0)
    columns: list[DatasetColumnProfile]
    duplicate_rows: int = Field(ge=0)
    target_column_guess: str
    target_guess_method: str
    class_imbalance_ratio: float | None = Field(default=None, ge=1)
    warnings: list[str]

    model_config = ConfigDict(extra="forbid")
