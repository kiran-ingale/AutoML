from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict

from backend.app.schemas.dataset import DatasetProfile
from backend.app.schemas.preprocessing import PreprocessingSummary


class RunRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    dataset_pointer: str | None
    raw_requirements_text: str | None
    parsed_requirements: dict[str, Any] | None
    profile: DatasetProfile | None
    preprocessing: PreprocessingSummary | None
    training_config: dict[str, Any] | None
    leaderboard: list[dict[str, Any]] | None
    error_stage: str | None
    error_message: str | None
    route: dict[str, Any] | None
    status: str
    created_at: datetime
