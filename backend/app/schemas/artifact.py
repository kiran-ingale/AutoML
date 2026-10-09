from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict


class ArtifactRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    run_id: UUID
    type: str
    storage_path: str
    created_at: datetime
