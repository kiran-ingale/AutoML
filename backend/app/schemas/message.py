from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict


class MessageRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    run_id: UUID
    role: str
    content: str
    created_at: datetime
