from datetime import datetime
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

from sqlalchemy import JSON, CheckConstraint, DateTime, String, Uuid, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.app.models.base import Base

if TYPE_CHECKING:
    from backend.app.models.artifact import Artifact
    from backend.app.models.message import Message


class Run(Base):
    __tablename__ = "runs"
    __table_args__ = (
        CheckConstraint(
            "status IN ('queued', 'running', 'succeeded', 'failed')",
            name="status_valid",
        ),
    )

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    dataset_pointer: Mapped[str | None] = mapped_column(String(2048))
    raw_requirements_text: Mapped[str | None] = mapped_column(String)
    parsed_requirements: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    profile: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    preprocessing: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    training_config: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    leaderboard: Mapped[list[dict[str, Any]] | None] = mapped_column(JSON)
    error_stage: Mapped[str | None] = mapped_column(String(64))
    error_message: Mapped[str | None] = mapped_column(String)
    route: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    status: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
        default="queued",
        server_default="queued",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    artifacts: Mapped[list["Artifact"]] = relationship(
        back_populates="run",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    messages: Mapped[list["Message"]] = relationship(
        back_populates="run",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
