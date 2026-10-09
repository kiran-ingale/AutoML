"""Add persisted training configuration, leaderboard, and failure details.

Revision ID: 0004_training_state
Revises: 0003_run_preprocessing
Create Date: 2026-10-09
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0004_training_state"
down_revision: str | None = "0003_run_preprocessing"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("runs", sa.Column("training_config", sa.JSON(), nullable=True))
    op.add_column("runs", sa.Column("leaderboard", sa.JSON(), nullable=True))
    op.add_column("runs", sa.Column("error_stage", sa.String(length=64), nullable=True))
    op.add_column("runs", sa.Column("error_message", sa.String(), nullable=True))


def downgrade() -> None:
    op.drop_column("runs", "error_message")
    op.drop_column("runs", "error_stage")
    op.drop_column("runs", "leaderboard")
    op.drop_column("runs", "training_config")
