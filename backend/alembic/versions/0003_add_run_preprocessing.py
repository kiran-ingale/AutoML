"""Store preprocessing summary on each run.

Revision ID: 0003_run_preprocessing
Revises: 0002_status_constraint
Create Date: 2026-10-09
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0003_run_preprocessing"
down_revision: str | None = "0002_status_constraint"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("runs", sa.Column("preprocessing", sa.JSON(), nullable=True))


def downgrade() -> None:
    op.drop_column("runs", "preprocessing")
