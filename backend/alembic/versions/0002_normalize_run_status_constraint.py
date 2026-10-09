"""Normalize the run status check constraint name.

Revision ID: 0002_status_constraint
Revises: 0001_core_tables
Create Date: 2026-10-09
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0002_status_constraint"
down_revision: str | None = "0001_core_tables"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

OLD_NAME = "ck_runs_ck_runs_status_valid"
EXPECTED_NAME = "ck_runs_status_valid"


def upgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    constraint_names = {
        constraint["name"] for constraint in inspector.get_check_constraints("runs")
    }
    if OLD_NAME in constraint_names and EXPECTED_NAME not in constraint_names:
        op.execute(
            "ALTER TABLE runs RENAME CONSTRAINT "
            '"ck_runs_ck_runs_status_valid" TO "ck_runs_status_valid"'
        )


def downgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    constraint_names = {
        constraint["name"] for constraint in inspector.get_check_constraints("runs")
    }
    if EXPECTED_NAME in constraint_names and OLD_NAME not in constraint_names:
        op.execute(
            "ALTER TABLE runs RENAME CONSTRAINT "
            '"ck_runs_status_valid" TO "ck_runs_ck_runs_status_valid"'
        )
