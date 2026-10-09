from collections.abc import Generator
from uuid import uuid4

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from backend.app.models import Artifact, Base, Message, Run
from backend.app.repositories import (
    create_artifact,
    create_message,
    create_run,
    delete_run,
    get_run,
    list_artifacts,
    list_messages,
)
from backend.app.schemas.artifact import ArtifactRead
from backend.app.schemas.message import MessageRead
from backend.app.schemas.run import RunRead


@pytest.fixture
def session() -> Generator[Session, None, None]:
    engine = create_engine("sqlite+pysqlite:///:memory:")

    @event.listens_for(engine, "connect")
    def enable_foreign_keys(connection, record) -> None:
        del record
        cursor = connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine, expire_on_commit=False)
    with session_factory() as db_session:
        yield db_session
    engine.dispose()


def test_run_round_trips_through_repository_and_schema(session: Session) -> None:
    run = create_run(
        session,
        dataset_pointer="storage/example.csv",
        raw_requirements_text="Prefer an interpretable classifier.",
    )
    run.parsed_requirements = {"priority": "explainability"}
    run.profile = {
        "rows": 12,
        "columns_count": 3,
        "columns": [],
        "duplicate_rows": 0,
        "target_column_guess": "target",
        "target_guess_method": "last_column_heuristic_requires_user_confirmation",
        "class_imbalance_ratio": None,
        "warnings": [],
    }
    run.route = {"path": "gbdt", "rationale": "CPU execution"}
    session.commit()

    loaded = get_run(session, run.id)

    assert loaded is not None
    assert RunRead.model_validate(loaded).model_dump(mode="json")["id"] == str(run.id)
    assert loaded.status == "queued"
    assert loaded.profile["rows"] == 12
    assert loaded.profile["columns_count"] == 3


def test_artifacts_and_messages_are_scoped_to_run(session: Session) -> None:
    run = create_run(session)
    other_run = create_run(session)

    artifact = create_artifact(
        session,
        run_id=run.id,
        artifact_type="report",
        storage_path="storage/report.md",
    )
    message = create_message(
        session,
        run_id=run.id,
        role="assistant",
        content="Training has not started yet.",
    )
    create_artifact(
        session,
        run_id=other_run.id,
        artifact_type="plot",
        storage_path="storage/other.png",
    )
    create_message(
        session,
        run_id=other_run.id,
        role="user",
        content="Other run message.",
    )

    assert [item.id for item in list_artifacts(session, run.id)] == [artifact.id]
    assert [item.id for item in list_messages(session, run.id)] == [message.id]
    assert ArtifactRead.model_validate(artifact).run_id == run.id
    assert (
        MessageRead.model_validate(message).content == "Training has not started yet."
    )


def test_deleting_run_cascades_to_children(session: Session) -> None:
    run = create_run(session)
    create_artifact(
        session,
        run_id=run.id,
        artifact_type="model",
        storage_path="storage/model.bin",
    )
    create_message(
        session,
        run_id=run.id,
        role="user",
        content="Start a run.",
    )

    delete_run(session, run)

    assert get_run(session, run.id) is None
    assert session.query(Artifact).filter_by(run_id=run.id).count() == 0
    assert session.query(Message).filter_by(run_id=run.id).count() == 0


def test_run_status_constraint_rejects_unknown_status(session: Session) -> None:
    session.add(Run(id=uuid4(), status="unknown"))

    with pytest.raises(IntegrityError):
        session.commit()
