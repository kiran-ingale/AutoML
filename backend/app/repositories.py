from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.models import Artifact, Message, Run


def create_run(
    session: Session,
    *,
    dataset_pointer: str | None = None,
    raw_requirements_text: str | None = None,
) -> Run:
    run = Run(
        dataset_pointer=dataset_pointer,
        raw_requirements_text=raw_requirements_text,
    )
    session.add(run)
    session.commit()
    session.refresh(run)
    return run


def get_run(session: Session, run_id: UUID) -> Run | None:
    return session.get(Run, run_id)


def delete_run(session: Session, run: Run) -> None:
    session.delete(run)
    session.commit()


def create_artifact(
    session: Session,
    *,
    run_id: UUID,
    artifact_type: str,
    storage_path: str,
) -> Artifact:
    artifact = Artifact(
        run_id=run_id,
        type=artifact_type,
        storage_path=storage_path,
    )
    session.add(artifact)
    session.commit()
    session.refresh(artifact)
    return artifact


def list_artifacts(session: Session, run_id: UUID) -> list[Artifact]:
    statement = (
        select(Artifact)
        .where(Artifact.run_id == run_id)
        .order_by(Artifact.created_at, Artifact.id)
    )
    return list(session.scalars(statement))


def create_message(
    session: Session,
    *,
    run_id: UUID,
    role: str,
    content: str,
) -> Message:
    message = Message(run_id=run_id, role=role, content=content)
    session.add(message)
    session.commit()
    session.refresh(message)
    return message


def list_messages(session: Session, run_id: UUID) -> list[Message]:
    statement = (
        select(Message)
        .where(Message.run_id == run_id)
        .order_by(Message.created_at, Message.id)
    )
    return list(session.scalars(statement))


def list_run_artifacts(session: Session, run_id: UUID) -> list[Artifact]:
    statement = (
        select(Artifact)
        .where(Artifact.run_id == run_id)
        .order_by(Artifact.created_at, Artifact.id)
    )
    return list(session.scalars(statement))
