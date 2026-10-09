import logging
from pathlib import Path
from uuid import UUID, uuid4

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from backend.app.core.config import get_settings
from backend.app.models import Run
from backend.app.pipeline.profiling import profile_csv

logger = logging.getLogger(__name__)


def create_dataset_run(
    session: Session,
    *,
    contents: bytes,
    raw_requirements_text: str,
) -> Run:
    profile = profile_csv(contents)
    run = Run(
        id=uuid4(),
        raw_requirements_text=raw_requirements_text,
        profile=profile,
    )
    storage_root = get_settings().storage_dir.resolve()
    relative_path = Path("runs") / str(run.id) / "dataset.csv"
    stored_path = storage_root / relative_path

    try:
        stored_path.parent.mkdir(parents=True, exist_ok=True)
        stored_path.write_bytes(contents)
        run.dataset_pointer = relative_path.as_posix()
        session.add(run)
        session.commit()
        session.refresh(run)
    except OSError:
        session.rollback()
        stored_path.unlink(missing_ok=True)
        logger.exception("Failed to store uploaded dataset for run %s", run.id)
        raise
    except SQLAlchemyError:
        session.rollback()
        stored_path.unlink(missing_ok=True)
        logger.exception("Failed to persist dataset run %s", run.id)
        raise

    logger.info(
        "Created dataset run %s (%d rows, %d columns)",
        run.id,
        profile["rows"],
        profile["columns_count"],
    )
    return run


def dataset_path_for_run(run_id: UUID, relative_pointer: str) -> Path:
    storage_root = get_settings().storage_dir.resolve()
    candidate = (storage_root / relative_pointer).resolve()
    expected_parent = (storage_root / "runs" / str(run_id)).resolve()
    if candidate.parent != expected_parent or candidate.name != "dataset.csv":
        raise ValueError("Stored dataset pointer is outside its run directory.")
    return candidate
