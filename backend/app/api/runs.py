import logging
from typing import Annotated
from uuid import UUID

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Form,
    HTTPException,
    UploadFile,
    status,
)
from pydantic import ValidationError
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from backend.app.core.config import get_settings
from backend.app.core.db import get_db, get_session_factory
from backend.app.pipeline.profiling import DatasetValidationError
from backend.app.repositories import get_run, list_run_artifacts
from backend.app.schemas.run import RunRead
from backend.app.schemas.run_artifact import RunArtifactRead
from backend.app.schemas.training import TrainingRequest
from backend.app.services.datasets import create_dataset_run
from backend.app.services.training import (
    TrainingAlreadyStartedError,
    TrainingDataError,
    TrainingRunNotFoundError,
    queue_training,
    train_run_background,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/runs", tags=["runs"])


@router.post("", response_model=RunRead, status_code=status.HTTP_201_CREATED)
async def create_run_from_upload(
    file: Annotated[UploadFile, File()],
    description: Annotated[str, Form(min_length=1, max_length=10_000)],
    session: Annotated[Session, Depends(get_db)],
) -> RunRead:
    try:
        description = description.strip()
        if not description:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail="Description cannot be blank.",
            )

        if not file.filename or not file.filename.lower().endswith(".csv"):
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail="Upload a file with a .csv extension.",
            )

        maximum_bytes = get_settings().max_upload_bytes
        contents = await file.read(maximum_bytes + 1)
        if len(contents) > maximum_bytes:
            raise HTTPException(
                status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                detail=f"CSV exceeds the maximum upload size of {maximum_bytes} bytes.",
            )

        try:
            run = create_dataset_run(
                session,
                contents=contents,
                raw_requirements_text=description,
            )
        except DatasetValidationError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=str(exc),
            ) from exc
        except OSError as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Dataset could not be stored.",
            ) from exc
        except SQLAlchemyError as exc:
            logger.exception("Failed to create dataset run")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Dataset run could not be created.",
            ) from exc

        return RunRead.model_validate(run)
    except ValidationError as exc:
        logger.exception("Generated dataset profile failed response validation")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Dataset profile could not be returned.",
        ) from exc
    finally:
        await file.close()


@router.get("/{run_id}", response_model=RunRead)
def read_run(
    run_id: UUID,
    session: Annotated[Session, Depends(get_db)],
) -> RunRead:
    run = get_run(session, run_id)
    if run is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Run was not found.",
        )
    return RunRead.model_validate(run)


@router.get("/{run_id}/artifacts", response_model=list[RunArtifactRead])
def read_run_artifacts(
    run_id: UUID,
    session: Annotated[Session, Depends(get_db)],
) -> list[RunArtifactRead]:
    if get_run(session, run_id) is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Run was not found.",
        )
    return [
        RunArtifactRead.model_validate(artifact)
        for artifact in list_run_artifacts(session, run_id)
    ]


@router.post(
    "/{run_id}/train",
    response_model=RunRead,
    status_code=status.HTTP_202_ACCEPTED,
)
def start_training(
    run_id: UUID,
    request: TrainingRequest,
    background_tasks: BackgroundTasks,
    session: Annotated[Session, Depends(get_db)],
) -> RunRead:
    try:
        run = queue_training(
            session,
            run_id=run_id,
            request=request,
        )
    except TrainingRunNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Run was not found.",
        ) from exc
    except TrainingDataError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(exc),
        ) from exc
    except TrainingAlreadyStartedError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc

    background_tasks.add_task(
        train_run_background,
        get_session_factory(),
        run_id=run_id,
        config=run.training_config,
    )
    return RunRead.model_validate(run)
