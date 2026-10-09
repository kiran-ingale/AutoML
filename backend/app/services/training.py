import logging
import shutil
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from backend.app.core.config import get_settings
from backend.app.models import Artifact, Run
from backend.app.pipeline.gbdt_path import TrainingError, train_autogluon
from backend.app.pipeline.preprocessing import PreprocessingError
from backend.app.repositories import get_run
from backend.app.schemas.training import TrainingRequest
from backend.app.services.datasets import dataset_path_for_run
from backend.app.services.preprocessing import prepare_run_dataset

logger = logging.getLogger(__name__)


class TrainingRunNotFoundError(LookupError):
    """Raised when the requested run does not exist."""


class TrainingAlreadyStartedError(RuntimeError):
    """Raised when a run already has an active or completed training attempt."""


class TrainingDataError(ValueError):
    """Raised when a run does not contain a usable dataset."""


def queue_training(
    session: Session,
    *,
    run_id: UUID,
    request: TrainingRequest,
) -> Run:
    run = get_run(session, run_id)
    if run is None:
        raise TrainingRunNotFoundError(f"Run {run_id} was not found.")
    if run.status in {"queued", "running", "succeeded"} and run.training_config:
        raise TrainingAlreadyStartedError(
            f"Run {run_id} already has a queued, active, or completed training attempt."
        )
    if run.status == "running" or run.status == "succeeded":
        raise TrainingAlreadyStartedError(
            f"Run {run_id} cannot be trained while its status is {run.status!r}."
        )
    if not run.dataset_pointer:
        raise TrainingDataError("Run has no stored dataset.")
    dataset_path = dataset_path_for_run(run.id, run.dataset_pointer)
    if not dataset_path.is_file():
        raise TrainingDataError("Stored dataset file is missing.")

    run.status = "queued"
    run.training_config = {
        **request.model_dump(mode="json"),
        "attempt_id": str(uuid4()),
        "engine": "autogluon.tabular",
        "presets": "good_quality",
    }
    run.leaderboard = None
    run.error_stage = None
    run.error_message = None
    session.commit()
    session.refresh(run)
    return run


def _update_run_failure(
    session_factory: sessionmaker[Session],
    *,
    run_id: UUID,
    stage: str,
    message: str,
) -> None:
    with session_factory() as session:
        run = get_run(session, run_id)
        if run is None:
            logger.error("Cannot persist failure: run %s no longer exists", run_id)
            return
        run.status = "failed"
        run.error_stage = stage
        run.error_message = message
        session.commit()


def _training_paths(run_id: UUID, attempt_id: str) -> tuple[Path, str]:
    storage_root = get_settings().storage_dir.resolve()
    relative_path = (Path("runs") / str(run_id) / "attempts" / attempt_id).as_posix()
    return storage_root / Path(relative_path), relative_path


def train_run_background(
    session_factory: sessionmaker[Session],
    *,
    run_id: UUID,
    config: dict[str, Any],
) -> None:
    attempt_id = str(config["attempt_id"])
    output_directory, relative_directory = _training_paths(run_id, attempt_id)
    with session_factory() as session:
        run = get_run(session, run_id)
        if run is None:
            logger.error("Cannot train run %s because it no longer exists", run_id)
            return
        if run.status != "queued" or run.training_config != config:
            logger.warning("Skipping stale training task for run %s", run_id)
            return
        run.status = "running"
        run.error_stage = None
        run.error_message = None
        session.commit()

    current_stage = "preprocessing"
    try:
        with session_factory() as session:
            prepared = prepare_run_dataset(
                session,
                run_id=run_id,
                target_column=str(config["target_column"]),
                task_type=config["task_type"],
                test_size=float(config["test_size"]),
                random_state=int(config["random_state"]),
            )

        current_stage = "training"
        result = train_autogluon(
            prepared=prepared,
            target_column=str(config["target_column"]),
            task_type=config["task_type"],
            output_directory=output_directory,
            time_limit_seconds=int(config["time_limit_seconds"]),
            evaluation_metric=config.get("evaluation_metric"),
        )

        leaderboard_relative = f"{relative_directory}/leaderboard.csv"
        model_relative = f"{relative_directory}/model"
        with session_factory() as session:
            run = get_run(session, run_id)
            if run is None:
                raise TrainingError(f"Run {run_id} was deleted during training.")
            run.leaderboard = result.leaderboard
            run.route = {
                "path": "gbdt",
                "engine": "autogluon.tabular",
                "best_model": result.best_model,
                "evaluation_metric": result.evaluation_metric,
            }
            run.status = "succeeded"
            run.error_stage = None
            run.error_message = None
            session.add_all(
                [
                    Artifact(
                        run_id=run_id,
                        type="model",
                        storage_path=model_relative,
                    ),
                    Artifact(
                        run_id=run_id,
                        type="leaderboard",
                        storage_path=leaderboard_relative,
                    ),
                ]
            )
            session.commit()
        logger.info(
            "Training run %s succeeded with model %s", run_id, result.best_model
        )
    except (PreprocessingError, FileNotFoundError, TrainingError) as exc:
        if isinstance(exc, TrainingError):
            current_stage = "training"
        logger.error("Training run %s failed during %s: %s", run_id, current_stage, exc)
        _update_run_failure(
            session_factory,
            run_id=run_id,
            stage=current_stage,
            message=str(exc),
        )
    except SQLAlchemyError as exc:
        logger.exception("Database operation failed during training run %s", run_id)
        try:
            _update_run_failure(
                session_factory,
                run_id=run_id,
                stage="persistence",
                message=f"Training result could not be saved ({type(exc).__name__}).",
            )
        except SQLAlchemyError:
            logger.exception("Could not persist database failure for run %s", run_id)
    except Exception as exc:
        logger.exception("Unexpected training failure for run %s", run_id)
        _update_run_failure(
            session_factory,
            run_id=run_id,
            stage=current_stage,
            message=f"Training failed unexpectedly ({type(exc).__name__}).",
        )
    finally:
        if output_directory.exists():
            with session_factory() as session:
                run = get_run(session, run_id)
                if run is not None and run.status != "succeeded":
                    try:
                        shutil.rmtree(output_directory)
                    except OSError:
                        logger.exception(
                            "Could not remove incomplete model artifacts for run %s",
                            run_id,
                        )
