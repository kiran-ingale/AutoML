from uuid import UUID

import pandas as pd
from pandas.errors import EmptyDataError, ParserError
from sqlalchemy.orm import Session

from backend.app.models import Run
from backend.app.pipeline.preprocessing import (
    PreparedDataset,
    PreprocessingError,
    TaskType,
    prepare_dataset,
)
from backend.app.services.datasets import dataset_path_for_run


class DatasetRunNotFoundError(LookupError):
    """Raised when the requested dataset run does not exist."""


class StoredDatasetNotFoundError(FileNotFoundError):
    """Raised when a run's stored dataset file is missing."""


def prepare_run_dataset(
    session: Session,
    *,
    run_id: UUID,
    target_column: str,
    task_type: TaskType,
    test_size: float = 0.2,
    random_state: int = 42,
) -> PreparedDataset:
    run = session.get(Run, run_id)
    if run is None:
        raise DatasetRunNotFoundError(f"Run {run_id} was not found.")
    if not run.dataset_pointer:
        raise PreprocessingError(f"Run {run_id} has no stored dataset.")

    dataset_path = dataset_path_for_run(run.id, run.dataset_pointer)
    if not dataset_path.is_file():
        raise StoredDatasetNotFoundError(f"Stored dataset for run {run_id} is missing.")

    try:
        dataframe = pd.read_csv(dataset_path, low_memory=False)
    except EmptyDataError as exc:
        raise PreprocessingError("Stored CSV contains no tabular data.") from exc
    except (ParserError, UnicodeError, ValueError) as exc:
        raise PreprocessingError(
            "Stored CSV could not be parsed; verify the stored dataset."
        ) from exc

    prepared = prepare_dataset(
        dataframe,
        target_column=target_column,
        task_type=task_type,
        test_size=test_size,
        random_state=random_state,
    )
    run.preprocessing = prepared.summary.model_dump(mode="json")
    session.commit()
    return prepared
