from collections.abc import Generator
from pathlib import Path
from typing import Literal
from uuid import uuid4

import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from sklearn.model_selection import train_test_split
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from backend.app.core.config import get_settings
from backend.app.models import Base, Run
from backend.app.pipeline.preprocessing import PreprocessingError, prepare_dataset
from backend.app.schemas.run import RunRead
from backend.app.services.preprocessing import (
    DatasetRunNotFoundError,
    StoredDatasetNotFoundError,
    prepare_run_dataset,
)


@pytest.fixture
def session_and_storage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Generator[tuple[Session, Path], None, None]:
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine, expire_on_commit=False)
    storage_dir = tmp_path / "storage"
    monkeypatch.setattr(get_settings(), "storage_dir", storage_dir)
    with session_factory() as session:
        try:
            yield session, storage_dir
        finally:
            session.rollback()
            engine.dispose()


def make_classification_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [float(value) if value % 4 else np.nan for value in range(20)],
            "category": ["a", "b"] * 10,
            "constant": ["same"] * 20,
            "target": ["yes", "no"] * 10,
        }
    )


def test_prepare_mixed_classification_data_and_report_transforms() -> None:
    prepared = prepare_dataset(
        make_classification_data(),
        target_column="target",
        task_type="classification",
    )

    assert sparse.issparse(prepared.X_train)
    assert sparse.issparse(prepared.X_validation)
    assert prepared.X_train.shape[0] == len(prepared.y_train) == 16
    assert prepared.X_validation.shape[0] == len(prepared.y_validation) == 4
    assert prepared.X_train.shape[1] == prepared.X_validation.shape[1]
    assert prepared.summary.split_method == "stratified"
    assert prepared.summary.numeric_columns == ["age"]
    assert prepared.summary.categorical_columns == ["category"]
    assert prepared.summary.dropped_constant_columns == ["constant"]
    assert prepared.summary.missing_feature_values_before_imputation == 5
    assert np.isfinite(prepared.X_train.data).all()
    assert {prepared.y_train.value_counts()[key] for key in ("yes", "no")} == {8}
    assert {prepared.y_validation.value_counts()[key] for key in ("yes", "no")} == {2}


def test_preprocessing_fit_uses_only_training_rows_and_ignores_new_categories() -> None:
    dataframe = pd.DataFrame(
        {
            "number": list(range(20)),
            "category": ["known"] * 20,
            "target": list(range(20)),
        }
    )
    train_indices, validation_indices = train_test_split(
        dataframe.index,
        test_size=0.2,
        random_state=3,
    )
    dataframe.loc[validation_indices[0], "category"] = "validation-only"

    prepared = prepare_dataset(
        dataframe,
        target_column="target",
        task_type="regression",
        random_state=3,
    )

    learned_median = (
        prepared.transformer.named_transformers_["numeric"]
        .named_steps["imputer"]
        .statistics_[0]
    )
    expected_training_median = dataframe.loc[train_indices, "number"].median()
    assert learned_median == expected_training_median
    assert prepared.X_train.shape[1] == prepared.X_validation.shape[1]
    assert prepared.summary.split_method == "random"
    assert len(prepared.summary.warnings) >= 1


def test_missing_targets_are_removed_and_recorded() -> None:
    dataframe = make_classification_data()
    dataframe.loc[[0, 1], "target"] = None

    prepared = prepare_dataset(
        dataframe,
        target_column="target",
        task_type="classification",
    )

    assert prepared.summary.rows_without_target == 2
    assert len(prepared.y_train) + len(prepared.y_validation) == 18
    assert any("Dropped 2 row(s)" in warning for warning in prepared.summary.warnings)


@pytest.mark.parametrize(
    ("dataframe", "target_column", "task_type", "message"),
    [
        (pd.DataFrame(), "target", "classification", "at least one row"),
        (make_classification_data(), "absent", "classification", "was not found"),
        (
            pd.DataFrame({"feature": [1, 2], "target": [1, 1]}),
            "target",
            "classification",
            "at least two distinct values",
        ),
        (
            pd.DataFrame({"feature": [1, 2], "target": ["one", "two"]}),
            "target",
            "regression",
            "must be numeric",
        ),
        (
            pd.DataFrame({"feature": [1, 2], "target": [1.0, np.inf]}),
            "target",
            "regression",
            "must be finite",
        ),
        (
            pd.DataFrame({"constant": [1] * 10, "target": [0, 1] * 5}),
            "target",
            "classification",
            "No usable feature columns",
        ),
        (
            pd.DataFrame({"feature": [1, 2, 3], "target": ["a", "a", "b"]}),
            "target",
            "classification",
            "cannot preserve every class",
        ),
    ],
)
def test_invalid_training_data_has_actionable_errors(
    dataframe: pd.DataFrame,
    target_column: str,
    task_type: Literal["classification", "regression"],
    message: str,
) -> None:
    with pytest.raises(PreprocessingError, match=message):
        prepare_dataset(
            dataframe,
            target_column=target_column,
            task_type=task_type,
        )


def test_prepare_run_dataset_loads_csv_and_persists_summary(
    session_and_storage: tuple[Session, Path],
) -> None:
    session, storage_dir = session_and_storage
    run = Run(
        id=uuid4(),
        profile=None,
    )
    run.dataset_pointer = f"runs/{run.id}/dataset.csv"
    dataset_path = storage_dir / run.dataset_pointer
    dataset_path.parent.mkdir(parents=True)
    make_classification_data().to_csv(dataset_path, index=False)
    session.add(run)
    session.commit()

    prepared = prepare_run_dataset(
        session,
        run_id=run.id,
        target_column="target",
        task_type="classification",
    )

    assert prepared.summary.target_column == "target"
    session.refresh(run)
    assert run.preprocessing == prepared.summary.model_dump(mode="json")
    assert RunRead.model_validate(run).preprocessing == prepared.summary


def test_prepare_run_dataset_reports_missing_run_and_file(
    session_and_storage: tuple[Session, Path],
) -> None:
    session, _ = session_and_storage
    run_id = uuid4()
    with pytest.raises(DatasetRunNotFoundError, match="was not found"):
        prepare_run_dataset(
            session,
            run_id=run_id,
            target_column="target",
            task_type="classification",
        )

    run = Run(id=run_id, dataset_pointer=f"runs/{run_id}/dataset.csv")
    session.add(run)
    session.commit()

    with pytest.raises(StoredDatasetNotFoundError, match="is missing"):
        prepare_run_dataset(
            session,
            run_id=run_id,
            target_column="target",
            task_type="classification",
        )
