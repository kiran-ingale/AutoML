from collections.abc import Generator
from pathlib import Path
from uuid import UUID, uuid4

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from backend.app.api import runs as runs_api
from backend.app.core.config import get_settings
from backend.app.core.db import get_db
from backend.app.main import app
from backend.app.models import Artifact, Base, Run
from backend.app.pipeline.gbdt_path import TrainingError, TrainingResult
from backend.app.schemas.training import TrainingRequest
from backend.app.services import training as training_service
from backend.app.services.training import (
    TrainingAlreadyStartedError,
    queue_training,
    train_run_background,
)


@pytest.fixture
def database_and_storage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Generator[tuple[sessionmaker[Session], Path], None, None]:
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    @event.listens_for(engine, "connect")
    def enable_foreign_keys(connection, record) -> None:
        del record
        cursor = connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine, expire_on_commit=False)
    storage_dir = tmp_path / "storage"
    monkeypatch.setattr(get_settings(), "storage_dir", storage_dir)
    try:
        yield session_factory, storage_dir
    finally:
        engine.dispose()


def create_run_with_dataset(
    session_factory: sessionmaker[Session],
    storage_dir: Path,
) -> UUID:
    run_id = uuid4()
    relative_pointer = f"runs/{run_id}/dataset.csv"
    dataset_path = storage_dir / relative_pointer
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "number": list(range(20)),
            "category": ["a", "b"] * 10,
            "target": ["yes", "no"] * 10,
        }
    ).to_csv(dataset_path, index=False)
    with session_factory() as session:
        session.add(
            Run(
                id=run_id,
                dataset_pointer=relative_pointer,
                status="queued",
            )
        )
        session.commit()
    return run_id


def valid_request() -> TrainingRequest:
    return TrainingRequest(
        target_column="target",
        task_type="classification",
        time_limit_seconds=1,
    )


def test_queue_training_persists_configuration_and_rejects_duplicate(
    database_and_storage: tuple[sessionmaker[Session], Path],
) -> None:
    session_factory, storage_dir = database_and_storage
    run_id = create_run_with_dataset(session_factory, storage_dir)

    with session_factory() as session:
        queued = queue_training(session, run_id=run_id, request=valid_request())
        assert queued.status == "queued"
        assert queued.training_config["engine"] == "autogluon.tabular"
        assert queued.training_config["target_column"] == "target"

        with pytest.raises(TrainingAlreadyStartedError):
            queue_training(session, run_id=run_id, request=valid_request())


def test_background_training_persists_success_and_artifacts(
    database_and_storage: tuple[sessionmaker[Session], Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_factory, storage_dir = database_and_storage
    run_id = create_run_with_dataset(session_factory, storage_dir)
    with session_factory() as session:
        run = queue_training(session, run_id=run_id, request=valid_request())
        config = dict(run.training_config)

    def fake_train_autogluon(**kwargs) -> TrainingResult:
        model_path = kwargs["output_directory"] / "model"
        model_path.mkdir(parents=True)
        (model_path / "predictor.txt").write_text("mock model", encoding="utf-8")
        leaderboard_path = kwargs["output_directory"] / "leaderboard.csv"
        leaderboard_path.write_text(
            "model,score_test\nMockModel,0.9\n", encoding="utf-8"
        )
        return TrainingResult(
            model_path=model_path,
            leaderboard_path=leaderboard_path,
            leaderboard=[{"model": "MockModel", "score_test": 0.9}],
            best_model="MockModel",
            evaluation_metric="accuracy",
        )

    monkeypatch.setattr(training_service, "train_autogluon", fake_train_autogluon)
    train_run_background(session_factory, run_id=run_id, config=config)

    with session_factory() as session:
        run = session.get(Run, run_id)
        assert run is not None
        assert run.status == "succeeded"
        assert run.route == {
            "path": "gbdt",
            "engine": "autogluon.tabular",
            "best_model": "MockModel",
            "evaluation_metric": "accuracy",
        }
        assert run.leaderboard == [{"model": "MockModel", "score_test": 0.9}]
        assert run.preprocessing["target_column"] == "target"
        artifacts = session.query(Artifact).filter_by(run_id=run_id).all()
        assert {artifact.type for artifact in artifacts} == {"model", "leaderboard"}
        assert all(
            (storage_dir / artifact.storage_path).exists() for artifact in artifacts
        )


def test_background_training_failure_is_persisted_and_retryable(
    database_and_storage: tuple[sessionmaker[Session], Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_factory, storage_dir = database_and_storage
    run_id = create_run_with_dataset(session_factory, storage_dir)
    with session_factory() as session:
        run = queue_training(session, run_id=run_id, request=valid_request())
        config = dict(run.training_config)

    def fail_training(**kwargs) -> TrainingResult:
        del kwargs
        raise TrainingError("Training failed safely.")

    monkeypatch.setattr(training_service, "train_autogluon", fail_training)

    train_run_background(session_factory, run_id=run_id, config=config)

    with session_factory() as session:
        run = session.get(Run, run_id)
        assert run is not None
        assert run.status == "failed"
        assert run.error_stage == "training"
        assert run.error_message == "Training failed safely."
        retried = queue_training(session, run_id=run_id, request=valid_request())
        assert retried.status == "queued"
        assert retried.training_config["attempt_id"] != config["attempt_id"]


def test_training_endpoints_queue_run_and_return_artifacts(
    database_and_storage: tuple[sessionmaker[Session], Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_factory, storage_dir = database_and_storage
    run_id = create_run_with_dataset(session_factory, storage_dir)

    def override_get_db() -> Generator[Session, None, None]:
        with session_factory() as session:
            yield session

    def skip_background_task(*args, **kwargs) -> None:
        del args, kwargs

    monkeypatch.setattr(runs_api, "get_session_factory", lambda: session_factory)
    monkeypatch.setattr(runs_api, "train_run_background", skip_background_task)
    app.dependency_overrides[get_db] = override_get_db
    try:
        client = TestClient(app)
        response = client.post(
            f"/runs/{run_id}/train",
            json={
                "target_column": "target",
                "task_type": "classification",
                "time_limit_seconds": 1,
            },
        )
        assert response.status_code == 202
        assert response.json()["status"] == "queued"
        assert (
            client.get(f"/runs/{run_id}").json()["training_config"]["target_column"]
            == "target"
        )
        assert client.get(f"/runs/{run_id}/artifacts").json() == []
        assert (
            client.post(
                f"/runs/{run_id}/train",
                json={"target_column": "target", "task_type": "classification"},
            ).status_code
            == 409
        )
    finally:
        app.dependency_overrides.clear()


def test_training_request_rejects_metric_for_wrong_task() -> None:
    response = TestClient(app).post(
        f"/runs/{uuid4()}/train",
        json={
            "target_column": "target",
            "task_type": "classification",
            "evaluation_metric": "root_mean_squared_error",
        },
    )

    assert response.status_code == 422
