from collections.abc import Generator
from pathlib import Path
from uuid import UUID

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from backend.app.core.config import get_settings
from backend.app.core.db import get_db
from backend.app.main import app
from backend.app.models import Base, Run
from backend.app.services.datasets import dataset_path_for_run


@pytest.fixture
def client_and_storage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Generator[tuple[TestClient, Path, sessionmaker[Session]], None, None]:
    database_engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(database_engine)
    session_factory = sessionmaker(
        bind=database_engine,
        expire_on_commit=False,
    )
    settings = get_settings()
    monkeypatch.setattr(settings, "storage_dir", tmp_path / "storage")
    monkeypatch.setattr(settings, "max_upload_bytes", 1024)

    def override_get_db() -> Generator[Session, None, None]:
        with session_factory() as session:
            yield session

    app.dependency_overrides[get_db] = override_get_db
    try:
        yield TestClient(app), settings.storage_dir, session_factory
    finally:
        app.dependency_overrides.clear()
        database_engine.dispose()


def test_upload_creates_run_stores_csv_and_returns_profile(
    client_and_storage: tuple[TestClient, Path, sessionmaker[Session]],
) -> None:
    client, storage_dir, session_factory = client_and_storage
    response = client.post(
        "/runs",
        data={"description": "Predict customer renewal."},
        files={
            "file": (
                "customers.csv",
                b"age,plan,renewed\n30,basic,yes\n,pro,no\n30,basic,yes\n",
                "text/csv",
            )
        },
    )

    assert response.status_code == 201
    result = response.json()
    assert result["status"] == "queued"
    assert result["raw_requirements_text"] == "Predict customer renewal."
    assert result["profile"]["rows"] == 3
    assert result["profile"]["columns_count"] == 3
    assert result["profile"]["target_column_guess"] == "renewed"
    assert result["profile"]["class_imbalance_ratio"] == 2.0
    assert result["profile"]["duplicate_rows"] == 1
    assert result["profile"]["columns"][0]["missing_count"] == 1

    with session_factory() as session:
        run = session.get(Run, UUID(result["id"]))
        assert run is not None
        stored_path = dataset_path_for_run(run.id, run.dataset_pointer)
        assert stored_path.is_relative_to(storage_dir)
        assert stored_path.is_file()
        assert stored_path.read_bytes().startswith(b"age,plan,renewed")


@pytest.mark.parametrize(
    ("filename", "contents", "expected_status", "expected_detail"),
    [
        ("customers.txt", b"a,b\n1,2\n", 415, "Upload a file with a .csv extension."),
        ("empty.csv", b"", 422, "CSV file is empty."),
        (
            "duplicates.csv",
            b"a,a\n1,2\n",
            422,
            "CSV contains duplicate column headers: 'a'",
        ),
        ("header.csv", b"a,b\n", 422, "CSV must contain at least one data row."),
        (
            "encoding.csv",
            b"\xff,\xfe",
            422,
            "CSV must use UTF-8 or UTF-8 with BOM encoding.",
        ),
    ],
)
def test_upload_rejects_invalid_csv(
    client_and_storage: tuple[TestClient, Path, sessionmaker[Session]],
    filename: str,
    contents: bytes,
    expected_status: int,
    expected_detail: str,
) -> None:
    client, _, session_factory = client_and_storage
    response = client.post(
        "/runs",
        data={"description": "Test invalid input."},
        files={"file": (filename, contents, "text/csv")},
    )

    assert response.status_code == expected_status
    assert expected_detail in response.json()["detail"]
    with session_factory() as session:
        assert session.query(Run).count() == 0


def test_upload_rejects_files_over_configured_limit(
    client_and_storage: tuple[TestClient, Path, sessionmaker[Session]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, storage_dir, session_factory = client_and_storage
    monkeypatch.setattr(get_settings(), "max_upload_bytes", 8)
    response = client.post(
        "/runs",
        data={"description": "Check file size limit."},
        files={"file": ("large.csv", b"a,b\n123,456\n", "text/csv")},
    )

    assert response.status_code == 413
    assert not (storage_dir / "runs").exists()
    with session_factory() as session:
        assert session.query(Run).count() == 0


def test_upload_rejects_blank_description(
    client_and_storage: tuple[TestClient, Path, sessionmaker[Session]],
) -> None:
    client, _, session_factory = client_and_storage
    response = client.post(
        "/runs",
        data={"description": "   "},
        files={"file": ("data.csv", b"a,b\n1,2\n", "text/csv")},
    )

    assert response.status_code == 422
    assert response.json()["detail"] == "Description cannot be blank."
    with session_factory() as session:
        assert session.query(Run).count() == 0
