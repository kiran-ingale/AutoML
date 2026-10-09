from fastapi.testclient import TestClient
from sqlalchemy.exc import OperationalError

from backend.app.api import health
from backend.app.main import app

client = TestClient(app)


def test_health_check_returns_ok() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_readiness_check_returns_ok_when_database_is_available(
    monkeypatch,
) -> None:
    monkeypatch.setattr(health, "check_database_connection", lambda: None)

    response = client.get("/ready")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_readiness_check_returns_503_when_database_is_unavailable(
    monkeypatch,
) -> None:
    def raise_connection_error() -> None:
        raise OperationalError("SELECT 1", {}, RuntimeError("connection refused"))

    monkeypatch.setattr(health, "check_database_connection", raise_connection_error)

    response = client.get("/ready")

    assert response.status_code == 503
    assert response.json() == {"detail": "Database is not ready"}
