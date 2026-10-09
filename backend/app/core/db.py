from collections.abc import Generator
from functools import lru_cache

from sqlalchemy import Engine, create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from backend.app.core.config import get_settings


class DatabaseNotConfiguredError(RuntimeError):
    """Raised when database access is requested without a configured URL."""


def _database_url() -> str:
    database_url = get_settings().database_url
    if not database_url:
        raise DatabaseNotConfiguredError(
            "Set DATABASE_URL in the environment or project .env file."
        )
    return database_url


@lru_cache
def get_engine(database_url: str) -> Engine:
    return create_engine(database_url, pool_pre_ping=True)


def get_database_engine() -> Engine:
    return get_engine(_database_url())


def check_database_connection() -> None:
    with get_database_engine().connect() as connection:
        connection.execute(text("SELECT 1"))


def get_db() -> Generator[Session, None, None]:
    session_factory = sessionmaker(
        bind=get_database_engine(),
        autoflush=False,
        autocommit=False,
    )
    with session_factory() as session:
        yield session
