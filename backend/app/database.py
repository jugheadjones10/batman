"""Database connection and session management."""

from collections.abc import AsyncGenerator
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import declarative_base

Base = declarative_base()


def get_database_url(project_path: Path) -> str:
    """Get SQLite database URL for a project."""
    db_path = project_path / "project.sqlite"
    return f"sqlite+aiosqlite:///{db_path}"


class DatabaseManager:
    """Manages database connections for projects."""

    def __init__(self):
        self._engines: dict[str, any] = {}
        self._session_factories: dict[str, async_sessionmaker] = {}

    async def get_session(self, project_path: Path) -> AsyncGenerator[AsyncSession, None]:
        """Get a database session for a specific project."""
        path_str = str(project_path)

        if path_str not in self._engines:
            url = get_database_url(project_path)
            engine = create_async_engine(url, echo=False)
            self._engines[path_str] = engine
            self._session_factories[path_str] = async_sessionmaker(
                engine, class_=AsyncSession, expire_on_commit=False
            )

            # Create tables
            async with engine.begin() as conn:
                from backend.app.db.models import Base as DBBase

                await conn.run_sync(DBBase.metadata.create_all)

        async with self._session_factories[path_str]() as session:
            yield session

    async def close_all(self):
        """Close all database connections."""
        for engine in self._engines.values():
            await engine.dispose()
        self._engines.clear()
        self._session_factories.clear()


db_manager = DatabaseManager()
