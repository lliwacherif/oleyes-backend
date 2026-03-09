"""
Async PostgreSQL engine, session factory, and declarative Base.

Usage in endpoints:
    from app.core.database import get_db
    @router.get("/example")
    async def example(db: AsyncSession = Depends(get_db)):
        ...
"""

import logging
from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from app.core import config

logger = logging.getLogger("oleyes.database")

# ── Engine ────────────────────────────────────────────────────────────
engine = create_async_engine(
    config.DATABASE_URL,
    echo=False,
    pool_size=5,
    max_overflow=10,
)

# ── Session factory ──────────────────────────────────────────────────
async_session = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


# ── Declarative Base ─────────────────────────────────────────────────
class Base(DeclarativeBase):
    """Base class for all ORM models."""
    pass


# ── Dependency ───────────────────────────────────────────────────────
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency that yields a DB session and closes it after."""
    async with async_session() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


# ── Lifecycle helpers (called from main.py) ──────────────────────────
async def init_db() -> None:
    """Create all tables from registered models (if any)."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database tables synced.")


async def close_db() -> None:
    """Dispose of the connection pool."""
    await engine.dispose()
    logger.info("Database connection pool closed.")
