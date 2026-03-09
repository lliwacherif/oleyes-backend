"""
Drop and recreate the user_contexts table with new columns.

Usage:
    cd backend
    python scripts/migrate_context.py
"""

import asyncio
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import app.models  # noqa: F401
from app.core.database import engine, Base
from app.models.user_context import UserContext


async def main() -> None:
    async with engine.begin() as conn:
        await conn.run_sync(
            lambda sync_conn: UserContext.__table__.drop(sync_conn, checkfirst=True)
        )
        print("Dropped user_contexts table (if existed).")
        await conn.run_sync(Base.metadata.create_all)
        print("Recreated all tables. Done.")
    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
